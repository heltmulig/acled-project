"""
STK-INF4000 ML-project ACLED dataset module.
"""

import datetime
import time
import io
import sys
import os.path

import logging
import numpy
import pandas
import pymongo
import requests


# Local file cache
DATADIR = 'data/'


# Utility function
def ymd(datestr):
    """
    Utility function to convert a string-formatted date to a datetime object.
    Used when importing date fields from csv to mongodb.
    """
    return datetime.datetime.strptime(datestr, "%Y-%m-%d")


# ACLED API
###########
class ACLED:
    """
    Class used to provide a consitently updated ACLED dataset.
    """

    ACLED_COLUMN_DTYPES = {
        'data_id'           : numpy.int64,
        'gwno'              : numpy.int64,
        'event_id_cnty'     : str,
        'event_id_no_cnty'  : str,
        'event_date'        : str,              # datetime.datetime
        'year'              : numpy.int64,
        'time_precision'    : numpy.int64,
        'event_type'        : str,
        'actor1'            : str,
        'ally_actor_1'      : str,
        'inter1'            : numpy.int64,
        'actor2'            : str,
        'ally_actor_2'      : str,
        'inter2'            : numpy.int64,
        'interaction'       : numpy.int64,
        'country'           : str,
        'admin1'            : str,
        'admin2'            : str,
        'admin3'            : str,
        'location'          : str,
        'latitude'          : numpy.float64,
        'longitude'         : numpy.float64,
        'geo_precision'     : numpy.int64,
        'source'            : str,
        'notes'             : str,
        'fatalities'        : numpy.int64,
    }

    URL_ACLED_READ = 'http://acleddata.com/api/acled/read.csv'

    def __init__(self, db_name='acled', collection_name='api',
                 mongodb_host='127.0.0.1', mongodb_port=27017):
        self.db_name = db_name
        self.collection_name = collection_name
        self.mongodb_host = mongodb_host
        self.mongodb_dbport = mongodb_port

        self.logger = logging.getLogger(self.__class__.__name__)

        self.conn = pymongo.MongoClient(mongodb_host, mongodb_port)
        self.db = self.conn[self.db_name]
        self.c = self.db[self.collection_name]

        self.acled_download_database()

        # add geo_point fields to database, if necessary
        #try:
        #    self.c.find({'geo_point': { '$exists': True }}).next()
        #except StopIteration:
        #    print('Appending geo_points to existing database...')
        #    df = self.mongodb_get_entire_database()
        #    #self.append_geo_points(df)
        #    self.c.drop()
        #    self.pandas_df_to_mongodb(df)


    def acled_download_database(self):
        """
        Download database from ACLED server if it does not exist or is empty.
        """
        if self.db_name not in self.conn.database_names() \
            or self.collection_name not in self.db.collection_names() \
            or self.c.count() == 0:
            print("Database '%s' does not exist or is empty." % self.db_name)
            ans = input("Grab it from ACLED server (Y/N)? ")
            if ans.upper() == 'Y':
                self.acled_to_mongodb("1995-01-01")
            else:
                sys.exit(1)


    def acled_api_request(self, query):
        """
        Query the ACLED API server (HTTP GET request).
        """
        r = requests.get(self.URL_ACLED_READ + query)
        assert r.status_code == 200
        return r.text


    def acled_api_page_flipper(self, query):
        """
        The ACLED server prefers returning data in pages, each page with about
        500 lines. This function will iteratively request all the pages for the
        quven query, collating a singular response.

        Args:
            query (str): As according to ACLED_API-User-Guide_28072016.pdf
        """
        page = 1
        data = None

        # ACLED API returns 'No data has been found' at end of data
        while True:
            time.sleep(2)           # treat ACLED server gently
            print('.', end='')
            sys.stdout.flush()
            result = self.acled_api_request(query + '&page=%i' % page)
            if result.startswith('No data'):
                break
            if page == 1:
                data = result
            else:
                data += result.split("\n", 1)[1]     # remove csv header
            page += 1

        print('\n%s %i pages retrieved from ACLED API.' % (str(datetime.datetime.now()), page-1))
        return data


    def _str_to_datetime(self, pandas_df):
        """
        Function to convert string formatted dates to datetime objects.
        """
        print(datetime.datetime.now(), 'Apply str->datetime on event_date...')
        pandas_df['event_date'] = pandas_df['event_date'].apply(ymd)


    def append_geo_points(self, pandas_df):
        """
        Function to append a new column of geo_points.
        """
        from shapely.geometry import Point
        df = pandas_df.loc[:, ['latitude','longitude']].apply(
            lambda df: Point(df.latitude, df.longitude), 
            axis=1)
        pandas_df.loc[:, 'geo_point'] = df


    def get_acled_data_gt_date(self, date):
        """
        Query ACLED server for data _newer_ (greater than) than given date.
        Date format (str): %Y-%m-%d (YYYY-MM-DD)
        """
        query = '?event_date=%s&event_date_where=>' % date
        return self.acled_api_page_flipper(query)


    def get_acled_data_gte_date(self, date):
        """
        Query ACLED server for data _newer than or equal to_ the given date.

        Args:
            date (str)  :    Date format (str): %Y-%m-%d (YYYY-MM-DD)

        Returns:
            Complete response from ACLED server.
        """
        query = '?event_date=%s&event_date_where=>%%3D' % date
        return self.acled_api_page_flipper(query)


    def _get_indexes_of_duplicates(self, pandas_df):
        """
        Find duplicate data_id entries in pandas DataFrame.
        """
        # keep=False => Mark all duplicates as True in returned pandas.Series
        duplicates = pandas_df['data_id'].duplicated(keep=False)
        dup_indexes = [index for index, duplicate in enumerate(duplicates) if duplicate is True]
        if dup_indexes:
            print("Warning! Found duplicates: ", dup_indexes)
            print(pandas_df.iloc[dup_indexes]
                  [['data_id', 'event_date', 'location', 'country', 'fatalities']])
        return dup_indexes


    def _remove_duplicates(self, pandas_df):
        """
        Naively remove duplicates inplace, except for the first occurrence.

        Args:
            pandas_df (pandas.DataFrame): DataFrame to search for duplicates
        """
        pandas_df.drop_duplicates(inplace=True)

        # After dropping duplicate whole rows, check if we still have duplicate
        # 'data_id' entries, indicating a collision for distinct events
        self._get_indexes_of_duplicates(pandas_df)


    def _csv_assert_header_consistency(self, csvblob):
        """
        Ensure consistency in column names for data to import and mongodb.
        """
        csv_columns = set(csvblob.split('\n', 1)[0].lower().split(','))
        acled_columns = set(self.ACLED_COLUMN_DTYPES.copy().keys())
        sym_diff = acled_columns.symmetric_difference(csv_columns)
        assert not sym_diff, "OH NO! Inconsistent header names found: %s " % (sym_diff)

    
    def pandas_df_to_mongodb(self, pandas_df):
        """
        Imports a pandas.DataFrame into mongodb.

        Returns:
            PyMongo.InsertManyResult of operation
        """
        # if collection is empty, after this
        # Index fields: ['_id_', 'data_id_1']  <-- why '_1' !?
        # if not 'data_id' in self.c.index_information():
        #   self.c.create_index('data_id',unique=True)
        result = self.c.insert_many(pandas_df.to_dict('records'))
        #print("Index fields:",sorted(list(self.c.index_information())))

        return result


    def csv_to_mongodb(self, csvblob, sep=','):
        """
        This function imports a csv buffer into mongodb.

        Args:
            csvblob (str):  A string buffer containing lines of csv data.
            sep (str)    :  Separator to use (default: ',')

        Returns:
            True if successful
        """
        print(datetime.datetime.now(), "Make DataFrame...")
        self._csv_assert_header_consistency(csvblob)

        f = io.StringIO(csvblob)
        df = pandas.read_csv(f, dtype=self.ACLED_COLUMN_DTYPES, sep=sep,
                             engine='c', keep_default_na=False)
        f.close()
 
        self._remove_duplicates(df)
        # dup_indexes = self._get_indexes_of_duplicates(df)
        self._str_to_datetime(df)

        result = self.pandas_df_to_mongodb(df)
        print(len(result.inserted_ids), "records inserted to mongodb,",
              len(csvblob.splitlines()), "lines in csv. (Investigate difference)")
 
        return self.pandas_df_to_mongodb(df)


    def acled_to_mongodb(self, datestr, to_csv_file=None):
        """
        This function will retrieve new entries from the ACLED API
        server and insert them into mongodb. To transfer the entire
        ACLED dataset, use "1995-01-01" as datestr.

        Arg:
            datestr (str): Find entries newer than this date.
                Date format: %Y-%m-%d (YYYY-MM-DD)
            to_csv_file (str): Filename for a csv copy of the retrieved data.

        Returns:
            True on success.
        """

        date = ymd(datestr)
        time_delta_years = (datetime.datetime.today() - date).days / float(365)

        if time_delta_years > 2:  # warn if more than two years of new data
            print('Warning! Query ACLED server for all data newer than date', datestr,
                  'could cause heavy load on the server.')
            answer = input('Continue (Y/N)? ')
            if answer.upper() != 'Y':
                print('Stopping.')
                return False

        print(datetime.datetime.now(), "Querying ACLED API (one dot is 500 rows) ", end='')

        csvblob = self.get_acled_data_gt_date(datestr)  # gte?

        if csvblob is None:     # no new data received
                return False

        if to_csv_file is not None:
            open(to_csv_file, "w").write(csvblob)
            print(datetime.datetime.now(), "Wrote", to_csv_file)

        return self.csv_to_mongodb(csvblob)


    def mongodb_get_newest_event_date(self):
        """
        Returns the newest event_date in the mongodb database.
        """
        #query = {'timestamp': {'$gt':starttime,'$lt':endtime}}
        data = self.c.find().sort([('event_date', pymongo.DESCENDING)]).limit(1)
        if data.count() == 0:
            return None
        return data.next()['event_date']


    def mongodb_update_database(self, to_csv_file=None, force=False):
        """
        Query the mongodb server to find the most recent entry and query
        ACLED API server for new data if data is older than 7 days.

        Args:
            to_csv_file (str):  Filename to write new csv data to
            force (bool):       Enforce update regardless of date
        """
        date = self.mongodb_get_newest_event_date()
        if not date:    # empty databae
            self.acled_download_database()

        delta_days = (datetime.datetime.today() - date).days

        if force or delta_days > 7:
            datestr = date.strftime('%Y-%m-%d')
            # TODO:
            # Ask for one day prior to datestr (due to $gt greater than)?
            self.acled_to_mongodb(datestr, to_csv_file)


    def mongodb_get_entire_database(self):
        """
        Returns the entire ACLED mongodb collection as pandas DataFrame.
        """
        return pandas.DataFrame(list(self.c.find()))


    def mongodb_get_query(self, query):
        """
        Executes a standard mongodb find()-query and returns the result
        as a pandas DataFrame.
        """
        return pandas.DataFrame(list(self.c.find(query)))


    def mongodb_delete_many(self, del_filter=None):
        """
        Performs a mongodb delete-many()-operation on the ACLED collection.
        """
        if not del_filter:
            starttime = datetime.datetime.strptime('2017-02-01', '%Y-%m-%d')
            del_filter = {'event_date': {'$gt':starttime}}
        result = self.c.delete_many(del_filter)
        print('Deleted', result.deleted_count, 'entries.')


    def mongodb_delete_collection(self):
        """
        Deletes the ACLED collection in mongodb database.
        """
        self.c.drop()


if __name__ == "__main__":
    ds_acled = ACLED()
    #ds_acled.mongodb_delete_many()
    #ds_acled.mongodb_delete_collection()
    #ds_acled.csv_to_mongodb(open('acled.entire.csv','r').read())
    #ds_acled.acled_to_mongodb('2017-01-15')
    ds_acled.mongodb_update_database()
    #starttime = datetime.datetime(2017, 1, 7)
    #endtime = datetime.datetime(2017, 1, 10)
    #query = {'event_date': {'$gt':starttime,'$lt':endtime}}
    #df = ds_acled.mongodb_get_query(query)
    df = ds_acled.mongodb_get_entire_database()
    print(df.columns)
    print(datetime.datetime.now(), "Done.")
