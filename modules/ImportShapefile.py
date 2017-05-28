import geopandas as gpd
import numpy as np
import bokeh

""" TODO
Consider adding parameter with list of dicts [{'continent': 'Africa'}]
resulting in:

mask = data_world['continent']=='Africa'
self.df = data_world.loc[mask,:]
"""

class ImportShapefile(object):
    """Importing 'ESRI shapefile' as geopandas df and adding 'x','y'

    __init__ opens shapefile and calls necessary class-functions to create a
    geopandas df that includes columns 'x' and 'y' consisting of lists created
    from the shapefiles geometry objects' (Polygons/MultiPolygons) exterior
    function.

    Suggested usage:
    link = [dir to .shp-file]
    df_full = ImportShapefile(link).get_df()

    Credit:
    Functions in this class are taken from/based on the git-repo in ref [1].

    References
        [1] https://automating-gis-processes.github.io/2016/
    """

    def _get_xy_coords(self, exterior, coord_type):
        """ Returns either x or y coordinates from passed geometry.exterior

        Keyword arguments:
        exterior -- Exterior object (geometry.exterior) of a Polygon
        coord_type -- 'x' or 'y'
        """
        if coord_type == 'x':
            return exterior.coords.xy[0]
        elif coord_type == 'y':
            return exterior.coords.xy[1]

    def _get_poly_coords(self, geometry, coord_type):
        """ Passes exterior of geometry to self._get_xy_coords."""
        ext = geometry.exterior
        return self._get_xy_coords(ext, coord_type)

    def _multi_poly_handler(self, multi_polygon, coord_type):
        """ Iterates over the 'Polygons' contained in the passed 'MultiPolygon'

        Function combines the different Polygons separated by an np.nan, as this
        is the preferred format of Bokeh: https://github.com/bokeh/bokeh/issues/2321
        Credit to [1] for solution.

        Keyword arguments:
        multi_polygon -- MultiPolygon object contining several Polygon objects
        coord_type -- 'x' or 'y'
        """
        for i, part in enumerate(multi_polygon):
            # On the first part of the Multi-geometry initialize the coord_array (np.array)
            if i == 0:
                coord_arrays = np.append(self._get_poly_coords(part, coord_type), np.nan)
            else:
                coord_arrays = np.concatenate([coord_arrays,
                                              np.append(self._get_poly_coords(part, coord_type), np.nan)])
        # Return the coordinates
        return coord_arrays

    def _get_coords(self, row, geom_col, coord_type):
        """ Returns list containing geometry type asked for

        Only "Polygon" and "MultiPolygon" implemented in class. Other geometry
        types raises exception.

        Keywords arguments:
        row -- Geopandas df row passed to function
        geom_col -- Name of column in shapefile containing shapely-geometry
        coord_type -- 'x' or 'y'
        """
        geom = row[geom_col]
        gtype = geom.geom_type

        if gtype == "Polygon":
            #return self._get_poly_coords(geom, coord_type) # Possibly needed for comp. w/new Bokeh version. Not implemented.
            return list(self._get_poly_coords(geom, coord_type))
        elif gtype == "MultiPolygon":
            #return self._multi_poly_handler(geom, coord_type) # Possibly needed for comp. w/new Bokeh version. Not implemented.
            return list(self._multi_poly_handler(geom, coord_type))
        else:
            err_msg = "Geometry type (",gtype,") not suppert by function"
            raise TypeError(err_msg)

    def _add_coordinate_data(self, df, geom_col):
        """ Returns (x,y) containing pairwise points of elements geo-exterior"""
        coord_types = ['x', 'y']

        x = df.apply(self._get_coords,
                     geom_col=geom_col,
                     coord_type='x',
                     axis = 1)

        y = df.apply(self._get_coords,
                     geom_col=geom_col,
                     coord_type='y',
                     axis = 1)
        return x,y

    def get_df(self):
        """Returns self.df (created at initiation)."""
        return self.df

    def __init__(self, link, geom_col='geometry'):
        """Loads geopandas dataframe and prepares parameters.

        Loads 'ESRI shapefile', initiates self.df and adds 'x' and 'y' columns
        (from 'geom_col' column of shapefile).

        Keyword arguments:
        link -- Directory to shapefile (.shp)
        geom_col -- Name of column in shapefile containing shapely-geometry
        """
        self.df = gpd.read_file(link)
        (self.df['x'], self.df['y']) = self._add_coordinate_data(self.df, geom_col)

        return None
