"""Function that runs the pre-processing necessary to join the ACLED data set
and using it together with the ESRI shapefile data.

This pre-processing is specific to the data points in ACLED as well as the
particular ESRI shapefile. New datapoints may require updates to this file.
"""
def acled_preprocess(acled_df, gpd_df, verbose=False):
    """ Function handlig pre-processing of the ACLED data and the ESRI
    shapefile data (used by geopandas for plotting).

    Consistency in between the two data sources is needed in order to use
    them together.

    If this function fails, it is likely due to new data in the ACLED data
    base that is inconsistent with the existing data (e.g. country name
    spelled differently or similar). Try running function with 'verbose=True'
    to diagnose the problem.
    """

    def compare_countries_acled_esri(acled, esri, verbose=False):
        """ Prints difference between the acled and esri column containing
        country names
        """
        a_set = set(acled['country'].unique())
        g_set = set(esri.index.unique())

        a_diff = a_set.difference(g_set)
        g_diff = g_set.difference(a_set)

        if verbose:
            print("-- Comparing names in the two datasets --")
            print("Names in ACLED, not in ESRI: \n", a_diff)
            print("Names in ESRI, not in ACLED: \n", g_diff)

        return a_diff, g_diff

    # Mask ESRI data on countries in Africa:
    mask = gpd_df['continent'] == 'Africa'
    gpd_df = gpd_df.loc[mask, :].reset_index(drop=True)
    gpd_df = gpd_df.loc[:, ('name', 'subregion', 'x', 'y', 'geometry', 'pop_est')]
    gpd_df.set_index("name", inplace=True)

    # First comparison, in order to print initial difference if verbose=True
    compare_countries_acled_esri(acled_df, gpd_df, verbose)

    # Manually assigning the matching countries:
    new_names = {
        "Côte d'Ivoire": "Ivory Coast",
        "Dem. Rep. Congo": "Democratic Republic of Congo",
        "S. Sudan": "South Sudan",
        "Central African Rep.": "Central African Republic",
        "Congo": "Republic of Congo",
        "Eq. Guinea": "Equatorial Guinea"
    }

    gpd_df.rename(new_names, inplace=True)

    # Removing whitespace in a 'Mozambique ' entry):
    acled_df.loc[acled_df['country'] == 'Mozambique ', 'country'] = 'Mozambique'

    """Finally, 'W. Sahara', 'Somaliland' are more complicated matters, as
    they are disputed territories.

    'W. Sahara':
        The approach that aligns best with the ACLED dataset is
        to consider Western Sahara to be a part of Morocco, ref e.g.:
        https://en.wikipedia.org/wiki/Sahrawi_Arab_Democratic_Republic [...]
            [...] #International_recognition_and_membership

    'Somaliland':
        In the ACLED dataset, Somaliland is considered part of Somalia

    We do the pairwise merging as described above.
    """
    merge_pairs = [['Somalia', 'Somaliland'],
                   ['Morocco', 'W. Sahara']]

    for pair in merge_pairs:
        assert pair[0] in gpd_df.index, "{0} not in index of gpd_df.".format(pair[0])
        assert pair[1] in gpd_df.index, "{0} not in index of gpd_df.".format(pair[1])

        new_geometry = gpd_df.loc[pair, 'geometry'].unary_union
        gpd_df.loc[pair[0], 'geometry'] = new_geometry
        gpd_df.loc[pair[0], 'pop_est'] = gpd_df.loc[pair, 'pop_est'].sum()

        new_combined_x = list(new_geometry.exterior.coords.xy[0])
        new_combined_y = list(new_geometry.exterior.coords.xy[1])

        gpd_df.set_value(pair[0], 'x', new_combined_x)
        gpd_df.set_value(pair[0], 'y', new_combined_y)

        gpd_df.drop(pair[1], inplace=True)

    # Serialize geometry-column, for Bokeh:
    from shapely.geometry import mapping
    gpd_df['geometry'] = gpd_df['geometry'].apply(lambda x: mapping(x))

    # Final check:
    a_diff, g_diff = compare_countries_acled_esri(acled_df, gpd_df, verbose)

    error_message = "Country names not alligned after pre-processing. " \
                    "Try running this function with 'verbose=True' to " \
                    "investigate possible discrepancies in new data."
    assert (not a_diff) and (not g_diff), error_message

    return acled_df, gpd_df
