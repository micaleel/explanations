import os
import sys

import elasticsearch
import numpy as np
import pandas as pd
from logbook import Logger, StreamHandler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from explanations import es_utils
from explanations.constants import *

StreamHandler(sys.stdout).push_application()
log = Logger(__name__)


def get_items_desc(df_extractions, df_items):
    """Get DataFrame of item descriptions """
    # For each item, get the number of users that've reviewed it, and also its average sentiment
    df_item_details = (df_extractions.groupby('item_id', sort=False, as_index=False)
                       .agg({'user_id': lambda x: len(x.unique()),
                             'sentiment': np.mean})
                       .rename(columns={'user_id': 'n_reviews',
                                        'sentiment': 'avg_sentiment'}))

    # Create an item-amenity matrix, where each element represents the average sentiment of an item's amenity.
    df_item_amenities = df_extractions.pivot_table(index='item_id', columns='amenity', values='sentiment',
                                                   fill_value='0').reset_index()

    # Create a DataFrame indexed by item_id with the columns average_rating, n_reviews, avg_sentiment,
    #  and sentiments for each and every amenity of the item
    df_items_desc = (df_items[['item_id', 'average_rating']].merge(df_item_amenities).merge(df_item_details)
                     .set_index('item_id'))
    return df_items_desc


def summarize_amenities(df=None):
    total_mentions = len(df)

    def _summarize():
        for amenity, _df in df.groupby('amenity'):
            if not _df.empty:
                sentiment_values = _df.sentiment.values
                mentions = len(sentiment_values)  # number of times feature is mentioned
                avg_sentiment = np.mean(sentiment_values)  # average sentiment of feature

                # convert real sentiment values to 1,0,-1 equivalent to pos, neu, neg
                polarities = np.sign(sentiment_values).astype(int)

                uniques, counts = np.unique(polarities, return_counts=True)
                sentiment_counts = {k: v for k, v in zip(uniques, counts)}
                pos_neg = (sentiment_counts.get(1, 0) + sentiment_counts.get(-1, 0))
                polarity_ratio = 0 if pos_neg == 0 else 1.0 * sentiment_counts.get(1, 0) / pos_neg
                opinion_ratio = 1.0 * (sentiment_counts.get(1, 0) + sentiment_counts.get(-1, 0)) / len(sentiment_values)
                yield dict(amenity=amenity,
                           senti_avg=avg_sentiment,
                           polarity_ratio=polarity_ratio,
                           opinion_ratio=opinion_ratio,
                           mentions=float(mentions) / total_mentions)

    return pd.DataFrame(_summarize())


def user_profile_and_features(df_extractions, user_ids=None):
    # build user features & user profiles
    if user_ids:
        _df_extractions = df_extractions[df_extractions.user_id.isin(user_ids)]
    else:
        _df_extractions = df_extractions

    log.info('Creating user features and user profiles ...')
    g_df_extractions = _df_extractions.groupby('user_id', sort=False)

    df_user_features = g_df_extractions.apply(summarize_amenities).reset_index().drop('level_1', axis=1)
    df_user_profiles = g_df_extractions.agg({'item_id': lambda x: x.unique().tolist()}).reset_index()
    df_user_profiles.rename(columns={'item_id': 'item_ids'}, inplace=True)

    return df_user_features, df_user_profiles


def _sanitize_related_items(related_items, item_ids):
    """Remove orphaned item_ids from related_items"""
    if isinstance(related_items, str):
        related_items = set(related_items.split(','))

    related_items = set(map(str, related_items))
    missing_ids = item_ids.symmetric_difference(related_items).difference(item_ids)
    return [rh for rh in related_items if rh not in missing_ids]


def build_profiles(df_items, df_extractions, user_ids):
    df_items_desc = get_items_desc(df_extractions, df_items)

    log.info('Computing pairwise similarities ...')
    scaler = MinMaxScaler()
    df_similarities = pd.DataFrame(cosine_similarity(scaler.fit_transform(df_items_desc)))
    df_similarities.index = df_items_desc.index
    df_similarities.columns = df_items_desc.index

    if df_items.item_id.nunique() != len(df_items_desc):
        log.warning('df_items.item_id.nunique() != len(df_items_desc) [{}, {}]'.format(df_items.item_id.nunique(),
                                                                                       len(df_items_desc)))
    df_items['related_items_sims'] = df_items.apply(lambda s: df_similarities.ix[s.item_id, s.related_items].tolist(),
                                                    axis=1)

    df_user_features, df_user_profiles = user_profile_and_features(df_extractions, user_ids)
    df_user_profiles = df_user_profiles.merge(get_user_amenity_ranks(df_user_features), on='user_id', how='inner')
    df_item_profiles = df_items[['item_id', 'average_rating', 'related_items', 'related_items_sims']].copy()
    # df_amenities = pd.DataFrame(AMENITIES, columns=['amenity'])
    df_amenities = pd.DataFrame(sorted(df_extractions.amenity.unique()), columns=['amenity'])
    df_item_features = (df_extractions.groupby('item_id', sort=False).apply(summarize_amenities)
                        .reset_index().drop('level_1', axis=1))

    result = dict(user_profiles=df_user_profiles, item_profiles=df_item_profiles, amenities=df_amenities,
                  user_features=df_user_features, item_features=df_item_features)

    return result


def get_user_amenity_ranks(df_user_features):
    def _get_amenity_ranks():
        for user_id, df in df_user_features.groupby('user_id'):
            df_amenity_mentions = df[['amenity', 'mentions']].copy()
            # higher mentions have higher ranks
            df_amenity_mentions['importance'] = df_amenity_mentions.mentions.rank()
            df_amenity_mentions['importance'] = df_amenity_mentions.importance / df_amenity_mentions.importance.max()
            amenity_mentions_dict = df_amenity_mentions.set_index('amenity').importance.to_dict()
            amenities_vector = {a: amenity_mentions_dict.get(a, 0) for a in TA_AMENITIES}
            yield dict(user_id=user_id, amenity_ranks=list(amenities_vector.values()))

    df_user_amenity_ranks = pd.DataFrame(list(_get_amenity_ranks()))
    return df_user_amenity_ranks


def patch_item_profiles(df_item_profiles, df_item_features, df_amenities):
    # we convert the related items from a string to a list (if necessary)
    _related_items = df_item_profiles.related_items.values[0]
    if isinstance(_related_items, str):
        df_item_profiles.related_items = df_item_profiles.related_items.apply(lambda x: x.split(','))
    elif isinstance(_related_items, list) or isinstance(_related_items, np.ndarray):
        pass
    else:
        raise TypeError('Unrecognized type for df_item_profiles.related_items')

    # remove the columns from item profiles in case they've already been patched.
    log.info('Dropping columns ...')
    cols = ['mentions', 'opinion_ratio', 'polarity_ratio', 'senti_avg', 'pros_pol', 'cons_pol']

    for col in cols:
        if col in df_item_profiles.columns:
            df_item_profiles.drop(col, axis=1, inplace=True)

    # no need to carry around unnecessary columns
    if 'related_item_ids' in df_item_features.columns:
        df_item_features.drop('related_item_ids', axis=1, inplace=True)

    # we create the sentiment arrays for each feature
    cols = ['mentions', 'senti_avg', 'opinion_ratio', 'polarity_ratio', 'amenity']

    def _merge_amenities(d):
        df = pd.merge(df_amenities, d[cols], on='amenity', how='outer', sort=False).fillna(0)
        return pd.Series({k: df[k].values for k in cols if k != 'amenity'})

    grouped_features = df_item_features.groupby('item_id').apply(_merge_amenities).reset_index()
    df_item_profiles = pd.merge(df_item_profiles, grouped_features, on='item_id')

    df_item_profiles['pros_pol'] = df_item_profiles.polarity_ratio.apply(lambda x: x >= 0.5)
    df_item_profiles['cons_pol'] = df_item_profiles.polarity_ratio.apply(lambda x: ~(x >= 0.5))

    df_item_profiles.average_rating = df_item_profiles.average_rating.astype(float)
    df_item_profiles.average_rating.fillna(df_item_profiles.average_rating.median(), inplace=True)
    return df_item_profiles


def _patch_user_profiles(df_user_profiles, df_user_features, df_amenities, sent_threshold=0.7):
    drop_cols = ['item_ids', 'neighbour_ids']
    for drop_col in drop_cols:
        if drop_col in df_user_features.columns:
            df_user_features.drop(drop_col, axis=1, inplace=True)

    # we convert the related items from a string to a list (if necessary)
    _item_ids = df_user_profiles.item_ids.values[0]

    # remove the columns from user profiles in case they've already been patched.
    cols = ['mentions', 'opinion_ratio', 'polarity_ratio', 'senti_avg', 'pros_pol', 'cons_pol']
    for col in cols:
        if col in df_user_profiles.columns:
            df_user_profiles.drop(col, axis=1, inplace=True)

        if isinstance(_item_ids, str):
            df_user_profiles.item_ids = df_user_profiles.item_ids.apply(lambda x: x.split(','))
        elif isinstance(_item_ids, list) or isinstance(_item_ids, np.ndarray):
            pass
        else:
            raise TypeError('Unrecognized type for df_user_profiles.item_ids')

    cols = ['mentions', 'senti_avg', 'opinion_ratio', 'polarity_ratio', 'amenity']

    def _merge_amenities(d):
        df = pd.merge(df_amenities, d[cols], on='amenity', how='outer', sort=False).fillna(0)
        return pd.Series({k: df[k].values for k in cols if k != 'amenity'})

    grouped_features = df_user_features.groupby('user_id').apply(_merge_amenities).reset_index()
    # create the new columns
    df_user_profiles = pd.merge(df_user_profiles,
                                # df_user_profiles.user_id.apply(_gen),
                                grouped_features,
                                on='user_id')

    df_user_profiles['pros_pol'] = df_user_profiles.polarity_ratio.apply(lambda x: x >= sent_threshold)
    df_user_profiles['cons_pol'] = df_user_profiles.polarity_ratio.apply(lambda x: ~(x >= sent_threshold))

    return df_user_profiles


def _check_consistency(df_extractions, df_items, df_recommendations, recommendation_size):
    log.info('Checking consistency of recommendations ...')
    # ensure that profiles for all recommendations exists. If recommendation doesn't match
    # an existing profile, remove the item for which it was recommended for.
    item_ids = set(df_items.item_id.astype(str))
    df_items['related_items'] = df_items.related_items.apply(lambda x: _sanitize_related_items(x, item_ids))
    df_items['n_related_items'] = df_items.related_items.apply(len)
    without_recs = df_items[df_items.n_related_items == 0].item_id.astype(str)
    log.warning('{} items have no recommendations'.format(len(without_recs)))
    log.info(
        'n_related_items.unique() = {}'.format(','.join([str(s) for s in sorted(df_items.n_related_items.unique())])))
    log.warning('{} items have invalid recommendation sizes and will be removed from future steps.'.format(
        len(df_items[df_items.n_related_items != int(recommendation_size)])))
    df_items = df_items[df_items.n_related_items == recommendation_size]
    # Validate columns in df_items and df_extractions
    assert len(set(ITEM_COLS).intersection(df_items.columns)) == len(ITEM_COLS)
    assert len(set(OPINION_COLS).intersection(df_extractions.columns)) == len(OPINION_COLS), '{} {}'.format(
        OPINION_COLS, ','.join(df_extractions.columns.tolist()))
    assert len(set(df_items.item_id).intersection(df_recommendations.item_id)) == df_items.item_id.nunique()


# TODO Hook up explanation generation to the pipeline (add to notebook)
# TODO Hook up BeerAdvocate to the pipeline (split file into items.csv & extractions.csv, convert ratings to sentiments)
# TODO Hook up plot generation (only the ES code; save to disk)
# TODO Hook up personalized recommendation


def build_profiles_main(options):
    # Ensure that items and extractions file exist.
    for path in [options.items_csv_file, options.extractions_csv_file, options.recommendations_csv_file]:
        assert os.path.exists(path), "File '{}' does not exist".format(path)

    # Ensure that output directory exists.
    output_dir = options.output_dir if options.output_dir.endswith('/') else options.output_dir + '/'
    if not os.path.exists(output_dir):
        os.makedirs(os.path.realpath(output_dir))

    log.info('Loading items, extractions and recommendations ...')
    df_items = pd.read_csv(options.items_csv_file).rename(columns=COL_ALIASES)[ITEM_COLS]
    df_extractions = pd.read_csv(options.extractions_csv_file).rename(columns=COL_ALIASES)[OPINION_COLS]
    df_recommendations = pd.read_csv(options.recommendations_csv_file).rename(columns=COL_ALIASES)

    # Ensure all item_ids are strings (regardless of dataset) for consistency.
    for df in [df_items, df_extractions, df_recommendations]:
        if 'item_id' in df.columns:
            df['item_id'] = df.item_id.astype(str)

    # Filter items and extractions, if necessary.
    if options.item_ids_csv_file:
        try:
            df = pd.read_csv(options.item_ids_csv_file)
            assert 'item_id' in df.columns
            log.info('Sampling items and extractions ...')
            df_items = df_items[df_items.item_id.isin(df.item_id.unique())]
            df_extractions = df_extractions[df_extractions.item_id.isin(df.item_id.unique().tolist())]
        except OSError as ex:
            log.warn(ex)

    # Filter users, if necessary.
    user_ids = None
    if options.user_ids_csv_file:
        try:
            df = pd.read_csv(options.user_ids_csv_file)
            assert 'user_id' in df.columns
            user_ids = df.user_id.unique().tolist()
        except OSError as ex:
            log.warn(ex)

    # remove extractions that don't match any item in our records.
    df_extractions = df_extractions[df_extractions.item_id.isin(df_items.item_id)]
    df_items = df_items[df_items.item_id.isin(df_extractions.item_id)]

    # merge non-personalized recommendations onto df_items.
    df_items = df_items.merge(df_recommendations, how='left')

    _check_consistency(df_extractions, df_items, df_recommendations, options.recommendation_size)

    log.info('Building user and item profiles ...')
    data = build_profiles(df_items, df_extractions, user_ids)

    log.info('Patching user profiles ...')
    data['user_profiles'] = _patch_user_profiles(df_user_profiles=data['user_profiles'],
                                                 df_user_features=data['user_features'],
                                                 df_amenities=data['amenities'],
                                                 sent_threshold=options.sentiment_threshold)

    log.info('Patching item profiles ...')
    data['item_profiles'] = patch_item_profiles(df_item_profiles=data['item_profiles'],
                                                df_item_features=data['item_features'],
                                                df_amenities=data['amenities'])

    if hasattr(options, 'save_csv_files') and options.save_csv_files:
        for k, v in data.items():
            output_path = '{}{}.csv'.format(options.output_dir, k)
            log.info("Saving '{}' to '{}'...".format(k, output_path))
            v.to_csv(output_path, index=False)

    log.info('Connecting to Elasticsearch at {}'.format(options.es_host))
    es = elasticsearch.Elasticsearch(options.es_host)

    log.info("Saving item_profiles to Elasticsearch in index '{}' ...".format(options.item_index))
    es_utils.pd_to_es(data['item_profiles'], es, index=options.item_index, key_col='item_id')

    log.info("Saving user_profiles to Elasticsearch in index '{}' ...".format(options.user_index))
    es_utils.pd_to_es(data['user_profiles'], es, index=options.user_index, key_col='user_id')

    log.info("Saving user_profiles to Elasticsearch in index '{}' ...".format(options.user_index))
    es_utils.pd_to_es(data['item_profiles'][['item_id', 'related_items', 'related_items_sims']], es,
                      index=options.rec_index, key_col='item_id')
