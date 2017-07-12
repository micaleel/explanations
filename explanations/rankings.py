import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from explanations.rank import rank

__all__ = ['rank_explanations', 'drop_columns_if_exists', 'rank_by_similarities']

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def rank_explanations(df_explanations, df_item_profiles, cols):
    """
    Args:
        df_item_profiles (pd.DataFrame): Item profiles
        df_explanations (pd.DataFrame): Explanations
        cols (list): Column names to rank explanations by.
    Returns:
        pd.DataFrame
    """
    # assert len(set(cols).difference(['average_rating']).intersection(set(df_explanations.columns))) == len(cols)

    # create column names to store rank values.
    rank_cols = ['rank_{}'.format(col) for col in cols]
    drop_columns_if_exists(df_explanations, ['average_rating'] + rank_cols, inplace=True)
    # add item-related columns to explanations.
    merge_cols = ['item_id', 'average_rating']
    if 'item_id' not in df_item_profiles.columns:
        df_item_profiles.reset_index('item_id', inplace=True)
    df_explanations = df_explanations.merge(df_item_profiles[merge_cols], how='left', on='item_id')

    def _rank_explanations():
        if 'item_id' in df_item_profiles.columns:  # speed lookups by setting item_id as index.
            df_item_profiles.set_index('item_id', inplace=True)

        for _, (session_id, df_session) in enumerate(df_explanations.groupby('session_id')):
            target_item_id = session_id.split('#')[1]
            # get items IDs and similarities of recommendations made for the target item.
            # include the target item in the returned ndarray.
            sims = np.append(df_item_profiles.ix[target_item_id].related_items_sims, [1.0])
            items = np.append(df_item_profiles.ix[target_item_id].related_items, target_item_id)

            df_item_sims = pd.DataFrame(list(zip(items, sims)), columns=['item_id', 'related_items_sims'])

            df_session = pd.merge(df_session, df_item_sims[['item_id', 'related_items_sims']], on='item_id')
            items = []
            session_size = len(df_session)

            for col, r_col in zip(cols, rank_cols):
                r = session_size - (rankdata(df_session[col].values, method='ordinal').astype(int))
                items.append((r_col, r))

            items.append(('explanation_id', df_session.explanation_id.values))
            yield pd.DataFrame.from_items(items)

    df_ranks = pd.concat(_rank_explanations())
    if 'item_id' not in df_item_profiles.columns:
        df_item_profiles.reset_index('item_id', inplace=True)

    return df_explanations.merge(df_ranks, on='explanation_id')


def drop_columns_if_exists(df, columns, inplace=False):
    """Drops columns from a DataFrame"""
    for c in columns:
        if c in df.columns:
            df.drop(c, axis=1, inplace=inplace)

    if not inplace:
        return df


def rank_by_similarities(df_explanations, df_similarities, df_item_profiles, recommendation_length=9,
                         skip_item_ids=None):
    """
    Ranks explanations by similarities of recommendations/related_items
    Each row in the similarities DataFrame represents a recommendation session.
    It contains a recommendation list of a target user-item pair and different
    similarity values for the recommendations. This function ranks each session's
    recommendations (or explanations) using different similarity values. Afterwards,
    the target item is added to the top of the recommendations with similarity value of 1.
    """

    # merge related_sims from df_item_profiles into df_similarities
    # TODO: confirm that df_item_profiles.related_items == df_similarities.related_items
    if 'item_id' not in df_item_profiles.columns:
        df_item_profiles.reset_index('item_id', inplace=True)
    merge_cols = ['item_id', 'related_items_sims']
    drop_columns_if_exists(df_similarities, ['related_items_sims'], inplace=True)
    assert len(merge_cols) == len(set(df_item_profiles.columns).intersection(merge_cols))
    df_similarities = df_similarities.merge(df_item_profiles[merge_cols], on='item_id', how='left')

    def _rank_by_similarities():
        """Ranks the explanations/recommendations in the similarities DataFrame"""
        sims_df_cols = sorted(df_similarities.columns)
        assert 'personalized_sims' in sims_df_cols
        assert 'related_items_sims' in sims_df_cols

        def _rank_similarities(similarities):
            """Ranks a list of similarity values, resolves ties randomly.
             The biggest value gets (i.e. most relevant item) the highest rank. Thus, we
             invert the ranking so that the highest ranked item has a value of 0,
             and the lowest has a value of N-1.
            """
            assert isinstance(similarities, np.ndarray) or isinstance(similarities, list)

            # TODO Use scipy.stats.rank instead.
            ranks = np.array(rank(similarities, ties='random'))
            # ranks = rankdata(similarities, method='ordinal')
            return ranks.max() - ranks

        for i, row in df_similarities.iterrows():
            session_id = "{}#{}".format(row.user_id, row.item_id)

            if skip_item_ids and row.item_id in skip_item_ids:
                continue

            if not isinstance(row.related_items, list):
                msg = "Skipping session {} because its recommendations have an invalid format {}"
                logger.warning(msg.format(session_id, type(row.related_items)))
            else:
                if len(row.related_items) == recommendation_length:
                    # append seed item to the similarities DataFrame
                    personalized_sims = np.append(row.personalized_sims, [1.0])
                    related_items_sims = np.append(row.related_items_sims, [1.0])
                    related_items = np.append(row.related_items, row.item_id)

                    # Rank the explanations by their similarities and
                    # recreate explanation_id for each related_item.
                    items = [('rank_related_items_sims_np', _rank_similarities(related_items_sims)),
                             ('rank_related_items_sims_p', _rank_similarities(personalized_sims)),
                             ('related_items_sims_p', personalized_sims),
                             ('related_items_sims_np', related_items_sims),
                             ('explanation_id', list(map(lambda x: "{}##{}".format(session_id, x), related_items)))]

                    yield pd.DataFrame.from_items(items)
                else:
                    # logging.warn('Unexpected recommendation length "{}"'.format(len(row.related_items)))
                    pass

    df_ranks = pd.concat(_rank_by_similarities())
    drop_columns_if_exists(df_explanations, [c for c in df_ranks if c != 'explanation_id'], inplace=True)
    return df_explanations.merge(df_ranks, on='explanation_id')
