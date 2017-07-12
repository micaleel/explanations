from functools import partial
from multiprocessing import Pool

import pandas as pd
import scipy.spatial.distance as dist
import scipy.stats as stats
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from ncdg_ex import perturb

TIMEOUT = 120
MAX_RETRIES = 10
RETRY_ON_TIMEOUT = True
QUERY_SIZE = 1
PRESERVE_ORDER = False

# TODO: rename variable 'perturb_scale' to 'noise_level'
__all__ = ['load_sessions', 'compute_edit_distances']


def load_sessions(es_host, index_session):
    """Loads sessions as Elasticsearch documents"""
    client = Elasticsearch(es_host, timeout=TIMEOUT, max_retries=MAX_RETRIES, retry_on_timeout=RETRY_ON_TIMEOUT)
    query = {'query': {'match_all': {}}}
    documents = scan(client, query=query, index=index_session, size=QUERY_SIZE, preserve_order=PRESERVE_ORDER)
    return documents


def compute_edit_dist(df_explanations, perturb_scale=0.0, gold_std_col='rank_target_item_average_rating'):
    """Computes the edit distance between the rankings from different approaches.

        Args:
            perturb_scale: noise level; higher values indicate more noise.
            gold_std_col: column with gold standard ranking.
            df_explanations: DataFrame of explanations for a single session.

        Returns:
            List of dictionaries, each representing edit distances for the session.
                The dictionary will contain the following keys:
                - session_id: the session identifier.
                - tau: kendall tau between compared column and gold standard.
                - hamming: hamming distance between compared column and gold standard.
                - col: column compared with gold standard
    """
    if perturb_scale > 0:
        df_explanations['target_item_average_rating'] = perturb(df_explanations.target_item_average_rating,
                                                                scale=perturb_scale)
        ranks = stats.rankdata(df_explanations.target_item_average_rating)
        df_explanations['rank_target_item_average_rating'] = ranks.tolist()

    session_id = df_explanations.ix[0].session_id

    # compare gold standard ranking with others:
    # calculate kendall tau and hamming distance between gold standard and other rankings
    for col in df_explanations.columns:
        if col.startswith('rank_') and col != gold_std_col:
            tau, _ = stats.kendalltau(df_explanations.rank_target_item_average_rating, df_explanations[col])
            hamming = dist.hamming(df_explanations.rank_target_item_average_rating, df_explanations[col])
            yield dict(session_id=session_id, tau=tau, hamming=hamming, col=col)


def _compute_edit_dist(document, perturb_scale=0.0, gold_std_col='rank_target_item_average_rating'):
    """Computes the edit distance between the rankings from different approaches.

    Args:
        perturb_scale: noise level; higher values indicate more noise.
        gold_std_col: column with gold standard ranking.
        document: Elasticsearch document representing a session.

    Returns:
        List of dictionaries, each representing edit distances for the session.
            The dictionary will contain the following keys:
            - session_id: the session identifier.
            - tau: kendall tau between compared column and gold standard.
            - hamming: hamming distance between compared column and gold standard.
            - col: column compared with gold standard
    """
    df_explanations = pd.DataFrame(document['_source']['explanations'])
    distances = compute_edit_dist(df_explanations, perturb_scale=perturb_scale, gold_std_col=gold_std_col)
    return list(distances)


def compute_edit_distances(es_host, index_session, perturb_scale=0, processes=3, chunksize=3):
    documents = load_sessions(es_host=es_host, index_session=index_session)
    p_compute_edit_distance = partial(_compute_edit_dist, perturb_scale=perturb_scale)

    def _f():
        with Pool(processes) as p:
            for distances in p.imap_unordered(p_compute_edit_distance, documents, chunksize=chunksize):
                yield from distances

    df_distances = pd.DataFrame(_f())
    return df_distances
