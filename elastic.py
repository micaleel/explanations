"""
This module contains helpers for working with data in Elasticsearch
"""
import logging

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from explanations.config import Config
import os

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.getLogger('elasticsearch').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

__all__ = ['load_all_datasets', 'load_explanations', 'load_index', 'optimize']


def load_index(es_host: str, index: str, timeout: int = 120, max_retries: int = 10,
               retry_on_timeout: int = True, preserve_order: bool = False,
               lists_to_ndarray: bool = False) -> pd.DataFrame:
    """Load explanations from Elasticsearch into a DataFrame

    Args:
        es_host (str): Elasticsearch host
        index (str): Elasticsearch index with session data
        retry_on_timeout (int):
        preserve_order (bool):
        timeout (int):
        max_retries (int):
        lists_to_ndarray (bool): If True, converts fields with list values into numpy arrays.

    Returns:
        DataFrame: Explanations.
    """

    def _load():
        client = Elasticsearch(es_host, timeout=timeout, max_retries=max_retries, retry_on_timeout=retry_on_timeout)
        query = {'query': {'match_all': {}}}
        docs = scan(client, query=query, index=index, preserve_order=preserve_order)

        for d in docs:
            yield d['_source']

    df_out = pd.DataFrame(_load())
    if lists_to_ndarray:
        for c in df_out.select_dtypes(include=['object']).columns:
            if isinstance(df_out[c].values[0], list):
                df_out[c] = df_out[c].apply(np.array)
    return df_out


def export_es_to_hdf5(config_path, hdf5_dir):
    """Exports explanations from Elasticsearch to HDF5

    Args:
        config_path: Path to configuration file
        hdf5_dir: Where to store the output HDF5 file
    """
    config = Config.from_file(config_path)
    df_explanations = load_explanations_es(es_host=config.es_host, index_session=config.index_session)

    # Save to HDF5, but use format="table" since df_explanations contains categorical columns.
    path = os.path.join(hdf5_dir, '{}.h5'.format(config.index_session))  # Output file path
    logger.info('Saving to {}'.format(path))
    # df_explanations.to_hdf(path_or_buf=path, mode='w', complevel='blosc', key='index_session', format="table")
    df_explanations.to_hdf(path_or_buf=path, mode='w', complevel='blosc', key='index_session', format="fixed")


def load_explanations_es(es_host: str, index_session: str) -> pd.DataFrame:
    """Loads explanations from Elasticsearch

    Args:
        es_host (str): Elasticsearch host
        index_session (str): Name of the index for sessions/explanations
    """
    logger.info('Loading from {} ({})'.format(es_host, index_session))

    def _load_sessions():
        """Loads sessions as Elasticsearch documents"""
        client = Elasticsearch(es_host)
        docs = scan(client, query={'query': {'match_all': {}}}, index=index_session, preserve_order=False)
        return docs

    def _create_explanations(docs):
        """Creates a DataFrame of explanations from an Elasticsearch index"""
        for d in docs:
            # Yielding a DataFrame is then concatenating it is 13 times slower
            # than iterating to the list of `explanations` and yielding them
            for explanation in d['_source']['explanations']:
                yield explanation

    documents = _load_sessions()
    df_explanations = pd.DataFrame(_create_explanations(docs=documents))
    df_explanations = optimize(df_explanations)

    # Convert lists to ndarrays
    for c in df_explanations.select_dtypes(include=['object']).columns:
        if isinstance(df_explanations[c].values[0], list):
            df_explanations[c] = df_explanations[c].apply(np.array)

    return df_explanations


def load_all_datasets(config_paths, hdf_dir: str = None, override_hdf: bool = False):
    """Loads all explanations from a list configuration paths:

    Args:
        config_paths (list): List of configuration paths
        hdf_dir (str): Directory to store/retrieve HDF5 files.
        override_hdf (bool): Whether or not to create new copies of HDF5 files.
    Returns:
        DataFrame: all explanations from all configuration paths merged into a single DataFrame.
    """

    def load_all():
        """Load seed rankings for all datasets."""
        for idx, path in enumerate(config_paths):
            config = Config.from_file(path)
            logger.info(
                'Loading explanation from {}/{} ({} of {})...'.format(
                    config.es_host, config.index_session, idx, len(config_paths)
                )
            )

            df_explanations = load_explanations(config.es_host, config.index_session, hdf_dir=hdf_dir,
                                                override_hdf=override_hdf)
            parts = config.short_name.split('-')
            dataset = parts[0]
            strength_fn = parts[1]
            weights = parts[2]

            # dataset, strength_fn, weights = config.short_name.split('-')
            df_explanations['dataset'] = dataset
            df_explanations['strength_fn'] = strength_fn
            df_explanations['weights'] = weights
            df_explanations['short_name'] = config.short_name
            df_explanations['alpha'] = config.alpha

            for col in ('short_name', 'dataset', 'strength_fn', 'weights'):
                df_explanations[col] = df_explanations[col].astype('category')

            yield df_explanations

    df_explanations_all = pd.concat(load_all())
    logger.info('Loaded {:,} explanations from {} config paths'.format(len(df_explanations_all), len(config_paths)))
    return df_explanations_all


def load_explanations(es_host: str, index_session: str, preserve_order: bool = False, lists_to_ndarray: bool = False,
                      hdf_dir: str = None, override_hdf: bool = False):
    """Load explanations from Elasticsearch into a DataFrame

    Args:
        es_host (str): Elasticsearch host
        index_session (str): Elasticsearch index with session data
        preserve_order (bool):
        lists_to_ndarray (bool): If True, converts fields with list values into numpy arrays.
        hdf_dir (str): Directory to store HDF5 files.
        override_hdf (bool): Whether or not to store new versions of HDF5 files
    Returns:
        DataFrame: Explanations.
    """
    if hdf_dir:
        path = os.path.join(hdf_dir, '{}.h5'.format(index_session))
        if hdf_dir and override_hdf and os.path.exists(path):
            os.remove(path=path)

    if hdf_dir and os.path.isfile(path):
        # Load from HDF5
        logger.info('Loading from HDF5 {}'.format(path))
        df_explanations = pd.read_hdf(path_or_buf=path, key='index_session')
        df_explanations = optimize(df_explanations)
        return df_explanations

    def _load_sessions():
        """Loads sessions as Elasticsearch documents"""
        client = Elasticsearch(es_host,
                               # timeout=timeout,
                               # max_retries=max_retries, retry_on_timeout=retry_on_timeout
                               )
        query = {'query': {'match_all': {}}}
        docs = scan(client, query=query,
                    # size=size,
                    index=index_session, preserve_order=preserve_order)
        return docs

    def _create_explanations(docs):
        """Creates a DataFrame of explanations from an Elasticsearch index"""
        for d in docs:
            # Yielding a DataFrame is then concatenating it is 13 times slower
            # than iterating to the list of `explanations` and yielding them
            for explanation in d['_source']['explanations']:
                yield explanation

    documents = _load_sessions()
    df_explanations = pd.DataFrame(_create_explanations(docs=documents))
    df_explanations = optimize(df_explanations)

    if lists_to_ndarray:
        for c in df_explanations.select_dtypes(include=['object']).columns:
            if isinstance(df_explanations[c].values[0], list):
                df_explanations[c] = df_explanations[c].apply(np.array)

    if hdf_dir:
        # Save to HDF5
        logger.info('Saving copy to {}'.format(path))
        # NOTE: Use format="table" since df_explanations contains categorical columns.
        df_explanations.to_hdf(path_or_buf=path, mode='w', complevel='blosc', key='index_session', format="table")
    return df_explanations


def optimize(df_explanations, categoricals=False):
    """Optimizes a DataFrame

        df_explanations: Input data frame
        categoricals (bool): Whether string values should be converted to categorical columns
    Returns:
        pd.DataFrame
    """
    obj_to_str_cols = frozenset(['better_count',
                                 'better_pro_scores',
                                 'cons',
                                 'cons_comp',
                                 'pros',
                                 'pros_comp',
                                 'related_items_sims_np',
                                 'seed_item_id',
                                 'target_item_mentions',
                                 'target_item_sentiment',
                                 'user_id',
                                 'worse_con_scores',
                                 'worse_count'])
    cols_to_remove = frozenset(
        ['cons', 'cons_comp', 'pros', 'pros_comp', 'related_items_sims_np', 'seed_item_id', 'user_id'])
    cols_to_remove = frozenset([])
    temp_drop_cols = frozenset(['better_count', 'target_item_mentions', 'target_item_sentiment', 'worse_count'])
    temp_drop_cols = frozenset([])

    categorical_cols = frozenset(['dataset', 'short_name', 'strength_fn', 'target_item_id', 'weights'])

    for c in df_explanations.columns:
        if c in categorical_cols and categoricals:
            df_explanations[c] = df_explanations[c].astype('category')

        if c in cols_to_remove or c in temp_drop_cols:
            df_explanations.drop(c, axis=1, inplace=True)

    float64_cols = list(df_explanations.select_dtypes(include=[np.float64]).columns)
    for c in float64_cols:
        df_explanations[c] = df_explanations[c].astype(np.float32)

    int64_cols = list(df_explanations.select_dtypes(include=[np.int64]).columns)
    for c in int64_cols:
        df_explanations[c] = df_explanations[c].astype(np.int32)

    obj_cols = list(df_explanations.select_dtypes(include=[np.object]).columns)
    for c in obj_cols:
        if c in obj_to_str_cols:
            df_explanations[c] = df_explanations[c].astype(str)

    return df_explanations
