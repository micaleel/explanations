
import yaml
import pandas as pd
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan # TODO: Delete this line
from glob import glob
from pprint import pprint
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display
import matplotlib as mpl

es_params = dict(
    timeout = 120,
    max_retries = 10,
    retry_on_timeout=True,
    query_size=1,
    preserve_order=False
)

def load_explanations(es_host, session_index):
    def _load_sessions():
        """Loads sessions as Elasticsearch documents"""
        client = Elasticsearch(
            es_host,
            timeout=es_params['timeout'],
            max_retries=es_params['max_retries'],
            retry_on_timeout=es_params['retry_on_timeout']
        )

        query = {'query': {'match_all': {}}}

        documents = scan(
            client,
            query=query,
            index=session_index,
            size=es_params['query_size'],
            preserve_order=es_params['preserve_order']
        )

        return documents

    def _create_explanations(documents):
        """Creates a DataFrame of explanations from an Elasticsearch index"""
        for d in documents:
            # Yielding a DataFrame is then concatenating it is 13 times slower
            # than iterating to the list of `explanations` and yielding them
            for explanation in d['_source']['explanations']:
                yield explanation


    documents = _load_sessions()
    df_explanations = pd.DataFrame(_create_explanations(documents))

    return df_explanations
