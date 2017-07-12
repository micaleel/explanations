import time
import traceback
from itertools import islice

import elasticsearch
import numpy as np
import six
from elasticsearch.helpers import scan, streaming_bulk
from logbook import Logger

from explanations.utils import consume, drop_columns_if_exists

log = Logger()

SCAN_KWARGS = {
    'scroll': '20m',
    'raise_on_error': True,
    'preserve_order': False,
    'size': 1000,
    'timeout': '100m'
}


def es_action(source, index=None, type=None, id_key=None):
    type = index if type is None else type
    return {
        "_index": index,
        "_type": type,
        "_id": source.get(id_key),
        "_source": source
    }


def profiles_to_es(es=None, df=None, index=None, key_col=None, chunk_size=10000, mapping=None):
    """
    assumes we have a dataframe of this form:

    index mentions(list) polarity_ratio(list)

    where the index is the user_id

    and converts it to a list of dictionaries for adding to ES:

    [{'mentions': [0.017241379310344827,...],
    'polarity_ratio': [1.0,...],
    'user_id': u'0001105EC50F4429ED9904C837DEC59F'},
    ...
    ]
    """

    def gen():
        #
        first_row = df.iloc[0]
        for col in df.columns:
            if isinstance(first_row[col], np.ndarray):
                # log.info('converting: {}', col)
                df[col] = df[col].apply(lambda x: x.tolist())

        for i, (idx, row) in enumerate(df.iterrows()):
            doc = row.to_dict()
            doc[key_col] = idx
            yield doc
        return

    if mapping is None:
        mapping = {"mappings": {
            index: {
                "_all": {"enabled": True},
                "dynamic_templates": [
                    {"item_id": {
                        "match": "item_id",
                        "mapping": {
                            "type": "string",
                            "index": "not_analyzed"
                        }
                    }},
                    {'amenity_ranks': {
                        "match": "amenity_ranks",
                        "mapping": {
                            "type": "double"
                        }
                    }},
                    {"string_fields": {
                        "match": "*",
                        "match_mapping_type": "string",
                        "mapping": {
                            "type": "string", "index": "not_analyzed"
                        }
                    }
                    }]
            }
        }
        }
    es.indices.delete(index=index, ignore=[400, 404])
    es.indices.create(index=index, ignore=[400, 404], body=mapping)

    consume(
        streaming_bulk(es, map(lambda doc: es_action(doc, index=index, id_key=key_col), gen()), chunk_size=chunk_size),
        None)
    es.indices.refresh(index=index)
    return


def pd_to_es(df, es, index=None, cols=None, key_col=None):
    if key_col in df.columns:
        df.index = df[key_col]
        drop_columns_if_exists(df, ['index', key_col])

    df.index.name = None

    if cols:
        df = df[cols]

    profiles_to_es(es=es, df=df, index=index, key_col=key_col, chunk_size=10000, mapping=None)


def es_poll(options=None, expected_count=0, index=None, sleep_interval=500e-3):
    """

    Args:
        options
        expected_count
        index: can be a single index or a list of indices to count
        sleep_interval: poll interval
    """
    es = elasticsearch.Elasticsearch(['http://localhost:{}/'.format(options.es_port)],
                                     timeout=120, max_retries=10,
                                     retry_on_timeout=True)

    def get_count():
        try:
            if isinstance(index, list):
                _count = sum(es.count(index=idx, doc_type=idx).get('count', 0) for idx in index)
            else:
                _count = es.count(index=index, doc_type=index).get('count', 0)
        except:
            _count = 0
            log.info('{}', traceback.format_exc())
        return _count

    count = get_count()
    while count < expected_count:
        log.info('count: {}/{}', count, expected_count)
        time.sleep(sleep_interval)
        count = get_count()

    log.info('count: {}/{} [{}]', count, expected_count, index)


def es_to_dict(es=None, index=None, fields=['item_ids'], query=None, options=None):
    """
    Args:
        es: Elasticsearch connection
        fields: list of user fields to include
        index:
        query:
        options:
    Returns:
        dict
             {'rudzud': 'item_ids': [u'33644', u'40815', u'6523', u'67171', ...],
               ...
             }

         This one really stung me..!
         https://www.elastic.co/guide/en/elasticsearch/reference/master/search-request-fields.html
         'Field values fetched from the document it self are always returned as an array.
          Metadata fields like _routing and _parent fields are never returned as an array.'
    """
    if es is None:
        es = elasticsearch.Elasticsearch(['http://localhost:{}/'.format(options.es_port)],
                                         timeout=120, max_retries=10,
                                         retry_on_timeout=True)

    log.info('converting to dict: {} -> {}'.format(index, fields))
    if query is None:
        query = {"_source": fields, "query": {"match_all": {}}}
    return {doc.get('_id'): doc.get('_source') for doc in
            scan(es, query=query, index=index, **SCAN_KWARGS)}


def user_seed_item_gen(dict_user_profiles):  # , max_users=None):
    gen = six.iteritems(dict_user_profiles)

    for user_id, doc in gen:
        item_ids = doc.get('item_ids')
        for seed_item_id in item_ids:
            yield {"user_id": user_id, "seed_item_id": seed_item_id}
    return


def user_seed_item_gen_es(es, max_users=None):
    query = {"query": {"match_all": {}}, "fields": ['item_ids']}
    gen = scan(es, query=query, index='users', **SCAN_KWARGS)

    if max_users is not None:
        gen = islice(gen, max_users)

    for doc in gen:
        user_id = doc.get('_id')
        item_ids = doc.get('fields').get('item_ids')
        for seed_item_id in item_ids:
            yield {"user_id": user_id, "seed_item_id": seed_item_id}
    return


def create_index(es, index=None, mapping=None, settings=None):
    log.info("deleting old index {}", index)
    # we clear out the old collection if it exists and create it from scratch
    if mapping is None:
        mapping = {"mappings": {
            index: {
                "_all": {"enabled": False},
                "_source": {"enabled": True},
                "dynamic_templates": [
                    {"averages": {
                        "match": "*_avg",
                        "mapping": {
                            "type": "double"
                        }
                    }},
                    {"averages_": {
                        "match": "*_avg_*",
                        "mapping": {
                            "type": "double"
                        }
                    }},
                    {
                        "string_fields": {
                            "match": "*",
                            "match_mapping_type": "string",
                            "mapping": {
                                "type": "string", "index": "not_analyzed"
                            }
                        }
                    }]
            }
        }
        }

    if settings is not None:
        mapping['settings'] = settings
    es.indices.delete(index=index, ignore=[400, 404])
    es.indices.create(index=index, ignore=[400, 404], body=mapping)
    return
