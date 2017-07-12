import json

import elasticsearch
import pymongo
from elasticsearch.helpers import streaming_bulk
from redis import StrictRedis

from explanations.es_utils import es_poll
from explanations.logstash import start_logstash_redis
from explanations.utils import count_iter_items


class WriteBuffer(object):
    def __init__(self, options=None):
        self.options = options
        self.client = self.get_client()
        self.logstash_proc = None
        return

    def get_client(self):
        return

    def start_logstash(self):
        self.logstash_proc = None
        return

    def poll(self, index=None, expected_count=0):
        return es_poll(options=self.options, index=index, expected_count=expected_count)

    def finish(self):
        if self.logstash_proc is not None:
            self.logstash_proc.kill()
        return

    def consume(self, gen):
        return


class WriteBufferES(WriteBuffer):
    def __init__(self, options=None):
        super(WriteBufferES, self).__init__(options=options)
        return

    def get_client(self):
        es = elasticsearch.Elasticsearch(['http://localhost:{}/'.format(self.options.es_port)],
                                         **self.options.ES_ARGS)
        return es

    def consume(self, gen):
        count = count_iter_items(streaming_bulk(self.client,
                                                gen(),
                                                chunk_size=self.options.chunksize,
                                                refresh='false', consistency='one',
                                                max_chunk_bytes=524288000))
        return count


# class WriteBufferDummy(WriteBuffer):
#     def __init__(self, options=None):
#         super(WriteBufferDummy, self).__init__(options=options)
#         return
#
#     def get_client(self):
#         return None
#
#     def consume(self, gen):
#         chunk_size = self.options.chunksize
#         count = 0
#         for i, doc in enumerate(gen()):
#             count += 1
#
#         return count


class WriteBufferMongo(WriteBuffer):
    def __init__(self, options=None):
        super(WriteBufferMongo, self).__init__(options=options)
        self.db = 'db'
        self.coll = 'data'
        return

    def get_client(self):
        mongo = pymongo.MongoClient("mongodb://localhost:{}".format(self.options.mongo_port))
        return mongo

    def consume(self, gen):
        chunk_size = self.options.chunksize
        bulk = self.client[self.db][self.coll].initialize_unordered_bulk_op(bypass_document_validation=False)
        for i, doc in enumerate(gen()):
            bulk.insert(doc)
            if i % chunk_size == 0:
                bulk.execute({"w": 0})
                bulk = self.client[self.db][self.coll].initialize_unordered_bulk_op(bypass_document_validation=False)

        count = i
        if i % chunk_size != 0:
            bulk.execute({"w": 0})
        return count


class WriteBufferRedis(WriteBuffer):
    def __init__(self, options=None):
        super(WriteBufferRedis, self).__init__(options=options)
        return

    def start_logstash(self):
        self.logstash_proc = start_logstash_redis(options=self.options)
        return

    def get_client(self):
        redis = StrictRedis(port=self.options.redis_port)
        redis.flushdb()
        return redis

    def consume(self, gen):
        data = map(json.dumps, gen())
        self.client.rpush("data", *data)
        return len(data)
