import os
import subprocess

from explanations.log import get_logger

log = get_logger()


def start_logstash_redis(options=None):
    logstash_config = """
    input
    {
        redis {
            data_type = > "list"
            key = > "%%SESSION_KEY%%"
            codec = > "json"
            batch_count = > %%BATCH_COUNT%%
            threads = > %%THREADS%%
            type = > "%%SESSION_KEY%%"
        }
        redis {
            data_type = > "list"
            key = > "%%EXPLANATION_KEY%%"
            codec = > "json"
            batch_count = > %%BATCH_COUNT%%
            threads = > %%THREADS%%
            type = > "%%EXPLANATION_KEY%%"
        }
    }

    filter {
        metrics {
            meter = > "sessions"
            add_tag = > "metric"
        }
    }

    output {
        if "metric" in [tags] {
            stdout {
                codec = > line {
                    format = > "rate (per second in 1m): %{[sessions][rate_1m]} count: %{[sessions][count]}"}
                }
        }
        else {
            elasticsearch {
                action = > "index"
                codec = > "json"
                index = > "%{type}"
                document_type = > "%{type}"
            }
        }
    }
    """
    logstash_config = logstash_config.replace('%%THREADS%%', '{}'.format(options.redis_threads))
    logstash_config = logstash_config.replace('%%BATCH_COUNT%%', '{}'.format(options.redis_batch_count))
    logstash_config = logstash_config.replace('%%SESSION_KEY%%', '{}'.format(options.session_index))
    logstash_config = logstash_config.replace('%%EXPLANATION_KEY%%', '{}'.format(options.explanation_index))
    logstash_config = logstash_config.replace('\n', '')
    logstash_cmd = "{}/bin/logstash".format(options.logstash_basedir)

    # os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_112.jdk/Contents/Home"
    os.environ['LS_HEAP_SIZE'] = '2g'
    # es = elasticsearch.Elasticsearch(['http://localhost:{}/'.format(options.es_port)], timeout=120, max_retries=10, retry_on_timeout=True)
    # es.indices.delete(options.session_index, ignore=[400, 404])
    # create_index(es, index=options.session_index)

    cmd = [logstash_cmd,
           ' -e ', "'{}'".format(logstash_config),
           # '--config', './logstash_py.conf',
           ' --verbose ',
           ' --pipeline-workers ', str(options.ls_pipeline_workers),
           ' --pipeline-batch-size ', str(options.ls_pipeline_batch_size),
           ' --log ', ' ./logstash.log'
           ]
    log.info('cmd {}', ''.join(cmd))
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    return proc
