import os
import sys
from optparse import OptionParser
from pprint import pformat

import elasticsearch
import yaml

from explanations.build_profile import build_profiles_main
from explanations.es_utils import create_index
from explanations.explainmp import build_sessions_worker_mp, get_info
from explanations.log import get_logger
from explanations.utils import timeit

if '__pypy__' in sys.builtin_module_names:
    pypy = True
else:
    pypy = False

log = get_logger()


def prepare_es(options=None):
    es = elasticsearch.Elasticsearch(['http://localhost:{}/'.format(options.es_port)], **options.ES_ARGS)
    settings = {
        'index.refresh_interval': '-1',
        'index.translog.durability': 'async',
        'index.translog.interval': '65536s',
        'index.translog.flush_threshold_size': '24gb',
        'index.requests.cache.enable': 'false',
    }
    es.indices.put_settings(settings, index='_all', ignore=[400, 404])
    create_index(es, index=options.session_index, settings=settings)


@timeit
def main(options=None):
    if options is None:
        options = get_options()

    prepare_es(options=options)

    if options.build_profiles:
        build_profiles_main(options=options)
    else:
        log.info('Skipping the process of building user and item profiles ...')

    build_sessions_worker_mp(options=options)


def get_options():
    parser = OptionParser()
    parser.add_option('-c', '--config', dest='config_file', help='Configuration file with input parameters')
    parser.add_option("--nworkers", dest='nworkers', help="number of mp workers", default=3, type=int)

    # this is for testing only- limit the number of users to save time
    parser.add_option("--max_users", dest="max_users", default=None, type=int)
    parser.add_option("--splitsize", dest="splitsize", default=1000, type=int)
    parser.add_option("--chunksize", dest="chunksize", default=500, type=int)
    parser.add_option("--build_profiles", dest="build_profiles", default=True, action='store_true')

    message = "Some columns in the explanation need to be sourced from the target item (e.g. star and average rating)"
    parser.add_option("--target_item_cols", dest="target_item_cols", default="average_rating", type=str,
                      help=message)

    # these columns will be used for ranking the explanations
    parser.add_option("--rank_cols", dest="rank_cols", type=str,
                      default="target_item_average_rating,strength,strength_comp,related_items_sims_np")

    message = "CRITICAL = 15,ERROR = 14,WARNING = 13, NOTICE = 12, INFO = 11, DEBUG = 10, TRACE = 9,NOTSET = 0"
    parser.add_option("--log", dest="loglevel", default=0, type=int, help=message)

    parser.add_option("--info", action="callback", callback=get_info, nargs=17)

    (options, args) = parser.parse_args()
    if not options.config_file:
        parser.error('Configuration file not specified')

    if not os.path.isfile(options.config_file):
        log.error('Cannot find configuration file at `{}`'.format(options.config_file))
        exit()

    config = yaml.load(open(options.config_file))
    for k, v in config.items():
        options.__dict__[k] = v

    if options.session_index is None:
        options.session_index = "{}_session".format(options.rec_index)
    if options.explanation_index is None:
        options.explanation_index = "{}_explanation".format(options.rec_index)

    options.ES_ARGS = {'timeout': 120, 'max_retries': 10, 'retry_on_timeout': True}

    log.level = options.loglevel
    log.info('Options: {}', pformat(options.__dict__, indent=4))
    return options


if __name__ == '__main__':
    main()
