#!/usr/bin/env python
# from gevent import monkey; monkey.patch_socket()

import multiprocessing as mp
import sys
import time
import traceback
from pprint import pformat

import elasticsearch

from explanations.build_explanations import *
from explanations.es_utils import es_action, es_to_dict, user_seed_item_gen, create_index
from explanations.log import get_logger
from explanations.pros_and_cons import *
from explanations.rankdata import rankdata
from explanations.sentiment_diff import *
from explanations.utils import split_every
from explanations.writebuffer import WriteBufferES

if '__pypy__' in sys.builtin_module_names:
    pypy = True
else:
    pypy = False

log = get_logger()


def mp_initializer(dict_item_profiles_arg, dict_user_profiles_arg, dict_rec_items_arg, options_arg):
    """
    Gets called to initialise each process in the pool

    Parameters
    ----------
    dict_item_profiles_arg
    dict_user_profiles_arg
    dict_rec_items_arg
    options_arg

    Returns
    -------

    """
    global options
    options = options_arg
    log.level = options.loglevel

    global wb
    wb = WriteBufferES(options=options)

    global dict_item_profiles
    global dict_user_profiles
    global dict_rec_items
    dict_item_profiles = dict_item_profiles_arg
    dict_user_profiles = dict_user_profiles_arg
    dict_rec_items = dict_rec_items_arg
    return


# @profile
def session_handler(obj):
    np.set_printoptions(precision=5, linewidth=300)
    MIN_SESSION_LENGTH = 9

    user_id = obj.get("user_id")
    seed_item_id = obj.get("seed_item_id")
    related_items = dict_rec_items.get(seed_item_id, {}).get('related_items', [])
    related_items_sims = dict_rec_items.get(seed_item_id, {}).get('related_items_sims', [])
    session_id = "{}#{}".format(user_id, seed_item_id)

    if seed_item_id in related_items:
        related_items.remove(seed_item_id)

    session_item_ids = related_items + [seed_item_id]
    session_item_sims = related_items_sims + [1.0]  # append a similarity value of 1 for the seed item.
    item_sims_dict = dict(list(zip(session_item_ids, session_item_sims)))
    user_dict = dict_user_profiles.get(user_id)
    # we need to filter out item_id's which are in the related_items but not in the item database
    session_items_dict = {item_id: dict_item_profiles.get(item_id) for item_id in
                          session_item_ids if dict_item_profiles.get(item_id) is not None}

    n_session_items = len(session_items_dict)
    if (n_session_items <= MIN_SESSION_LENGTH) or (n_session_items != len(related_items) + 1):
        # log.info('skipping session: {}', session_id)
        return

    session = {"user_id": user_id, "seed_item_id": seed_item_id,
               "session_id": session_id,
               "related_items": related_items,
               "index": options.session_index}

    # TODO: ranking goes here...
    strength_fn = options.strength_function
    strength_fn_args = dict(session_id=session_id, user_dict=user_dict, session_items_dict=session_items_dict,
                            target_item_cols=options.target_item_cols.split(','))
    if strength_fn in ['default', 'gen_explanation_base']:
        explanations = gen_explanation_base(**strength_fn_args)

    elif strength_fn == 'gen_explanation_only_pros':
        explanations = gen_explanation_only_pros(**strength_fn_args)

    elif strength_fn == 'gen_explanation_pros_and_cons':
        explanations = gen_explanation_pros_and_cons(**strength_fn_args)

    elif strength_fn == 'generate_explanation_all_features':
        explanations = generate_explanation_all_features(**strength_fn_args)

    elif strength_fn in ['bwhi', 'bwii', 'gen_explanation_bw_item_imp']:
        explanations = gen_explanation_bw_item_imp(**strength_fn_args)

    elif strength_fn in ['bwui', 'gen_explanation_bw_item_imp']:
        explanations = gen_explanation_bw_user_imp(**strength_fn_args)

    elif strength_fn in ['pc', 'gen_explanation_pros_minus_cons']:
        explanations = gen_explanation_pros_minus_cons(**strength_fn_args)

    elif strength_fn in ['pchi', 'pcii', 'gen_explanation_item_importance']:
        explanations = gen_explanation_item_importance(**strength_fn_args)

    elif strength_fn in ['pcui', 'gen_explanation_user_importance']:
        explanations = gen_explanation_user_importance(**strength_fn_args)

    elif strength_fn in ['bwsd', 'gen_explanation_sent_diffs']:
        explanations = gen_explanation_sent_diffs(**strength_fn_args)

    elif strength_fn in ['bwsdhi', 'gen_explanation_sent_diffs_item_imp']:
        explanations = gen_explanation_sent_diffs_item_imp(**strength_fn_args)

    elif strength_fn in ['bwsdui', 'gen_explanation_sent_diffs_user_imp']:
        explanations = gen_explanation_sent_diffs_user_imp(**strength_fn_args)

    elif strength_fn == 'gen_explanation_top_k':
        explanations = gen_explanation_top_k(k=3, **strength_fn_args)

    else:
        log.warn("Cannot find strength function '{}'. Falling back to default".format(strength_fn))
        explanations = gen_explanation_base(**strength_fn_args)

    # IMPORTANT: add related_item_sims to the explanations dictionary
    # This is necessary to enable ranking explanations by non-personalized similarites.
    # for explanation, similarity in zip(explanations, session_item_sims):
    #     explanation['related_items_sims_np'] = similarity
    for explanation in explanations:
        explanation['related_items_sims_np'] = item_sims_dict[explanation['target_item_id']]

    for col in options.rank_cols.split(','):
        ranks = rankdata([x[col] for x in explanations])
        if isinstance(ranks, (np.ndarray, np.generic)):
            ranks = ranks.tolist()
        for i, rank in enumerate(ranks):
            explanations[i]['rank_' + col] = rank

    session['explanations'] = explanations  # This makes explanations a sub-document of sessions

    yield es_action(session, index=options.session_index, id_key='session_id')

    # for explanation in explanations:
    #     yield es_action(explanation, index=options.explanation_index, id_key='explanation_id')

    return


# @profile
def mp_session_worker(session_list):
    """
    session_list = [ (u, i), ...]

    we pass each (u,i) pair to session_handler, which returns
    [session, exp1, exp2, exp3, ...]

    for each of these session, explanations objects- we construct the ES object for bulk indexing and send to the
    write handler
    """
    log.info('p_worker on session_list size: {},{}', type(session_list), len(session_list))
    count = 0
    try:
        # this simple function turns the session/explanation handling into a generator
        # for streaming to elasticsearch
        def session_gen():
            for sess in session_list:
                for obj in session_handler(sess):
                    yield obj
            return

        count += wb.consume(session_gen)
    except:
        log.info('OUCH! {}', traceback.format_exc())

    res = {'count': count}
    log.info('res: {}', res)
    return res


def build_sessions_worker_mp(options=None):
    """
            # Note: I am using imap_unordered which returns a generator. 'split_every' is also
            # a generator, so the sessions/explanations are only actually generated when we
            # iterate through the results

    """

    item_fields = ['opinion_ratio', 'item_name', 'related_items', 'polarity_ratio', 'mentions']
    user_fields = ['mentions', 'polarity_ratio', 'item_ids']
    if options.target_item_cols is not None:
        item_fields += options.target_item_cols.split(',')
    rec_fields = ['related_items', 'related_items_sims']

    dict_item_profiles = es_to_dict(index=options.item_index, fields=item_fields, options=options)
    dict_user_profiles = es_to_dict(index=options.user_index, fields=user_fields, options=options)
    dict_rec_profiles = es_to_dict(index=options.rec_index, fields=rec_fields, options=options)

    try:
        total_count = 0
        t_start = time.time()

        # this builds a generator for user x item session objects which are the base
        # for our explanations, and then splits into lists for sending to pool processes
        session_gen = split_every(options.splitsize, user_seed_item_gen(dict_user_profiles))
        # session_gen = [x.tolist() for x in np.array_split(list(user_seed_item_gen(dict_user_profiles)),
        #                                                   options.nworkers)])

        pool = mp.Pool(processes=options.nworkers, initializer=mp_initializer,
                       initargs=(dict_item_profiles, dict_user_profiles, dict_rec_profiles, options,))

        results = pool.imap_unordered(mp_session_worker, session_gen)
        log.info('results: {}', results)
        for result in results:
            log.info('result: {}', result)
            total_count += result['count']

        log.info('finished with pool')
        elapsed = time.time() - t_start
        log.info('count: {} {} {}', total_count, elapsed, float(total_count) / elapsed)

    except:
        log.info("OUCH! {}", traceback.format_exc())
    finally:
        log.info('closing pool')
        pool.close()
        pool.join()
        cleanup_es(options=options)
    return


# @profile
# TODO: ditch this function
def main(options=None):
    global es
    es = elasticsearch.Elasticsearch(['http://localhost:9200/'],
                                     timeout=120, max_retries=10, retry_on_timeout=True)
    create_index(es, index="sessions")

    global dict_item_profiles
    dict_item_profiles = es_to_dict(es, index='items',
                                    fields=['opinion_ratio', 'item_name', 'related_items', 'polarity_ratio',
                                            'mentions'])
    global dict_user_profiles
    dict_user_profiles = es_to_dict(es, index='users', fields=['mentions', 'polarity_ratio', 'item_ids'])

    max_n = 5
    session_gen = user_seed_item_gen(dict_user_profiles, max_n=max_n)
    map(mp_session_worker, split_every(10000, session_gen))
    return


def cleanup_es(options=None):
    es = elasticsearch.Elasticsearch(['http://localhost:{}/'.format(options.es_port)], **options.ES_ARGS)

    es.indices.refresh(index=options.session_index)
    # es.indices.refresh(index=options.explanation_index)

    settings = {'index.refresh_interval': '5s'}
    es.indices.put_settings(settings, index='_all')
    return


def get_info(option, opt, value, parser, *args, **kwargs):
    try:
        es_port = parser.values.es_port
    except:
        es_port = 9200

    ES_ARGS = {'timeout': 120, 'max_retries': 10, 'retry_on_timeout': True}
    es = elasticsearch.Elasticsearch(['http://localhost:{}/'.format(es_port)], **ES_ARGS)
    log.info('info: {}', pformat(es.info()))

    for index, settings in es.indices.get_settings(index='_all').items():
        if not index.startswith('.'):
            log.info('index: {}, {}', index, pformat(es.count(index=index)))
    sys.exit(0)
