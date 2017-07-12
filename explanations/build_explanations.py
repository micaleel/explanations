from __future__ import print_function, division

import numpy as np
import six

from explanations.log import get_logger

log = get_logger(format_string=(
    # u'[{record.time:%Y-%m-%d %H:%M:%S.%f}] ' +
    u'{record.func_name}:{record.lineno} {record.message}'
))

np.set_printoptions(precision=3)

# TODO: Ensure SENTIMENT_THRESHOLD is retrieved from configuration file
# TODO: Ensure COMPELLING_THRESHOLD is retrieved from configuration file
# TODO: Ensure MIN_SESSION_LENGTH is retrieved from configuration file

SENTIMENT_THRESHOLD = 0.7
COMPELLING_THRESHOLD = 0.5
MIN_SESSION_LENGTH = 2

"""
[
"target_item_sentiment -> <type 'list'>",
"is_seed -> <type 'bool'>",
"seed_item_id -> <type 'unicode'>",
"target_item_mentions -> <type 'list'>",
"session_id -> <type 'unicode'>",
"user_mentions -> <type 'list'>",
"user_sentiment -> <type 'list'>",
"item_id -> <type 'unicode'>",
"explanation_id -> <type 'unicode'>"

"better_count -> <type 'numpy.ndarray'>",
"worse_con_scores -> <type 'numpy.ndarray'>",
"better_pro_scores -> <type 'numpy.ndarray'>",
"worse_count -> <type 'numpy.ndarray'>",

]
"""


# @profile
def gen_explanation_base(session_id=None, user_dict=None, session_items_dict=None, target_item_cols=None):
    """Generate explanations using default method for computing strength"""

    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        target_item = session_items_dict.get(target_item_id)
        target_item_sentiment = np.array(target_item['polarity_ratio'])
        target_item_mentions = np.array(target_item['mentions'])

        # alternative_sentiment = np.array([s['polarity_ratio'] for s in session_items_dict.itervalues()])
        # senti_better_alternatives = (target_item_sentiment > alternative_sentiment)
        # this is the count of number of alternatives where the target has a better sentiment score
        # better_count = senti_better_alternatives.sum(0)

        better_count = (target_item_sentiment > alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment <= alternative_sentiment).sum(0) - 1

        # if its not better, then it must be worse
        # this is incorrect, there should be no -1 (-1 because the target item is included in here)
        # worse_count = session_length - better_count

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        is_seed = seed_item_id == target_item_id

        # VERY IMPORTANT: add the target item back to the session list
        # session_items_dict[target_item_id] = target_item

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (worse_con_scores > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores, pros)
        worse_con_scores_sum = np.dot(worse_con_scores, cons)

        # NOTE: Strength is compute as the sum of better pros scores - sum of worse cons scores
        # NOTE: all features are important to user.
        strength = better_pro_scores_sum - worse_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores, pros_comp)
        worse_con_scores_comp_sum = np.dot(worse_con_scores, cons_comp)
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        strength_comp = better_pro_scores_comp_sum - worse_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': target_item_mentions.tolist(),
               'target_item_sentiment': target_item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros,
               'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = target_item.get(col)

        explanations[idx] = exp
    return explanations


def gen_explanation_bw_item_imp(session_id=None, user_dict=None, session_items_dict=None, target_item_cols=None):
    """Generates explanations for a given session.

    - The strength is computed as the sum of better scores minus the sum of worse scores.
    - All features used in the strength computation are relevant to the user
    - Each better/worse scores is weighted by the importance of the feature to the target item.
    """
    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        is_seed = seed_item_id == target_item_id
        item = session_items_dict.get(target_item_id)
        item_sentiment = np.array(item['polarity_ratio'])
        item_mentions = np.array(item['mentions'])
        user_imp = user_mentions > 0
        item_imp = item_mentions > 0
        better_count = (item_sentiment > alternative_sentiment).sum(0)
        worse_count = (item_sentiment <= alternative_sentiment).sum(0) - 1

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        # Pros are features that have better scores, are higher than the sentiment_threshold
        #   and are important to the user
        # Cons are features that have worse scores, are higher than the sentiment_threshold
        #   and are important to the user
        pros = (better_pro_scores > 0) & (item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (worse_con_scores > 0) & (item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        X = better_pro_scores[pros] * item_mentions[pros]
        Y = worse_con_scores[cons] * item_mentions[cons]
        better_pro_scores_sum = np.dot(X, pros[pros > 0])
        worse_con_scores_sum = np.dot(Y, cons[cons > 0])

        # Compute strength by taking the difference between the
        #   (sum to better scores weighted by their item mention counts)
        # and
        #   (sum to worse scores weighted by their item mention counts)
        strength = better_pro_scores_sum - worse_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores[pros_comp] * item_mentions[pros_comp],
                                            pros_comp[pros_comp > 0])
        worse_con_scores_comp_sum = np.dot(worse_con_scores[cons_comp] * item_mentions[cons_comp],
                                           cons_comp[cons_comp > 0])
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        strength_comp = better_pro_scores_comp_sum - worse_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': item_mentions.tolist(),
               'target_item_sentiment': item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros,
               'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = item.get(col)

        explanations[idx] = exp
    return explanations


def gen_explanation_bw_user_imp(session_id=None, user_dict=None, session_items_dict=None, target_item_cols=None):
    """Generates explanations for a given session.

    - The strength is computed as the sum of better scores minus the sum of worse scores.
    - All features used in the strength computation are relevant to the user
    - Each better/worse scores is weighted by the importance of the feature to the target item.
    """
    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        is_seed = seed_item_id == target_item_id
        item = session_items_dict.get(target_item_id)
        item_sentiment = np.array(item['polarity_ratio'])
        item_mentions = np.array(item['mentions'])
        user_imp = user_mentions > 0
        item_imp = item_mentions > 0
        better_count = (item_sentiment > alternative_sentiment).sum(0)
        worse_count = (item_sentiment <= alternative_sentiment).sum(0) - 1

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        # Pros are features that have better scores, are higher than the sentiment_threshold
        #   and are important to the user
        # Cons are features that have worse scores, are higher than the sentiment_threshold
        #   and are important to the user
        pros = (better_pro_scores > 0) & (item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (worse_con_scores > 0) & (item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        X = better_pro_scores[pros] * user_mentions[pros]
        Y = worse_con_scores[cons] * user_mentions[cons]
        better_pro_scores_sum = np.dot(X, pros[pros > 0])
        worse_con_scores_sum = np.dot(Y, cons[cons > 0])

        # Compute strength by taking the difference between the
        #   (sum to better scores weighted by their item mention counts)
        # and
        #   (sum to worse scores weighted by their item mention counts)
        strength = better_pro_scores_sum - worse_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores[pros_comp] * user_mentions[pros_comp],
                                            pros_comp[pros_comp > 0])
        worse_con_scores_comp_sum = np.dot(worse_con_scores[cons_comp] * user_mentions[cons_comp],
                                           cons_comp[cons_comp > 0])
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        strength_comp = better_pro_scores_comp_sum - worse_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': item_mentions.tolist(),
               'target_item_sentiment': item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros,
               'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = item.get(col)

        explanations[idx] = exp
    return explanations


def gen_explanation_only_pros(session_id=None, user_dict=None, session_items_dict=None, target_item_cols=None):
    """

    Args:
        session_id:
        user_dict:
        session_items_dict:
        target_item_cols:

    Returns:

    """

    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        target_item = session_items_dict.get(target_item_id)
        target_item_sentiment = np.array(target_item['polarity_ratio'])
        target_item_mentions = np.array(target_item['mentions'])

        # alternative_sentiment = np.array([s['polarity_ratio'] for s in session_items_dict.itervalues()])
        # senti_better_alternatives = (target_item_sentiment > alternative_sentiment)
        # this is the count of number of alternatives where the target has a better sentiment score
        # better_count = senti_better_alternatives.sum(0)

        better_count = (target_item_sentiment > alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment <= alternative_sentiment).sum(0) - 1

        # if its not better, then it must be worse
        # this is incorrect, there should be no -1 (-1 because the target item is included in here)
        # worse_count = session_length - better_count

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        is_seed = seed_item_id == target_item_id

        # VERY IMPORTANT: add the target item back to the session list
        # session_items_dict[target_item_id] = target_item

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (worse_con_scores > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores, pros)
        worse_con_scores_sum = np.dot(worse_con_scores, cons)

        # NOTE: Strength is compute as the sum of better pros scores
        # NOTE: all features are important to user.
        strength = better_pro_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores, pros_comp)
        worse_con_scores_comp_sum = np.dot(worse_con_scores, cons_comp)
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        strength_comp = better_pro_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': target_item_mentions.tolist(),
               'target_item_sentiment': target_item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros, 'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = target_item.get(col)

        explanations[idx] = exp
    return explanations


def gen_explanation_pros_and_cons(session_id=None, user_dict=None, session_items_dict=None, target_item_cols=None):
    """

    Args:
        session_id:
        user_dict:
        session_items_dict:
        target_item_cols:

    Returns:

    """

    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        target_item = session_items_dict.get(target_item_id)
        target_item_sentiment = np.array(target_item['polarity_ratio'])
        target_item_mentions = np.array(target_item['mentions'])

        # alternative_sentiment = np.array([s['polarity_ratio'] for s in session_items_dict.itervalues()])
        # senti_better_alternatives = (target_item_sentiment > alternative_sentiment)
        # this is the count of number of alternatives where the target has a better sentiment score
        # better_count = senti_better_alternatives.sum(0)

        better_count = (target_item_sentiment > alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment <= alternative_sentiment).sum(0) - 1

        # if its not better, then it must be worse
        # this is incorrect, there should be no -1 (-1 because the target item is included in here)
        # worse_count = session_length - better_count

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        is_seed = seed_item_id == target_item_id

        # VERY IMPORTANT: add the target item back to the session list
        # session_items_dict[target_item_id] = target_item

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (worse_con_scores > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores, pros)
        worse_con_scores_sum = np.dot(worse_con_scores, cons)

        # NOTE: Strength is computed as the sum of better pros scores plus the sum of worse cons scores
        # NOTE: all features are important to user.
        strength = better_pro_scores_sum + worse_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores, pros_comp)
        worse_con_scores_comp_sum = np.dot(worse_con_scores, cons_comp)
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        strength_comp = better_pro_scores_comp_sum + worse_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': target_item_mentions.tolist(),
               'target_item_sentiment': target_item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros, 'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = target_item.get(col)

        explanations[idx] = exp
    return explanations


def generate_explanation_all_features(session_id=None, user_dict=None, session_items_dict=None, target_item_cols=None):
    """Generate non-personalized explanations by subtracting sum of worse scores from sum of better scores

        - Explanations are not personalized. All features are included in the explanations.
    Args:
        session_id:
        user_dict:
        session_items_dict:
        target_item_cols:

    Returns:

    """

    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        target_item = session_items_dict.get(target_item_id)
        target_item_sentiment = np.array(target_item['polarity_ratio'])
        target_item_mentions = np.array(target_item['mentions'])

        # alternative_sentiment = np.array([s['polarity_ratio'] for s in session_items_dict.itervalues()])
        # senti_better_alternatives = (target_item_sentiment > alternative_sentiment)
        # this is the count of number of alternatives where the target has a better sentiment score
        # better_count = senti_better_alternatives.sum(0)

        better_count = (target_item_sentiment > alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment <= alternative_sentiment).sum(0) - 1

        # if its not better, then it must be worse
        # this is incorrect, there should be no -1 (-1 because the target item is included in here)
        # worse_count = session_length - better_count

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        is_seed = seed_item_id == target_item_id

        # VERY IMPORTANT: add the target item back to the session list
        # session_items_dict[target_item_id] = target_item

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD)
        cons = (worse_con_scores > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD)
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores, pros)
        worse_con_scores_sum = np.dot(worse_con_scores, cons)

        # NOTE: Strength is compute as the sum of better pros scores - sum of worse cons scores
        # NOTE: all feaetures are important to user.
        strength = better_pro_scores_sum - worse_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores, pros_comp)
        worse_con_scores_comp_sum = np.dot(worse_con_scores, cons_comp)
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        strength_comp = better_pro_scores_comp_sum - worse_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': target_item_mentions.tolist(),
               'target_item_sentiment': target_item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros,
               'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = target_item.get(col)

        explanations[idx] = exp
    return explanations


def gen_explanation_user_importance(session_id=None, user_dict=None, session_items_dict=None, target_item_cols=None):
    """

    Args:
        session_id:
        user_dict:
        session_items_dict:
        target_item_cols:

    Returns:

    """

    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        target_item = session_items_dict.get(target_item_id)
        target_item_sentiment = np.array(target_item['polarity_ratio'])
        target_item_mentions = np.array(target_item['mentions'])

        # alternative_sentiment = np.array([s['polarity_ratio'] for s in session_items_dict.itervalues()])
        # senti_better_alternatives = (target_item_sentiment > alternative_sentiment)
        # this is the count of number of alternatives where the target has a better sentiment score
        # better_count = senti_better_alternatives.sum(0)

        better_count = (target_item_sentiment > alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment <= alternative_sentiment).sum(0) - 1

        # if its not better, then it must be worse
        # this is incorrect, there should be no -1 (-1 because the target item is included in here)
        # worse_count = session_length - better_count

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        is_seed = seed_item_id == target_item_id

        # VERY IMPORTANT: add the target item back to the session list
        # session_items_dict[target_item_id] = target_item

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (worse_con_scores > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores, pros)
        worse_con_scores_sum = np.dot(worse_con_scores, cons)
        user_mentions_pro_scores_sum = np.dot(user_mentions, pros)
        user_mentions_con_scores_sum = np.dot(user_mentions, cons)
        # NOTE: Strength is computed as the difference between the sum of user's mentions of pros
        # and the user's mentions of the cons.
        strength = user_mentions_pro_scores_sum - user_mentions_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores, pros_comp)
        worse_con_scores_comp_sum = np.dot(worse_con_scores, cons_comp)
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp
        user_mentions_pro_scores_comp_sum = np.dot(user_mentions, pros_comp)
        user_mentions_con_scores_comp_sum = np.dot(user_mentions, cons_comp)
        # compelling strength
        strength_comp = user_mentions_pro_scores_comp_sum - user_mentions_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': target_item_mentions.tolist(),
               'target_item_sentiment': target_item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros,
               'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = target_item.get(col)

        explanations[idx] = exp
    return explanations


def gen_explanation_item_importance(session_id=None, user_dict=None, session_items_dict=None,
                                    target_item_cols=None):
    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        target_item = session_items_dict.get(target_item_id)
        target_item_sentiment = np.array(target_item['polarity_ratio'])
        target_item_mentions = np.array(target_item['mentions'])
        better_count = (target_item_sentiment > alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment <= alternative_sentiment).sum(0) - 1

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        is_seed = seed_item_id == target_item_id

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (worse_con_scores > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores, pros)
        worse_con_scores_sum = np.dot(worse_con_scores, cons)

        # NOTE: Strength is compute as the sum of better pros scores - sum of worse cons scores
        # NOTE: all features are important to user.
        target_item_mentions_pro_scores_sum = np.dot(target_item_mentions, pros)
        target_item_mentions_con_scores_sum = np.dot(target_item_mentions, cons)

        strength = target_item_mentions_pro_scores_sum - target_item_mentions_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores, pros_comp)
        worse_con_scores_comp_sum = np.dot(worse_con_scores, cons_comp)
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        target_item_mentions_pro_scores_comp_sum = np.dot(target_item_mentions, pros_comp)
        target_item_mentions_con_scores_comp_sum = np.dot(target_item_mentions, cons_comp)
        strength_comp = target_item_mentions_pro_scores_comp_sum - target_item_mentions_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': target_item_mentions.tolist(),
               'target_item_sentiment': target_item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros,
               'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = target_item.get(col)

        explanations[idx] = exp
    return explanations


def generate_explanation_importance_no_user(session_id=None, user_dict=None, session_items_dict=None,
                                            target_item_cols=None):
    """

    Args:
        session_id:
        user_dict:
        session_items_dict:
        target_item_cols:

    Returns:

    """

    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        target_item = session_items_dict.get(target_item_id)
        target_item_sentiment = np.array(target_item['polarity_ratio'])
        target_item_mentions = np.array(target_item['mentions'])

        # alternative_sentiment = np.array([s['polarity_ratio'] for s in session_items_dict.itervalues()])
        # senti_better_alternatives = (target_item_sentiment > alternative_sentiment)
        # this is the count of number of alternatives where the target has a better sentiment score
        # better_count = senti_better_alternatives.sum(0)

        better_count = (target_item_sentiment > alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment <= alternative_sentiment).sum(0) - 1

        # if its not better, then it must be worse
        # this is incorrect, there should be no -1 (-1 because the target item is included in here)
        # worse_count = session_length - better_count

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        is_seed = seed_item_id == target_item_id

        # VERY IMPORTANT: add the target item back to the session list
        # session_items_dict[target_item_id] = target_item

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD) & (user_mentions <= 0)
        cons = (worse_con_scores > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD) & (user_mentions <= 0)
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores, pros)
        worse_con_scores_sum = np.dot(worse_con_scores, cons)

        # NOTE: Strength is compute as the sum of better pros scores - sum of worse cons scores
        # NOTE: all features are important to user.
        strength = better_pro_scores_sum - worse_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores, pros_comp)
        worse_con_scores_comp_sum = np.dot(worse_con_scores, cons_comp)
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        strength_comp = better_pro_scores_comp_sum - worse_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': target_item_mentions.tolist(),
               'target_item_sentiment': target_item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros,
               'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = target_item.get(col)

        explanations[idx] = exp
    return explanations


def gen_explanation_sent_diffs(session_id=None, user_dict=None, session_items_dict=None, target_item_cols=None):
    """Generate explanations using default method for computing strength"""

    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        target_item = session_items_dict.get(target_item_id)
        target_item_sentiment = np.array(target_item['polarity_ratio'])
        target_item_mentions = np.array(target_item['mentions'])

        is_seed = seed_item_id == target_item_id

        better_count = (target_item_sentiment > alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment <= alternative_sentiment).sum(0) - 1
        sentiment_deltas = (target_item_sentiment - alternative_sentiment).sum(0)

        # normalise better scores to session_length
        # better_pro_scores = better_count.astype(float) / session_length
        # worse_con_scores = worse_count.astype(float) / session_length
        better_pro_scores = sentiment_deltas
        worse_con_scores = sentiment_deltas

        user_imp = user_mentions > 0
        pros = (sentiment_deltas > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (sentiment_deltas > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(sentiment_deltas, pros)
        worse_con_scores_sum = np.dot(sentiment_deltas, cons)

        # NOTE: Strength is compute as the sum of better pros scores - sum of worse cons scores
        # NOTE: all features are important to user.
        strength = better_pro_scores_sum - worse_con_scores_sum

        # The sentiment threshold is a value that half the total number of items.
        # So, if the absolute difference between sentiment of the feature in the target item is
        # better/worse than a half of the items, then it's compelling.
        COMPELLING_THRESHOLD = sentiment_deltas.shape[0] / 2

        pros_comp = pros & (np.absolute(better_pro_scores) > COMPELLING_THRESHOLD)
        cons_comp = cons & (np.absolute(worse_con_scores) > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores, pros_comp)
        worse_con_scores_comp_sum = np.dot(worse_con_scores, cons_comp)
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        strength_comp = better_pro_scores_comp_sum - worse_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': target_item_mentions.tolist(),
               'target_item_sentiment': target_item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros,
               'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = target_item.get(col)

        explanations[idx] = exp
    return explanations


def topk_ind(arr, k):
    ind = np.argpartition(arr, -k)[-k:]
    res = np.zeros(arr.size, dtype=bool)
    res[ind] = True
    return res


def botk_ind(arr, k):
    ind = np.argpartition(arr, k)[:k]
    res = np.zeros(arr.size, dtype=bool)
    res[ind] = True
    return res


def gen_explanation_top_k(k=4, session_id=None, user_dict=None, session_items_dict=None,
                          target_item_cols=None):
    """

    Args:
        session_id:
        user_dict:
        session_items_dict:
        target_item_cols:

    Returns:

    """

    session_item_ids = session_items_dict.keys()
    session_length = len(session_item_ids) - 1

    if session_length < MIN_SESSION_LENGTH:
        return

    user_id, seed_item_id = session_id.split('#')
    user_mentions = np.array(user_dict['mentions'])
    user_sentiment = user_dict['polarity_ratio']
    alternative_sentiment = np.array([s['polarity_ratio'] for s in six.itervalues(session_items_dict)])
    explanations = [None] * len(session_item_ids)

    for idx, target_item_id in enumerate(session_item_ids):
        target_item = session_items_dict.get(target_item_id)
        target_item_sentiment = np.array(target_item['polarity_ratio'])
        target_item_mentions = np.array(target_item['mentions'])

        # alternative_sentiment = np.array([s['polarity_ratio'] for s in session_items_dict.itervalues()])
        # senti_better_alternatives = (target_item_sentiment > alternative_sentiment)
        # this is the count of number of alternatives where the target has a better sentiment score
        # better_count = senti_better_alternatives.sum(0)

        better_count = (target_item_sentiment > alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment <= alternative_sentiment).sum(0) - 1

        # if its not better, then it must be worse
        # this is incorrect, there should be no -1 (-1 because the target item is included in here)
        # worse_count = session_length - better_count

        # normalise better scores to session_length
        better_pro_scores = better_count.astype(float) / session_length
        worse_con_scores = worse_count.astype(float) / session_length

        is_seed = seed_item_id == target_item_id

        # VERY IMPORTANT: add the target item back to the session list
        # session_items_dict[target_item_id] = target_item

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (topk_ind(user_sentiment, k) | topk_ind(target_item_sentiment, k))
        cons = (better_pro_scores > 0) & (botk_ind(user_sentiment, k) | botk_ind(target_item_sentiment, k))
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores, pros)
        worse_con_scores_sum = np.dot(worse_con_scores, cons)

        # NOTE: Strength is compute as the sum of better pros scores - sum of worse cons scores
        # NOTE: all features are important to user.
        strength = better_pro_scores_sum - worse_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (better_pro_scores > COMPELLING_THRESHOLD)
        cons_comp = cons & (worse_con_scores > COMPELLING_THRESHOLD)
        n_pros_comp, n_cons_comp = np.count_nonzero(pros_comp), np.count_nonzero(cons_comp)

        is_comp = n_pros_comp > 0 or n_cons_comp > 0

        # average better/worse scores
        better_avg = 0.0 if n_pros == 0 else better_pro_scores_sum / n_pros
        worse_avg = 0.0 if n_cons == 0 else worse_con_scores_sum / n_cons

        # average better/worse scores (compelling)
        better_pro_scores_comp_sum = np.dot(better_pro_scores, pros_comp)
        worse_con_scores_comp_sum = np.dot(worse_con_scores, cons_comp)
        better_avg_comp = 0.0 if n_pros_comp == 0 else better_pro_scores_comp_sum / n_pros_comp
        worse_avg_comp = 0.0 if n_cons_comp == 0 else worse_con_scores_comp_sum / n_cons_comp

        # compelling strength
        strength_comp = better_pro_scores_comp_sum - worse_con_scores_comp_sum

        explanation_id = '{}##{}'.format(session_id, target_item_id)
        exp = {'explanation_id': explanation_id,
               'user_id': user_id,
               'session_id': session_id,
               'seed_item_id': seed_item_id,
               'target_item_id': target_item_id,
               'target_item_mentions': target_item_mentions.tolist(),
               'target_item_sentiment': target_item_sentiment.tolist(),
               'better_count': better_count.tolist(),
               'worse_count': worse_count.tolist(),
               'better_pro_scores': better_pro_scores.tolist(),
               'worse_con_scores': worse_con_scores.tolist(),
               'is_seed': is_seed,
               'pros': pros.tolist(),
               'cons': cons.tolist(),
               'n_pros': n_pros,
               'n_cons': n_cons,
               'strength': strength,
               'pros_comp': pros_comp.tolist(),
               'cons_comp': cons_comp.tolist(),
               'n_pros_comp': n_pros_comp,
               'n_cons_comp': n_cons_comp,
               'is_comp': is_comp,
               'better_avg': better_avg,
               'worse_avg': worse_avg,
               'better_avg_comp': better_avg_comp,
               'worse_avg_comp': worse_avg_comp,
               'strength_comp': strength_comp,
               }
        if target_item_cols is not None:
            for col in target_item_cols:
                exp["target_item_" + col] = target_item.get(col)

        explanations[idx] = exp
    return explanations


