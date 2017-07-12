import numpy as np
import six

from explanations.build_explanations import MIN_SESSION_LENGTH, SENTIMENT_THRESHOLD, COMPELLING_THRESHOLD


def gen_explanation_sent_diffs_item_imp(session_id=None, user_dict=None, session_items_dict=None,
                                        target_item_cols=None):
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
        item_mentions = np.array(target_item['mentions'])

        better_count = (target_item_sentiment - alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment - alternative_sentiment).sum(0)

        # normalise better scores to session_length
        better_pro_scores = better_count
        worse_con_scores = worse_count

        is_seed = seed_item_id == target_item_id

        # VERY IMPORTANT: add the target item back to the session list
        # session_items_dict[target_item_id] = target_item

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (worse_con_scores > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores[pros] * item_mentions[pros], pros[pros > 0])
        worse_con_scores_sum = np.dot(worse_con_scores[cons] * item_mentions[cons], cons[cons > 0])

        # NOTE: Strength is compute as the sum of better pros scores - sum of worse cons scores
        # NOTE: all features are important to user.
        strength = better_pro_scores_sum - worse_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (np.absolute(better_pro_scores) > COMPELLING_THRESHOLD)
        cons_comp = cons & (np.absolute(worse_con_scores) > COMPELLING_THRESHOLD)
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
        exp = {
            'explanation_id': explanation_id,
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


def gen_explanation_sent_diffs_user_imp(session_id=None, user_dict=None, session_items_dict=None,
                                        target_item_cols=None):
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
        item_mentions = np.array(target_item['mentions'])

        better_count = (target_item_sentiment - alternative_sentiment).sum(0)
        worse_count = (target_item_sentiment - alternative_sentiment).sum(0)

        # normalise better scores to session_length
        better_pro_scores = better_count
        worse_con_scores = worse_count

        is_seed = seed_item_id == target_item_id

        # VERY IMPORTANT: add the target item back to the session list
        # session_items_dict[target_item_id] = target_item

        user_imp = user_mentions > 0
        pros = (better_pro_scores > 0) & (target_item_sentiment > SENTIMENT_THRESHOLD) & user_imp
        cons = (worse_con_scores > 0) & (target_item_sentiment <= SENTIMENT_THRESHOLD) & user_imp
        n_pros, n_cons = np.count_nonzero(pros), np.count_nonzero(cons)

        better_pro_scores_sum = np.dot(better_pro_scores[pros] * user_mentions[pros], pros[pros > 0])
        worse_con_scores_sum = np.dot(worse_con_scores[cons] * user_mentions[cons], cons[cons > 0])

        # NOTE: Strength is compute as the sum of better pros scores - sum of worse cons scores
        # NOTE: all features are important to user.
        strength = better_pro_scores_sum - worse_con_scores_sum

        # TODO: fix normalisation of better/worse score
        pros_comp = pros & (np.absolute(better_pro_scores) > COMPELLING_THRESHOLD)
        cons_comp = cons & (np.absolute(worse_con_scores) > COMPELLING_THRESHOLD)
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
