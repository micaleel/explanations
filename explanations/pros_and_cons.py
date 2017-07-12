import numpy as np
import six

from explanations.build_explanations import MIN_SESSION_LENGTH, SENTIMENT_THRESHOLD, COMPELLING_THRESHOLD


def gen_explanation_pros_minus_cons(session_id=None, user_dict=None, session_items_dict=None,
                                    target_item_cols=None):
    """Generates personalized explanations where strength is computed as the number of pros minus number of cons.

        - All features are relevant to the user.
        - Strength is computed as the difference between the number of pros and the number of cons.
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
        strength = n_pros - n_cons

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
        strength_comp = n_pros_comp - n_cons_comp

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
