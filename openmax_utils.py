

import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
import libmr
import numpy as np

from sklearn.metrics import roc_auc_score

def calc_auroc(id_test_results, ood_test_results):
    #calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
   
    # The score is the OpenMax unknown probability, so OOD samples are the positive class.
    trues = np.array(([0] * len(id_test_results)) + ([1] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result   

def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    openmax_fc8 = np.asarray(openmax_fc8, dtype=np.float64)
    openmax_score_u = np.asarray(openmax_score_u, dtype=np.float64)

    unknown_logit = np.sum(openmax_score_u)
    all_logits = np.concatenate([openmax_fc8, np.array([unknown_logit], dtype=np.float64)])

    # Stabilize exponentials to avoid overflow when feature magnitudes are large.
    max_logit = np.max(all_logits)
    exp_logits = np.exp(all_logits - max_logit)
    total_denominator = np.sum(exp_logits)

    if total_denominator == 0.0 or not np.isfinite(total_denominator):
        return 0.0

    prob_unknowns = exp_logits[-1] / total_denominator
    
    
    #modified_scores = [prob_unknowns] + prob_scores.tolist()
    #assert len(modified_scores) == (NCLASSES+1)
    #return modified_scores

    return float(prob_unknowns)

def compute_distance(query_vector, mean_vec, distance_type = 'eucos'):
    """ 

    Output:
    --------
    query_distance : Distance between respective channels

    """

    if distance_type == 'eucos':
        query_distance = spd.euclidean(mean_vec, query_vector)/200. + spd.cosine(mean_vec, query_vector)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_vec, query_vector)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_vec, query_vector)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance
    
