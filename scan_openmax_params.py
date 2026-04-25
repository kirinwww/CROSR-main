from __future__ import print_function

import argparse
import itertools

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import numpy as np

from compute_openmax import get_scores
from evaluate_openmax import compute_threshold_metrics
from evaluate_openmax import find_best_balanced_threshold
from evaluate_openmax import find_fpr_at_target_tpr
from evt_fitting import weibull_tailfitting


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def get_args():
    parser = argparse.ArgumentParser(description='Scan OpenMax hyperparameters without retraining the model')
    parser.add_argument('--MAV_path', default="./saved_MAVs/cifar10/", type=str, help="Path to saved MAV files")
    parser.add_argument('--distance_scores_path', default="./saved_distance_scores/cifar10/", type=str, help="Path to saved distance score files")
    parser.add_argument('--feature_dir', default="./saved_features/cifar10", type=str, help="Path to saved feature files")
    parser.add_argument('--tail_sizes', default="10,20,30,50", type=str, help="Comma-separated Weibull tail sizes")
    parser.add_argument('--alpha_ranks', default="1,3,5,6", type=str, help="Comma-separated alpha ranks")
    parser.add_argument('--distance_types', default="eucos", type=str, help="Comma-separated distance types: eucos, euclidean, cosine")
    parser.add_argument('--top_k', default=10, type=int, help="How many best results to print")
    return parser.parse_args()


def evaluate_combo(mav_path, distance_scores_path, feature_dir, tail_size, alpha_rank, distance_type):
    weibull_model = weibull_tailfitting(
        mav_path,
        distance_scores_path,
        tailsize=tail_size,
        distance_type=distance_type
    )
    in_dist_scores = get_scores("val", weibull_model, feature_dir, alpha_rank, distance_type)
    open_set_scores = get_scores("open_set", weibull_model, feature_dir, alpha_rank, distance_type)

    scores = np.concatenate((in_dist_scores, open_set_scores))
    labels = np.array(([0] * len(in_dist_scores)) + ([1] * len(open_set_scores)))

    auroc = roc_auc_score(labels, scores)
    aupr_out = average_precision_score(labels, scores)
    fpr95, threshold95 = find_fpr_at_target_tpr(labels, scores, 0.95)
    best_metrics = find_best_balanced_threshold(labels, scores)
    best_threshold_metrics = compute_threshold_metrics(labels, scores, best_metrics['threshold'])

    return {
        'tail_size': tail_size,
        'alpha_rank': alpha_rank,
        'distance_type': distance_type,
        'auroc': auroc,
        'aupr_out': aupr_out,
        'fpr95': fpr95,
        'threshold95': threshold95,
        'balanced_accuracy': best_threshold_metrics['balanced_accuracy'],
        'overall_accuracy': best_threshold_metrics['overall_accuracy'],
        'ood_recall_tpr': best_threshold_metrics['ood_recall_tpr'],
        'id_recall_tnr': best_threshold_metrics['id_recall_tnr'],
        'precision_ood': best_threshold_metrics['precision_ood'],
        'threshold': best_threshold_metrics['threshold'],
    }


def format_float(value):
    if value is None:
        return "n/a"
    return "{:.6f}".format(value)


def main():
    args = get_args()

    tail_sizes = parse_int_list(args.tail_sizes)
    alpha_ranks = parse_int_list(args.alpha_ranks)
    distance_types = parse_str_list(args.distance_types)

    results = []
    for tail_size, alpha_rank, distance_type in itertools.product(tail_sizes, alpha_ranks, distance_types):
        result = evaluate_combo(
            args.MAV_path,
            args.distance_scores_path,
            args.feature_dir,
            tail_size,
            alpha_rank,
            distance_type
        )
        results.append(result)
        print(
            "tail={tail_size} alpha={alpha_rank} dist={distance_type} "
            "AUROC={auroc} AUPR_OUT={aupr_out} FPR95={fpr95} BAL_ACC={balanced_accuracy}".format(
                tail_size=result['tail_size'],
                alpha_rank=result['alpha_rank'],
                distance_type=result['distance_type'],
                auroc=format_float(result['auroc']),
                aupr_out=format_float(result['aupr_out']),
                fpr95=format_float(result['fpr95']),
                balanced_accuracy=format_float(result['balanced_accuracy']),
            )
        )

    ranked = sorted(results, key=lambda item: (item['auroc'], item['aupr_out'], item['balanced_accuracy']), reverse=True)

    print("")
    print("Top {} results by AUROC:".format(min(args.top_k, len(ranked))))
    for index, result in enumerate(ranked[:args.top_k], start=1):
        print(
            "{idx}. tail={tail_size} alpha={alpha_rank} dist={distance_type} "
            "AUROC={auroc} AUPR_OUT={aupr_out} FPR95={fpr95} "
            "BAL_ACC={balanced_accuracy} OOD_RECALL={ood_recall_tpr} "
            "ID_TNR={id_recall_tnr} OOD_PREC={precision_ood}".format(
                idx=index,
                tail_size=result['tail_size'],
                alpha_rank=result['alpha_rank'],
                distance_type=result['distance_type'],
                auroc=format_float(result['auroc']),
                aupr_out=format_float(result['aupr_out']),
                fpr95=format_float(result['fpr95']),
                balanced_accuracy=format_float(result['balanced_accuracy']),
                ood_recall_tpr=format_float(result['ood_recall_tpr']),
                id_recall_tnr=format_float(result['id_recall_tnr']),
                precision_ood=format_float(result['precision_ood']),
            )
        )


if __name__ == "__main__":
    main()
