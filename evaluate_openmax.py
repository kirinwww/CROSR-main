from __future__ import print_function

import argparse
import numpy as np

from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from compute_openmax import get_scores
from evt_fitting import weibull_tailfitting


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate OpenMax with additional metrics')
    parser.add_argument('--MAV_path', default="./saved_MAVs/cifar10/", type=str, help="Path to saved MAV files")
    parser.add_argument('--distance_scores_path', default="./saved_distance_scores/cifar10/", type=str, help="Path to saved distance score files")
    parser.add_argument('--feature_dir', default="./saved_features/cifar10", type=str, help="Path to saved feature files")
    parser.add_argument('--alpha_rank', default=10, type=int, help="Alpha rank classes to consider")
    parser.add_argument('--weibull_tail_size', default=20, type=int, help="Tail size to fit")
    parser.add_argument('--distance_type', default='eucos', choices=['eucos', 'euclidean', 'cosine'], help="Distance type for Weibull fitting and OpenMax")
    parser.add_argument('--threshold', default=None, type=float, help="Optional fixed threshold on unknown probability")
    parser.set_defaults(argument=True)
    return parser.parse_args()


def find_fpr_at_target_tpr(labels, scores, target_tpr):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.where(tpr >= target_tpr)[0]
    if len(valid) == 0:
        return None, None
    idx = valid[0]
    return fpr[idx], thresholds[idx]


def compute_threshold_metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(np.int32)

    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    tpr = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    tnr = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    overall_acc = float(tp + tn) / float(len(labels))
    balanced_acc = 0.5 * (tpr + tnr)

    return {
        'threshold': threshold,
        'overall_accuracy': overall_acc,
        'balanced_accuracy': balanced_acc,
        'ood_recall_tpr': tpr,
        'id_recall_tnr': tnr,
        'fpr': fpr,
        'precision_ood': precision,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def find_best_balanced_threshold(labels, scores):
    unique_thresholds = np.unique(scores)
    best_metrics = None
    best_score = -1.0
    for threshold in unique_thresholds:
        metrics = compute_threshold_metrics(labels, scores, threshold)
        if metrics['balanced_accuracy'] > best_score:
            best_score = metrics['balanced_accuracy']
            best_metrics = metrics
    return best_metrics


def main():
    args = get_args()

    weibull_model = weibull_tailfitting(
        args.MAV_path,
        args.distance_scores_path,
        tailsize=args.weibull_tail_size,
        distance_type=args.distance_type
    )

    in_dist_scores = get_scores("val", weibull_model, args.feature_dir, args.alpha_rank, args.distance_type)
    open_set_scores = get_scores("open_set", weibull_model, args.feature_dir, args.alpha_rank, args.distance_type)

    scores = np.concatenate((in_dist_scores, open_set_scores))
    labels = np.array(([0] * len(in_dist_scores)) + ([1] * len(open_set_scores)))

    roc_auc = roc_auc_score(labels, scores)
    aupr_out = average_precision_score(labels, scores)
    precision_in, recall_in, _ = precision_recall_curve(1 - labels, 1 - scores)
    aupr_in = auc(recall_in, precision_in)

    fpr95, threshold95 = find_fpr_at_target_tpr(labels, scores, 0.95)

    print("AUROC:", roc_auc)
    print("AUPR_OUT:", aupr_out)
    print("AUPR_IN:", aupr_in)
    if fpr95 is None:
        print("FPR@95TPR: not reachable")
    else:
        print("FPR@95TPR:", fpr95)
        print("Threshold@95TPR:", threshold95)

    if args.threshold is not None:
        fixed_metrics = compute_threshold_metrics(labels, scores, args.threshold)
        print("Fixed-threshold metrics:")
        for key in sorted(fixed_metrics.keys()):
            print("  {}: {}".format(key, fixed_metrics[key]))

    best_metrics = find_best_balanced_threshold(labels, scores)
    print("Best balanced-accuracy threshold metrics:")
    for key in sorted(best_metrics.keys()):
        print("  {}: {}".format(key, best_metrics[key]))


if __name__ == "__main__":
    main()
