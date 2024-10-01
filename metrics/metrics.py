from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, average_precision_score
from .spot import SPOT
from .affiliation import pr_from_events
from .affiliation.generics import convert_vector_to_events
import numpy as np
import pandas as pd


def get_thres_by_SPOT(init_score, test_score, q):
    s = SPOT(q=q)
    s.fit(init_score, test_score)
    s.initialize(verbose=False)
    ret = s.run()
    threshold = np.mean(ret['thresholds'])
    return threshold

def affiliation(gt, anomaly_score, thresholds, save_path, verbose=False):
    res_info = []
    for i, threshold in enumerate(thresholds):
        pred = (anomaly_score > threshold).astype(int)
        accuracy = accuracy_score(gt, pred)
        events_pred = convert_vector_to_events(pred)
        events_label = convert_vector_to_events(gt)
        Trange = (0, len(pred))
        result = pr_from_events(events_pred, events_label, Trange)
        P = result['precision']
        R = result['recall']
        F = 2 * P * R / (P + R)
        res = {
            'threshold': threshold,        
            'accuracy_affiliation': accuracy,
            'precision_affiliation': P,
            'recall_affiliation': R,
            'F1_affiliation': F,
        }
        if verbose: print(f"threshold_{threshold}:\taffiliation_F1_{F:.4f}")
        res_info.append(res)
    pd.DataFrame(res_info).to_csv(f"{save_path}/affiliation.csv", index=False)
    return res_info

def auc_roc(gt, anomaly_score, save_path, verbose=False):
    fpr, tpr, _ = roc_curve(gt, anomaly_score)
    auc_roc = auc(fpr, tpr)
    if verbose: print(f"auc_roc_{auc_roc:.4f}")
    res = {"AUC_ROC": auc_roc}
    pd.DataFrame([res]).to_csv(f"{save_path}/auc_roc.csv", index=False)
    return auc_roc

def evaluate(gt, init_score, anomaly_score, threshold, save_path, metric="affiliation", verbose=False):
    if metric == "affiliation":
        affiliation(gt, anomaly_score, thresholds=threshold, save_path=save_path, verbose=verbose)
    elif metric == "auc_roc":
        auc_roc(gt, anomaly_score, save_path, verbose=verbose)
    