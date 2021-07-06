import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate
import pandas as pd
import json
import h5py
import math
import yaml

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier

from models import LittleLamb
from config import Config

def reporting(ensem_preds, targets):
    best_th = 0
    best_score = 0

    for th in np.arange(0.0, 0.6, 0.01):
        pred = (ensem_preds > th).astype(int)
        score = f1_score(targets, pred)
        if score > best_score:
            best_th = th
            best_score = score

    print(f"\nAUC score: {roc_auc_score(targets, ensem_preds):12.4f}")
    print(f"Best threshold {best_th:12.4f}")

    preds = (ensem_preds > best_th).astype(int)
    # print(classification_report(targets, preds, digits=3))

    cm1 = confusion_matrix(targets, preds)
    print('\nConfusion Matrix : \n', cm1)
    total1=sum(sum(cm1))

    print('\n=============')
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print (f'Accuracy    : {accuracy1:12.4f}')

    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print(f'Sensitivity : {sensitivity1:12.4f}')

    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print(f'Specificity : {specificity1:12.4f}')

def fit_and_predict(model, X,y, X_test, columns, seed=42, is_single=False):
    ensem_preds = []
    ensem_tests = []
    print("[*] model", model.__class__)
    aucs = []
    preds = []
    test_preds = []
    targets = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[valid_idx]
        y_val = y[valid_idx]
        targets.append(y_val)
        print(f"fold {fold} train ratio {np.mean(y_train):.02f} val ratio {np.mean(y_val):.02f}")
        model.fit(X_train, y_train)
        if is_single:
            preds.append(model.predict_proba(X_val)[:,1])
            aucs.append(roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
            test_preds.append(model.predict_proba(X_test)[:,1])
        else:
            preds.append(model.predict_proba(X_val))
            aucs.append(roc_auc_score(y_val, model.predict_proba(X_val)))
            test_preds.append(model.predict_proba(X_test))
    
    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    ensem_preds.append(preds)
    test_preds = np.array(test_preds).mean(axis=0)
    ensem_tests.append(test_preds)

    print("(!) cv5 AUC ", np.mean(aucs), np.std(aucs))
    if hasattr(model, "feature_importances_"):
        print('='*20)
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,columns)), columns=['Value','Feature'])
        fi = feature_imp.sort_values(by='Value', ascending=False)[['Feature', 'Value']].head(10)
        print(tabulate(fi, headers='keys', tablefmt='psql'))
        return ensem_preds, targets, ensem_tests, fi

    print('>>'*40)
    return ensem_preds, targets, ensem_tests

if __name__ == '__main__':
    trial = yaml.safe_load(open('params.yaml'))['evaluation']['trials']
    seed = yaml.safe_load(open('params.yaml'))['evaluation']['seed']
    model_index = yaml.safe_load(open('params.yaml'))['evaluation']['model_index']
    collection_index = yaml.safe_load(open('params.yaml'))['evaluation']['collection_index']
    n_est = yaml.safe_load(open('params.yaml'))['evaluation']['n_est']
    cbt = yaml.safe_load(open('params.yaml'))['evaluation']['cbt']
    ss = yaml.safe_load(open('params.yaml'))['evaluation']['ss']

    # load feature
    if collection_index == 2:
        f = h5py.File(str(Config.FEATURES_PATH / "collection2.h5"), 'r')
    elif collection_index == 3:
        f = h5py.File(str(Config.FEATURES_PATH / "collection3.h5"), 'r')
    else:
        f = h5py.File(str(Config.FEATURES_PATH / "collection5.h5"), 'r')

    X = np.array(f["X"])
    X_test = np.array(f["X_test"])
    y = np.array(f["y"])
    columns = list(f["columns"])
    uuids = list(f['uuids'])
    pos_scale = float(np.array(f.get('pos_scale')))

    # load model
    if model_index == 1:
        model = lgb.LGBMClassifier(class_weight='balanced', 
                                            num_leaves=31, 
                                            max_depth=-1, 
                                            min_child_samples=2, 
                                            learning_rate=0.02,
                                            n_estimators=n_est, 
                                            colsample_bytree=cbt, 
                                            subsample=ss, 
                                            n_jobs=-1, 
                                            random_state=0+seed
                                           )
    elif model_index == 2:
        modle = xgb.XGBClassifier(max_depth=12,
                                           scale_pos_weight=1.,
                                           learning_rate=0.01, 
                                           n_estimators=n_est,
                                           subsample=ss, 
                                           colsample_bytree=cbt,
                                           nthread=-1,
                                           seed=0+seed,
                                           eval_metric='logloss'
                                          )
    elif model_index == 3:
        model = ExtraTreesClassifier(class_weight='balanced', 
                                              bootstrap=False, 
                                              criterion='entropy', 
                                              max_features=cbt, 
                                              min_samples_leaf=4, 
                                              min_samples_split=3, 
                                              n_estimators= n_est, 
                                              random_state=0+seed, 
                                              n_jobs=-1)
    else:
        model = LittleLamb(seed=(seed + 10*trial), nest_lgb=2., nest_xgb=2., nest_et=2., cbt=0.5, ss=0.5, pos_scale=pos_scale)
    
    if model_index > 0:
        ensem_preds, targets, ensem_tests, fim = fit_and_predict(model, X, y.astype(int), X_test, columns, seed=seed, is_single=True)
        fim.to_json(str(Config.FI_FILE_PATH))
    else:
        ensem_preds, targets, ensem_tests = fit_and_predict(model, X, y.astype(int), X_test, columns, seed=seed, is_single=False)


    ensem_tests = np.array(ensem_tests).mean(axis=0)
    ensem_preds = np.array(ensem_preds).mean(axis=0)
    reporting(ensem_preds, targets)

    # generate scores
    precision, recall, prc_thresholds = precision_recall_curve(targets, ensem_preds)
    fpr, tpr, roc_thresholds = roc_curve(targets, ensem_preds)
    avg_prec = average_precision_score(targets, ensem_preds)
    roc_auc = roc_auc_score(targets, ensem_preds)

    # save scores
    with open(str(Config.METRICS_FILE_PATH), 'w') as fd:
        json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]

    # save plots
    with open(str(Config.PRC_FILE_PATH), 'w') as fd:
        json.dump(
        {
            "prc": [
                {"precision": p, "recall": r, "threshold": t}
                for p, r, t in prc_points
            ]
        },
        fd,
        indent=4,
    )

    with open(str(Config.ROC_FILE_PATH), 'w') as fd:
        json.dump(
        {
            "roc": [
                {"fpr": fp, "tpr": tp, "threshold": t}
                for fp, tp, t in zip(fpr, tpr, roc_thresholds)
            ]
        },
        fd,
        indent=4,
    )