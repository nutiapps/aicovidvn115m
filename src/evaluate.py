import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate
import pandas as pd
import json
import h5py

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

def fit_and_predict(model, X,y, X_test, columns, seed=42):
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
        print(f"train ratio {np.mean(y_train):.02f} val ratio {np.mean(y_val):.02f}")
        model.fit(X_train, y_train)
        preds.append(model.predict_proba(X_val))
        test_preds.append(model.predict_proba(X_test))
        aucs.append(roc_auc_score(y_val, model.predict_proba(X_val)))
    
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
    # load feature
    f = h5py.File(str(Config.FEATURES_PATH / "collection5.h5"), 'r')
    X = np.array(f["X"])
    X_test = np.array(f["X_test"])
    y = np.array(f["y"])
    columns = list(f["columns"])
    uuids = list(f['uuids'])
    pos_scale = float(np.array(f.get('pos_scale')))

    # load model
    trial = 0
    model = LittleLamb(seed=(2024 + 10*trial), nest_lgb=2., nest_xgb=2., nest_et=2., cbt=0.5, ss=0.5, pos_scale=pos_scale)
    ensem_preds, targets, ensem_tests = fit_and_predict(model, X, y.astype(int), X_test, columns, 42)

    ensem_tests = np.array(ensem_tests).mean(axis=0)
    ensem_preds = np.array(ensem_preds).mean(axis=0)
    reporting(ensem_preds, targets)

    # generate scores
    precision, recall, thresholds = precision_recall_curve(targets, ensem_preds)
    auc = auc(recall, precision)

    # save feature importance
    # fi.to_json(str(Config.FI_FILE_PATH), index=None)

    # save scores
    with open(str(Config.METRICS_FILE_PATH), 'w') as f:
        json.dump({'auc': auc}, f)

    # save plots
    with open(str(Config.PLOT_FILE_PATH), 'w') as f:
        proc_dict = {'proc': [{
            'precision': p,
            'recall': r,
            'threshold': t
            } for p, r, t in zip(precision, recall, thresholds)
        ]}
        json.dump(proc_dict, f)