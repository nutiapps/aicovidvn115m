from config import Config
import joblib
from models import LittleLamb
import zipfile
import os
import pandas as pd
import numpy as np
import h5py
import yaml
import time


if __name__ == '__main__':
    Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    Config.SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    N= yaml.safe_load(open('params.yaml'))['trainpredict']['trials']
    seed = yaml.safe_load(open('params.yaml'))['trainpredict']['seed']

    # Start measuring time
    start_time = time.perf_counter()

    f = h5py.File(str(Config.FEATURES_PATH / "collection2.h5"), 'r')
    X2 = np.array(f["X"])
    X2_test = np.array(f["X_test"])
    y = np.array(f["y"])
    columns2 = list(f["columns"])
    uuids = list(f['uuids'])
    pos_scale = float(np.array(f.get('pos_scale')))

    f = h5py.File(str(Config.FEATURES_PATH / "collection3.h5"), 'r')
    X3 = np.array(f["X"])
    X3_test = np.array(f["X_test"])
    columns3 = list(f["columns"])

    f = h5py.File(str(Config.FEATURES_PATH / "collection5.h5"), 'r')
    X5 = np.array(f["X"])
    X5_test = np.array(f["X_test"])
    columns5 = list(f["columns"])

    a_B = 0.0
    a_C = 0.0
    a_D = 0.0

    for t in range(N):
        print('[*] Trial ', t)
        clf = LittleLamb(seed=seed + 10*t, nest_lgb=1., nest_xgb=1., nest_et=1., cbt=0.75, ss=0.75, pos_scale=pos_scale)
        clf.fit(X2, y)
        a_B += clf.predict(X2_test)
        joblib.dump(clf, str(Config.MODELS_PATH / f"x2_{t}.h5"))
        
        clf = LittleLamb(seed=(seed + 1 + 10*t), nest_lgb=2., nest_xgb=2., nest_et=2., cbt=0.5, ss=0.5, pos_scale=pos_scale)
        clf.fit(X3, y)
        a_C += clf.predict(X3_test)
        joblib.dump(clf, str(Config.MODELS_PATH / f"x3_{t}.h5"))

        clf = LittleLamb(seed=(seed + 2 + 10*t), nest_lgb=2., nest_xgb=2., nest_et=2., cbt=0.5, ss=0.5, pos_scale=pos_scale)
        clf.fit(X5, y)
        a_D += clf.predict(X5_test)
        joblib.dump(clf, str(Config.MODELS_PATH / f"x5_{t}.h5"))
        print(f"Training time: {time.perf_counter() - start_time:.2f} seconds")
    
    a = (a_B + a_C + a_D) / 3
    a = a - min(a)
    a = a / max(a)


    sub = pd.DataFrame({'uuid': uuids, 'assessment_result': a})
    sub.to_csv("results.csv", index=None)
    with zipfile.ZipFile(str(Config.SUBMISSION_PATH / "littlelamb-5x3-collection235-abc-trim-balanced-te-null-20.zip"), 'w') as zf:
        zf.write('results.csv')

    os.remove('results.csv')