import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import tensorflow_hub as hub
import time
import h5py
from scipy.io import wavfile
import opensmile

from knn_features import NearestNeighborsFeats
from features import Features
from config import Config
import yaml

def make_opensmile(filepath, returns_columns=True, nb_feat=6373):
  
    if nb_feat == 25:
        smile = opensmile.Smile(
                            feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                        )
    elif nb_feat == 88:
        smile = opensmile.Smile(
                            feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals,
                        )
    else:
        smile = opensmile.Smile(
                            feature_set=opensmile.FeatureSet.ComParE_2016,
                            feature_level=opensmile.FeatureLevel.Functionals,
                        )
    # the result is a pandas.DataFrame containing the features
    y = smile.process_file(filepath)
    if returns_columns:
        return y.values, y.columns.tolist()
    else:
        return y.values

def get_duration(filepath, is_split=False):
    if is_split:
        waveform, s = get_splits(filepath)
        sample_rate = 8000
    else:
        sample_rate, waveform = wavfile.read(filepath)
    return len(waveform)/sample_rate

def calc_smooth_mean(df, by, on, m=20):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return smooth
    
def compute_SAD(sig):
	# Speech activity detection based on sample thresholding
	# Expects a normalized waveform as input
	# Uses a margin of at the boundary
    fs = 8000
    sad_thres = 0.002
    sad_start_end_sil_length = int(20*1e-3*fs)
    sad_margin_length = int(50*1e-3*fs)

    sample_activity = np.zeros(sig.shape)
    sample_activity[np.power(sig,2)>sad_thres] = 1
    sad = np.zeros(sig.shape)
    for i in range(len(sample_activity)):
        if sample_activity[i] == 1:
            sad[i-sad_margin_length:i+sad_margin_length] = 1
    sad[0:sad_start_end_sil_length] = 0
    sad[-sad_start_end_sil_length:] = 0
    return sad

def get_splits(filepath):
    sr, y = wavfile.read(filepath)
    splits = compute_SAD(y)
    if splits.sum() > 0:
        return y[splits.argmax():], 1
    else:
        return y, 0

def extract_yamnet(path):
    sample_rate, waveform = wavfile.read(path)
    scores, embeddings, spectrogram = model(waveform)
    mean_global = embeddings.numpy().mean(axis=0)
    max_global = embeddings.numpy().max(axis=0)
    out = []
    for score, embed in zip(scores, embeddings):
        top_class_indices = np.argsort(score)[::-1][:3]
        # class 42 is cough
        if 42 in top_class_indices:
            out.append(embed)
    if len(out) > 0:
        out = np.array(out)
        mean_local = out.mean(axis=0)
        max_local = out.max(axis=0)
    else:
        mean_local = np.zeros_like(mean_global)
        max_local = np.zeros_like(max_global)
    return np.concatenate((mean_global, max_global, mean_local, max_local))

from sklearn.decomposition import TruncatedSVD

def make_yamnet_feat(train, test, return_columns=True, is_svd=True, n_comp=256, seed=42):
    columns = [f"yamnet_{i}" for i in range(4096)]
    X = []
    for filename in train.file_path.values:
        X.append(extract_yamnet(filename))
    X = np.array(X)
    
    X_test = []
    for filename in test.file_path.values:
        X_test.append(extract_yamnet(filename))
    X_test = np.array(X_test)
    
    if is_svd:
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        columns = [f"svd_{i}" for i in range(n_comp)]
        X = svd.fit_transform(X)
        X_test = svd.transform(X_test)
    
    if return_columns:
        return X, X_test, columns
    else:
        return X, X_test

def process_data(meta_path, age_smooth=None, gender_smooth=None, is_train=True):
    df = pd.read_csv(meta_path)
    if is_train:
        df['file_path'] = df.file_path.apply(lambda x: str(Config.ROOT_TRAIN_DIR / f"train_audio_files_8k/{x}"))
    else:
        df['file_path'] = df.file_path.apply(lambda x: str(Config.ROOT_TEST_DIR / f"private_test_audio_files_8k/{x}"))

    df['duration_s'] = df.file_path.apply(lambda x: get_duration(x))
    df['duration_split_s'] = df.file_path.apply(lambda x: get_duration(x, is_split=True))
    
    df['null_in_age'] = df['subject_age'].isnull() + 0
    df['subject_age'] = df['subject_age'].fillna(31.)
    
    df['null_in_gender'] = df['subject_gender'].isnull() + 0
    df['subject_gender'] = df['subject_gender'].fillna('other')
    
    df['subject_age'] = df['subject_age'].astype(int)
    cut_labels = ['na', 'u25', 'u45', 'u55', 'u65', u'75', 'u100']
    cut_bins = [-100, 0, 25, 40, 55, 65, 75, 100]
    df['age_bin'] = pd.cut(df['subject_age'], bins=cut_bins, labels=cut_labels)
    
    if is_train:
        df['merger'] = df.assessment_result.astype(str) + df.age_bin.astype(str) + df.subject_gender.astype(str)
        skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
        df['fold'] = -1
        for i, (train_idx, valid_idx) in enumerate(skf.split(df, df.merger)):
            df.loc[valid_idx, 'fold'] = i
    
    df['age_bin'] = pd.factorize(df.age_bin)[0]
    df['subject_gender'] = pd.factorize(df.subject_gender)[0]
    
    if is_train:
        age_smooth = calc_smooth_mean(df, by='age_bin', on='assessment_result', m=20)
        df['age_bin_te'] = df['age_bin'].map(age_smooth)
        gender_smooth = calc_smooth_mean(df, by='subject_gender', on='assessment_result', m=20)
        df['subject_gender_te'] = df['subject_gender'].map(gender_smooth)
        return df, age_smooth, gender_smooth
    else:
        df['age_bin_te'] = df['age_bin'].map(age_smooth)
        df['subject_gender_te'] = df['subject_gender'].map(gender_smooth)
        return df

def merge_feature(Xs, cols, return_col=True):
    columns = []
    for col in cols:
        columns += col
    X = np.concatenate(Xs, axis=1)
    X = np.nan_to_num(X)
    X = np.clip(X, -np.finfo(np.float32).max, np.finfo(np.float32).max)
    if return_col:
        return X, columns
    else:
        return X

def make_basic_feat(train, test):
    numerics = ['duration_s', 
                'duration_split_s', 
                'subject_age', 
                'subject_gender', 
                'age_bin_te', 
                'subject_gender_te', 
                'null_in_gender', 
                'null_in_age'
               ]
    Xnum = train[numerics].values
    y = train['assessment_result'].values
    Xnum_test = test[numerics].values
    
    return Xnum,y, Xnum_test, numerics

def make_acoustic_feat(train, test, is_split=False):
    # Class that contains the feature computation functions 
    FREQ_CUTS = [(0,200),(300,425),(500,650),(950,1150),(1400,1800),(2300,2400),(2850,2950),(3800,3900)]
    columns = [f"acoustic_{i}" for i in range(68)]
    extractor = Features(FREQ_CUTS)
    features_fct_list = ['EEPD','ZCR','RMSP','DF','spectral_features','SF_SSTD','SSL_SD','MFCC','CF','LGTH','PSD']

    X = []
    X_test = []
    
    for filename in train.file_path.values:
        feature_values_vec = []
        if is_split:
            wav, _ = get_splits(filename)
            d = (8000, wav)
        else:
            d = wavfile.read(filename)
        columns = []
        for feature in features_fct_list:
            feature_values, feature_names = getattr(extractor,feature)(d)
            for value  in feature_values:
                if isinstance(value,np.ndarray):
                    feature_values_vec.append(value[0])
                    columns.append(feature_names[0])
                else:
                    feature_values_vec.append(value)
                    columns.append(feature_names[0])
        feature_values_vec = np.array(feature_values_vec)
        X.append(feature_values_vec)

    X = np.array(X)
    
    for filename in test.file_path.values:
        feature_values_vec = []
        if is_split:
            wav, _ = get_splits(filename)
            d = (8000, wav)
        else:
            d = wavfile.read(filename)
        columns = []
        for feature in features_fct_list:
            feature_values, feature_names = getattr(extractor,feature)(d)
            for value  in feature_values:
                if isinstance(value,np.ndarray):
                    feature_values_vec.append(value[0])
                    columns.append(feature_names[0])
                else:
                    feature_values_vec.append(value)
                    columns.append(feature_names[0])
        feature_values_vec = np.array(feature_values_vec)
        X_test.append(feature_values_vec)
    
    X_test = np.array(X_test)
    
    return X, X_test, columns

def make_opensmile_feat(train, test, return_columns=True, is_svd=True, n_comp=300, nb_feature=6373, seed=42):
    X = []
    X_test = []
    
    for filename in train.file_path.values:
        feat, columns = make_opensmile(filename, nb_feat=nb_feature)
        X.append(feat[0])
    
    X = np.array(X)
    
    for filename in test.file_path.values:
        feat, columns = make_opensmile(filename, nb_feat=nb_feature)
        X_test.append(feat[0])
    
    X_test = np.array(X_test)
    
    if is_svd:
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        columns = [f"os_svd_{i}" for i in range(n_comp)]
        X = svd.fit_transform(X)
        X_test = svd.transform(X_test)
    
    if return_columns:
        return X, X_test, columns
    else:
        return X, X_test

def get_collection1():
    X, columns = merge_feature([Xb, Xa, Xm], [columns_b, columns_a, columns_m], return_col=True)
    X_test = merge_feature([Xb_test, Xa_test, Xm_test], [columns_b, columns_a, columns_m], return_col=False)
    print(X.shape, X_test.shape)
    return X, X_test, columns

def get_collection2():
    X, columns = merge_feature([Xb, Xa2, Xm], [columns_b, columns_a, columns_m], return_col=True)
    X_test = merge_feature([Xb_test, Xa2_test, Xm_test], [columns_b, columns_a, columns_m], return_col=False)
    print(X.shape, X_test.shape)
    return X, X_test, columns

def get_collection3():
    X, columns = merge_feature([Xb, Xo, Xm], [columns_b, columns_o, columns_m], return_col=True)
    X_test = merge_feature([Xb_test, Xo_test, Xm_test], [columns_b, columns_o, columns_m], return_col=False)
    print(X.shape, X_test.shape)
    return X, X_test, columns

def get_collection5():
    X, columns = merge_feature([Xa, Xb, Xo, Xs, Xm], [columns_a, columns_b, columns_o, columns_s, columns_m], return_col=True)
    X_test = merge_feature([Xa_test, Xb_test, Xo_test, Xs_test, Xm_test], [columns_a, columns_b, columns_o, columns_s, columns_m], return_col=False)
    print(X.shape, X_test.shape)
    return X, X_test, columns

def get_collection6():
    X, columns = merge_feature([Xa, Xm, Xo, Xs], [columns_a, columns_o, columns_m, columns_s], return_col=True)
    X_test = merge_feature([Xa_test, Xm_test, Xo_test, Xs_test], [columns_a, columns_o, columns_m, columns_s], return_col=False)
    print(X.shape, X_test.shape)
    return X, X_test, columns

if __name__ == '__main__':
    seed=yaml.safe_load(open('params.yaml'))['featurization']['seed']

    Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    # Start measuring time
    start_time = time.perf_counter()

    print("[*] process data")
    train, age_smooth, gender_smooth = process_data(str(Config.DATASET_PATH / "aicv115m_public_train/metadata_train_challenge.csv"))
    test = process_data(str(Config.DATASET_PATH / "aicv115m_private_test/metadata_private_test.csv"), age_smooth, gender_smooth, is_train=False)

    pos_scale = len(train[train.assessment_result==0]) / len(train[train.assessment_result==1])
    print("pos scale", pos_scale)

    print("[*] make basic feature")
    Xb, y, Xb_test, columns_b = make_basic_feat(train,test)
    print(Xb.shape, y.shape, Xb_test.shape)
    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")

    print("[*] make acoustic feature")
    Xa, Xa_test, columns_a = make_acoustic_feat(train, test, is_split=False)
    print(Xa.shape, Xa_test.shape)
    Xa2, Xa2_test, columns_a = make_acoustic_feat(train, test, is_split=True)
    print(Xa2.shape, Xa2_test.shape)
    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")

    print("[*] make open smile feature")
    Xo, Xo_test, columns_o = make_opensmile_feat(train, test, is_svd=False, nb_feature=6373)
    print(Xo.shape, Xo_test.shape)
    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")

    print("[*] make yamnet feature")
    Xs, Xs_test, columns_s = make_yamnet_feat(train,test, seed=seed)
    print(Xs.shape, Xs_test.shape)
    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")

    print("[*] make knn feature")
    NNF = NearestNeighborsFeats(n_jobs=4, k_list=[3,8,32], metric='minkowski')
    NNF.fit(Xs, y)

    Xm_test = NNF.predict(Xs_test)

    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    NNF = NearestNeighborsFeats(n_jobs=4, k_list=[3,8,16], metric='minkowski')
    Xm = cross_val_predict(NNF,Xs,y,cv=skf,n_jobs=1)

    print("train vs test shape", Xm.shape, Xm_test.shape)
    columns_m = [f"yn_knn_{i}" for i in range(23)]
    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")

    print("[*] prepare dataset")
    print(">> collection 2")
    X, X_test, columns = get_collection2()
    columns = [c.encode('utf-8') for c in columns]
    with h5py.File(str(Config.FEATURES_PATH / "collection2.h5"), "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("pos_scale", data=pos_scale)
        f.create_dataset("y", data=y)
        f.create_dataset("columns", data=columns)
        f.create_dataset("uuids", data=[user.encode('utf-8') for user in test.uuid.values])

    print(">> collection 3")
    X, X_test, columns = get_collection3()
    columns = [c.encode('utf-8') for c in columns]
    with h5py.File(str(Config.FEATURES_PATH / "collection3.h5"), "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("pos_scale", data=pos_scale)
        f.create_dataset("y", data=y)
        f.create_dataset("columns", data=columns)
        f.create_dataset("uuids", data=[user.encode('utf-8') for user in test.uuid.values])

    print(">> collection 5")
    X, X_test, columns = get_collection5()
    columns = [c.encode('utf-8') for c in columns]
    with h5py.File(str(Config.FEATURES_PATH / "collection5.h5"), "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("pos_scale", data=pos_scale)
        f.create_dataset("y", data=y)
        f.create_dataset("columns", data=columns)
        f.create_dataset("uuids", data=[user.encode('utf-8') for user in test.uuid.values])

    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")