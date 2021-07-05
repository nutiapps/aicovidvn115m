from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier

class LittleLamb(BaseEstimator, ClassifierMixin):  
    """mixed of lgb, xgb, et"""

    def __init__(self, seed=0, nest_lgb=1.0, nest_xgb=1.0, nest_et=1.0, cbt=0.75, ss=0.75, alpha=0.5, pos_scale=1.):
        """
        Top 3 tree models
        """
        self.classes_ = [0,1]
        self.models = [
                         lgb.LGBMClassifier(class_weight='balanced', 
                                            num_leaves=31, 
                                            max_depth=-1, 
                                            min_child_samples=2, 
                                            learning_rate=0.02,
                                            n_estimators=int(100*nest_lgb), 
                                            colsample_bytree=cbt, 
                                            subsample=ss, 
                                            n_jobs=-1, 
                                            random_state=0+seed
                                           ),
                         lgb.LGBMClassifier(class_weight=None, 
                                            num_leaves=31, 
                                            max_depth=-1, 
                                            min_child_samples=2, 
                                            learning_rate=0.01,
                                            n_estimators=int(200*nest_lgb), 
                                            colsample_bytree=cbt, 
                                            subsample=ss, 
                                            n_jobs=-1, 
                                            random_state=1+seed
                                           ),
                         xgb.XGBClassifier(max_depth=12,
                                           scale_pos_weight=1.,
                                           learning_rate=0.01, 
                                           n_estimators=int(100 * nest_xgb),
                                           subsample=ss, # 0.75
                                           colsample_bytree=cbt, # 0.75
                                           nthread=-1,
                                           seed=0+seed,
                                           eval_metric='logloss',
                                           use_label_encoder=False
                                          ),
                         xgb.XGBClassifier(max_depth=6,
                                           scale_pos_weight=pos_scale,
                                           learning_rate=0.01,
                                           n_estimators=int(200 * nest_xgb),
                                           subsample=ss, # 0.75
                                           colsample_bytree=cbt, # 0.75
                                           nthread=-1,
                                           seed=1+seed,
                                           eval_metric='logloss',
                                           use_label_encoder=False
                                          ),
                         ExtraTreesClassifier(class_weight='balanced', 
                                              bootstrap=False, 
                                              criterion='entropy', 
                                              max_features=cbt, 
                                              min_samples_leaf=4, 
                                              min_samples_split=3, 
                                              n_estimators= int(100 * nest_et), 
                                              random_state=0+seed, 
                                              n_jobs=-1),
                         
                      ]
        self.weights = [(1-alpha)*1, (1-alpha)*1, (1-alpha)*1, (1-alpha)*1, (1-alpha)*1]


    def fit(self, X, y=None):
        for t, clf in enumerate(self.models):
            clf.fit(X, y)
        return self

    def predict(self, X):
        suma = 0.0
        for t, clf in enumerate(self.models):
            a = clf.predict_proba(X)[:, 1]
            suma += (self.weights[t] * a)
        return (suma / sum(self.weights))
            
    def predict_proba(self, X):      
        return (self.predict(X))