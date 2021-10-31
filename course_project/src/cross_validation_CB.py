import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import catboost as cb


class CrossValidCatBoost:
    
    estimators = []
    
    def __init__(self, X, y, cat_features='auto'):
        
        self.X = X
        self.y = y
        self.cat_features = cat_features
        
        self.oof_preds = np.zeros(len(X))
        
    def train(self, params, n_splits=5, plot=True):
        
        X, y = self.X, self.y
        oof_preds = self.oof_preds
        
        train_auc, valid_auc = 0, 0
        colors = ['r', 'orange', 'g', 'c', 'b']
        
        plt.figure(figsize=(14, 8))
        plt.title('AUC')
        
        for i, (train_idx, valid_idx) in enumerate(KFold(n_splits=n_splits).split(X, y)):
            
            Xtrain, Xvalid = X.loc[train_idx], X.loc[valid_idx]
            ytrain, yvalid = y[train_idx], y[valid_idx]
            
            model = cb.CatBoostClassifier(**params)
            model.fit(
                Xtrain, ytrain,
                eval_set=[(Xvalid, yvalid)],
                cat_features=self.cat_features
            )
            
            oof_preds[valid_idx] = model.predict_proba(Xvalid)[:, 1]
            
            evals_train = model.evals_result_['learn']['AUC']
            evals_valid = model.evals_result_['validation']['AUC']
            
            train_auc += evals_train[-1]
            valid_auc += evals_valid[-1]
            
            if plot:
                plt.plot(evals_train, color=colors[i], label=f'fold = {i + 1} train', linestyle='--')
                plt.plot(evals_valid, color=colors[i], label=f'fold = {i + 1} validation')
            
            self.estimators.append(model)
            
        train_auc /= n_splits
        valid_auc /= n_splits

        print(f'\ntrain: mean: {round(train_auc, 4)}')
        print(f'valid: mean: {round(valid_auc, 4)}')
        
        plt.grid()
        plt.legend()
        
        return self.estimators
    
    def predict_proba(self, test):
        
        dtest = cb.Pool(test, cat_features=self.cat_features)

        preds = 0
        for estimator in self.estimators:
            preds += estimator.predict_proba(dtest)
            
        preds /= len(self.estimators)
        
        return preds
        
    def get_estimators(self):
        return self.estimators
    
    def get_oof_preds(self):
        return self.oof_preds