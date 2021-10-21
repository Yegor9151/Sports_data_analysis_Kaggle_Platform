import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import catboost as cb


def load_data(path):
    df = pd.read_csv(path)
    df.info()
    return df

class ScoresAUC:
    
    def __init__(self, score_name):
        self.score_name = score_name
        self.df = pd.DataFrame(columns=[score_name])
        
    def add_score(self, value, name):
        self.df.loc[name, self.score_name] = value
        return self.df
    
    def get_scores(self):
        return self.df
    
    def save_csv(self):
        self.df.to_csv('scoresAUC.csv')

    def load_csv(self):
        self.df = pd.read_csv('scoresAUC.csv', index_col=0)
        return self.df

def trainingCatBoost(df, target, params, cat_features=None, train_size=0.8, random_state=42, shuffle=True):
    
    cb_model = cb.CatBoostClassifier(**params)
    X, y = df.drop(target, axis=1), df[target]
    
    samples = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle
    )
    
    for sam in samples:
        print(sam.shape)
    print()
        
    Xtrain, Xvalid, ytrain, yvalid = samples
        
    dtrain = cb.Pool(Xtrain, ytrain, cat_features=cat_features)
    dvalid = cb.Pool(Xvalid, yvalid, cat_features=cat_features)
    
    cb_model.fit(dtrain, eval_set=dvalid)
    
    evals_train = cb_model.evals_result_['learn']['AUC']
    evals_valid = cb_model.evals_result_['validation']['AUC']
    
    last_eval = round(evals_valid[-1], 4)
    
    print(f'\ntrain: mean: {round(evals_train[-1], 4)}')
    print(f'valid: mean: {last_eval}')
    
    plt.figure(figsize=(10, 4))
    plt.title('AUC')
    plt.plot(evals_train, label='train')
    plt.plot(evals_valid, label='validation')
    plt.legend()
    plt.grid()
        
    return cb_model, last_eval