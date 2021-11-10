import numpy as np

from functools import reduce

from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data: pd.DataFrame
cards: list
forehand = ['FH']
trump = ['trump']
feature_columns: list

def initialize(d):
    global data, cards, forehand, trump, feature_columns
    data_path = Path('data')
    data = pd.read_csv(data_path / d, header=None)

    cards = [
        # Diamonds
        'DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6',
        # Hearts
        'HA', 'HK', 'HQ', 'HJ', 'H10', 'H9', 'H8', 'H7', 'H6',
        # Spades
        'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6',
        # Clubs
        'CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6'
    ]
    user = ['user']
    data.columns = cards + forehand + user + trump

    #dropping user column for better readability
    data.drop('user', axis='columns', inplace=True)

    data.trump = data.trump.astype('category')
    data[cards+forehand] = data[cards+forehand].astype(bool)

    data.trump.cat.rename_categories({0: 'DIAMONDS', 1: 'HEARTS', 2: 'SPADES', 3: 'CLUBS',
                                      4: 'OBE_ABE', 5: 'UNE_UFE', 6: 'PUSH', 10: 'PUSH'}, inplace=True)
    data.head()

    feature_columns = cards + forehand

def optimality_search_and_eval(names, classifiers, parameters):
    X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], data.trump, test_size=0.2,
                                                        stratify=data.trump, random_state=42)
    results = []
    for name, classifier, param in zip(names, classifiers, parameters):
        print(f'Commencing Grid Search for {name}')
        gs = GridSearchCV(classifier, param_grid=param, cv=5, scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        print("Best accuracy found: {:.3f}\n".format(gs.best_score_))
        results.append([name, gs.best_score_, gs.best_estimator_])
    return results

def train_and_validate():
    pass

