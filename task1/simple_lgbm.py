#!/usr/bin/env python
import numpy as np
import pandas as pd
import lightgbm


path_in = 'sample/'
path_out = 'sample/'
classes = 9


def make_features_weighted(data):
    weights = np.arange(data.shape[1], dtype=float)
    weights /= np.sum(weights)

    counters = pd.concat(
        [((data == j)*weights).sum(axis=1) for j in range(classes)], axis=1)

    data = data*weights
    return pd.concat((
        data[data!=0].mean(axis=1),
        data.mean(axis=1),
        data[data!=0].std(axis=1),
        data.std(axis=1),
        counters),
        axis=1)


train_labels = np.load(path_in + 'train_labels.npy')
train_data = pd.DataFrame(np.load(path_in + 'train_data.npy'))
test_data = pd.DataFrame(np.load(path_in + 'test_data.npy'))


model = lightgbm.LGBMClassifier(
    n_estimators=400, learning_rate=0.01, n_jobs=-1, max_depth=6)
model.fit(make_features_weighted(train_data), train_labels)


test_labels = model.predict(make_features_weighted(test_data))
np.save(path_out + 'test_labels_lgbm.npy', test_labels)
