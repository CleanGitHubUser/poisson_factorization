import numpy as np, pandas as pd

df = pd.read_csv('MovieLens_100K.csv')

# print(df.head(5))

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=.2)
users_train = set(train.UserId)
items_train = set(train.ItemId)
test = test.loc[test.UserId.isin(users_train) & test.ItemId.isin(items_train)].reset_index(drop=True)
del users_train, items_train
# del df
print(df.shape)
# print(train.head(5))

import math

k = 50
mean_y = 3.52986
shape_para = 20
a = shape_para
a_prime = shape_para
b_prime = a / math.sqrt(mean_y / k)
c = shape_para
c_prime = shape_para
d_prime = c / math.sqrt(mean_y / k)

from hpfrec2 import HPF

recommender = HPF(k=k, a=a, a_prime=a_prime, b_prime=b_prime,
                  c=c, c_prime=c_prime, d_prime=d_prime, full_llk=True,
                  check_every=5,
                  stop_thr=1e-10,
                  # stop_crit='maxiter',
                  maxiter=100,
                  reindex=True,
                  allow_inconsistent_math=True,
                  ncores=-1,
                  approx_rte=False,
                  # approx_rte=True,
                  cut_extreme_initial=0.1,
                  save_folder='D:/poisson_factorization')
recommender.fit(df)
# recommender.fit(train, val_set = test)

# test['Predicted'] = recommender.predict(user=test.UserId, item=test.ItemId)
# test['RandomItem'] = np.random.choice(train.ItemId, size=test.shape[0])
# test['PredictedRandom'] = recommender.predict(user=test.UserId, item=test.RandomItem)
# print("Average prediction for combinations in test set: ", test.Predicted.mean())
# print("Average prediction for random combinations: ", test.PredictedRandom.mean())

# top N recommendation data frame
# UserId_unique = pd.Series(test.UserId).unique()
# topN_list = [recommender.topN(user = UserId) for UserId in UserId_unique]
# topN_df = pd.DataFrame(topN_list, index = UserId_unique)
# print(topN_df)

# pred = recommender.predict(user=df.UserId, item=df.ItemId)
#
# print(len(list(test.UserId)))
# print(len(list(test.ItemId)))
# print(pred)
#
# # pd.DataFrame(test).to_csv('test.csv')
#
# pd.DataFrame(pred).to_csv('pred.csv')
