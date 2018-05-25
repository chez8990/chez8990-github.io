import pandas as pd
import numpy as np
import os
import sys
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df = pd.read_excel(r'data\data.xlsx', header=0)
df.dropna(inplace=True)

df['flight_date'] = pd.to_datetime(df['flight_date'])
df['month'] = df['flight_date'].apply(lambda x: x.month)
df['day'] = df['flight_date'].apply(lambda x: x.day)

df['sin_week'] = np.sin(2*np.pi*df['Week']/52)
df['cos_week'] = np.cos(2*np.pi*df['Week']/52)
df['sin_hour'] = np.sin(2*np.pi*df['std_hour']/24)
df['cos_hour'] = np.cos(2*np.pi*df['std_hour']/24)
df['sin_month'] = np.sin(2*np.pi*df['month']/12)
df['cos_month'] = np.cos(2*np.pi*df['month']/12)
df['sin_day'] = np.sin(2*np.pi*df['day']/31)
df['cos_day'] = np.cos(2*np.pi*df['day']/31)

df['x'] = np.cos(df['Latitude_arrival']) * np.cos(df['Longitude_arrival'])
df['y'] = np.cos(df['Latitude_arrival']) * np.sin(df['Longitude_arrival'])


loc_var = df[['x', 'y', 'Altitude_arrival', 'Distance']]

df[['x', 'y', 'Altitude_arrival', 'Distance']] = \
    (loc_var - loc_var.min(axis=0))/(loc_var.max(axis=0) - loc_var.min(axis=0))

df['Airline'] = df['Airline'].fillna('Na')
df = df[df['delay_time'] != 'Cancelled']

df['delay_time'] = df['delay_time'].astype(float)
df = df[(df['delay_time']<=10) & (df['delay_time']>=-1)]
# df['delayed'] = df['delay_time'].apply(lambda x: 1 if x>=.3 else 0)

country = df['Country_arrival']

df.drop(['Week', 'Departure', 'flight_id',
         'flight_date', 'month',
         'day', 'std_hour', 'Country_arrival',
         'Latitude_arrival', 'Longitude_arrival'], inplace=True, axis=1)

X = df.drop('delay_time', axis=1)
y = df['delay_time']

del df

cat_var = {'flight_no':0,
           'Arrival':1,
           'Airline':2} # corresponds to flight_no, Arrival, Airline

embedding_dim = [50, 10, 50]

# turn to label encoding
mtl = MultitargetLabelEncoder(cat_var.keys())
X = mtl.fit_transform(X)
n_categories = [len(enc.classes_) for enc in mtl.lab_enc.values()]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=.1)

# impact encoding
impenc = ImpactEncoding()
X_train_cat = impenc.fit_transform(X_train, y_train, list(cat_var.values()))
X_test_cat = impenc.transform(X_test, list(cat_var.values()))

# catboost

cgbm = CatBoostRegressor(iterations=20,
                          learning_rate=.01,
                          depth=5)
cgbm.fit(X_train, y_train, [0, 1, 2])
cgbm_pred = cgbm.predict(X_test)

# light gbm to test the accuracy of the target encoding scheme
lgbm = LGBMRegressor()
lgbm.fit(X_train_cat, y_train)

lgbm_pred = lgbm.predict(X_test_cat)

# entity embedding
inputs = [X_train.iloc[:, 0].values, X_train.iloc[:, 1].values,
          X_train.iloc[:, 2].values, X_train.iloc[:, 3:].values]


test_inputs = [X_test.iloc[:, 0].values, X_test.iloc[:, 1].values,
               X_test.iloc[:, 2].values, X_test.iloc[:, 3:].values]


embed_model = entity_embedding_model(3, n_categories, 13, embedding_dim, loss='mean_squared_error')
hist = embed_model.fit(inputs, y_train, epochs=5, batch_size=1028, validation_split=.1, shuffle=False)

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()

entity_pred = embed_model.predict(test_inputs, batch_size=512, verbose=1)


fpr, tpr, _ = roc_curve(y_test, cgbm_pred)
plt.plot(fpr, tpr, label='catboost_distance')


fpr, tpr, _ = roc_curve(y_test, entity_pred)
plt.plot(fpr, tpr, label='entity_embedding_distance')

fpr, tpr, _ = roc_curve(y_test, lgbm_pred)
plt.plot(fpr, tpr, label='impact_encoding_distance')

plt.legend()

flight_weights = embed_model.layers[3].get_weights()[0]
arrival_weights = embed_model.layers[4].get_weights()[0]
airline_weights = embed_model.layers[5].get_weights()[0]

tsne = TSNE(n_components=2, learning_rate=300, method='exact')

arrivals = tsne.fit_transform(arrival_weights)

countries = country.iloc[X.drop_duplicates('Arrival').sort_values('Arrival').index]

for i, coun in enumerate(countries):
    if coun =='China':
        plt.scatter(arrivals[i, 0], arrivals[i, 1], label=coun, c='blue')
    elif coun == 'Japan':
        plt.scatter(arrivals[i, 0], arrivals[i, 1], label=coun, c='red')
    elif coun == 'United Arab Emirates':
        plt.scatter(arrivals[i, 0], arrivals[i, 1], label=coun, c='green')
plt.legend()

for label, x, y in zip(mtl.lab_enc['Arrival'].classes_, arrivals[:, 0], arrivals[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
