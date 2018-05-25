import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import *

class MultitargetLabelEncoder(TransformerMixin):
    """
    Fit a label encoder per column
    """

    def __init__(self, cols=None):
        self.cols = cols
        self.lab_enc = {col: LabelEncoder() for col in cols}

    def fit(self, df):
        for col in self.cols:
            self.lab_enc[col].fit(df[col])

        return self

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            df[col] = self.lab_enc[col].transform(df[col])

        return df

    def fit_transform(self, df):
        df = df.copy()
        for col in self.cols:
            df[col] = self.lab_enc[col].fit_transform(df[col])

        return df


class ImpactEncoding(TransformerMixin):
    def __init__(self,
                 min_samples_leaf=100,
                 smoothing_param=10,
                 out_fold=20,
                 in_fold=10):

        self.trust = min_samples_leaf
        self.smooth = smoothing_param
        self.out_fold = out_fold
        self.in_fold = in_fold

    def _sigmoid_smoother(self, X):
        return 1 / (1 + np.exp(-(X - self.trust) / self.smooth))

    def _cal_mean(self, X, y, prior):
        feat = X

        temp = pd.concat([feat, y], axis=1)

        # Compute target mean
        averages = temp.groupby(by=feat.name)[y.name].agg(["mean", "count"])

        # Compute smoothing
        smoothing = self._sigmoid_smoother(averages['count'])

        # The bigger the count the less full_avg is taken into account
        averages['average'] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)

        del averages.index.name

        return averages

    def fit(self, X, y, cat_feat):
        self.fit_transform(X, y, cat_feat)

        return self

    def transform(self, X, cat_feat=None):
        X = X.copy()

        if cat_feat is None:
            cat_feat = range(0, X.shape[1])

        for cat in cat_feat:
            feat = X.iloc[:, cat]

            averages = self.S[feat.name]

            new_feat = feat.to_frame(feat.name).join(averages,
                                                     on=feat.name)['average'].fillna(self.prior)

            X[feat.name] = new_feat

        return X

    def fit_transform(self, X, y, cat_feat=None):

        X = X.fillna('Na')

        self.target = y

        if cat_feat is None:
            cat_feat = range(0, X.shape[1])

        self.prior = y.sum() / len(y)
        self.S = dict()

        impact_encoded = pd.DataFrame(index=X.index)

        i = 0

        for cat in cat_feat:

            feat = X.iloc[:, cat]

            outer_folds = KFold(self.out_fold).split(feat)
            #             outer_folds_mean = pd.DataFrame()

            feat_encoded = pd.DataFrame()

            for infold, oof in outer_folds:
                # subset the inner fold for training
                infold_feat = feat.iloc[infold]
                infold_target = y.iloc[infold]

                # further cv splits on subset
                inner_folds = KFold(self.in_fold).split(infold_feat)

                # record p(y|x) from the inner cv folds
                inner_folds_mean = pd.Series()

                for infold_inner, oof_inner in inner_folds:
                    # subset the subset of the data
                    infold_feat_inner = infold_feat.iloc[infold_inner]
                    infold_target_inner = infold_target.iloc[infold_inner]

                    # provide a small prior
                    inner_prior = infold_target_inner.mean()

                    # calculate p(y|x) and record it in innner_folds_mean
                    # this will be a df with n_category columns and 1 row of p(y|x)
                    small_average = self._cal_mean(infold_feat_inner, infold_target_inner, inner_prior)
                    inner_folds_mean = pd.concat([inner_folds_mean, small_average], axis=1)

                    # record the small p(y|x) from in-fold for out-of-fold indices to impact_encoded
                    # this is to simulate the lack of target in a testing data environment
                    small_average = feat.iloc[oof].to_frame().join(small_average, on=feat.name)['average'].fillna(
                        inner_prior)

                    # record the overall p(y|x) for this feature to feat_encoded
                    feat_encoded = pd.concat([feat_encoded, small_average], axis=0)

                    #                 outer_folds_mean = pd.concat([outer_folds_mean, inner_folds_mean.mean(axis=1)], axis=1)

            # feat_encoded has multiple indices per row due to multiple oof cocatenation
            # combine them via a groupby index operation and take the mean
            feat_encoded = feat_encoded.groupby(feat_encoded.index).mean()

            impact_encoded[feat.name] = feat_encoded

            #             averages = outer_folds_mean.mean(axis=1)
            averages = self._cal_mean(feat, y, self.prior)
            averages.name = feat.name

            self.S[feat.name] = averages

        X.iloc[:, cat_feat] = impact_encoded

        return X


class CatImputer(TransformerMixin):
    def __init__(self, cols, mode='mode'):
        """
        Imputer with the most frequent observation of categories
        """

        self.mode = mode
        self.cols = cols

    def fit(self, df, fill_val=None):

        imputer_val = []
        self.n_col = df.shape[1]
        if self.mode == 'mode':
            for col in self.cols:
                col_series = df[col]
                mode = col_series.mode()[0]
                imputer_val.append(mode)

        else:
            if fill_val is None:
                raise AttributeError('Provide values to fill na with')
            if len(fill_val) != self.n_col:
                raise AttributeError('There must be enough imputation values as there are columns to be imputed')

            imputer_val = fill_val

        self.fill = imputer_val
        return self

    def transform(self, df):
        df = df.copy()

        for col, val in zip(self.cols, self.fill):
            df[col] = df[col].fillna(val)

        return df

    def fit_transform(self, df, fill_val=None):
        imputer_val = []
        df = df.copy()

        self.n_col = df.shape[1]

        if self.mode == 'mode':
            for col in self.cols:
                col_series = df[col]
                mode = col_series.mode()[0]
                imputer_val.append(mode)

                df[col] = col_series.fillna(mode)

        else:
            if fill_val is None:
                raise AttributeError('Provide values to fill na with')
            if len(fill_val) != self.n_col:
                raise AttributeError('There must be enough imputation values as there are columns to be imputed')

            for col, val in zip(self.cols, fill_val):
                col_series = df[col]

                df[col] = col_series.fillna(val)

            imputer_val = fill_val

        self.fill = imputer_val
        return df


def replace_unseen_values(df_train, df_test, cols=None, replace_val=None):
    """
    Replace any categorical states that do not exists in the train set with a preset value
    """

    for col, val in zip(cols, replace_val):
        difference_val = set(df_test[col].unique()) - set(df_train[col].unique())

        if difference_val:
            df_test[col].replace(difference_val, [val] * len(difference_val), inplace=True)

    return df_test


def embedding_layer(n_categories, embedding_dim, name=None):
    """
    Builds an embedding layer according to specification
    args:
        n_categories: int, the number of categories for this variable.
        embedding_dim: int, the dimension of the latent space.
        name: str, name_space for the embedding layer.
    """

    input_tensor = Input(shape=(1,))
    x = Embedding(n_categories, embedding_dim, name=name)(input_tensor)
    x = Reshape(target_shape=(embedding_dim,))(x)

    return input_tensor, x


def entity_embedding_model(n_catvar,
                           categories,
                           n_contvar,
                           embedding_dims,
                           output_dim=1,
                           activation=None,
                           loss='binary_crossentropy',
                           name=None):
    """
    Build the entity embedding model
    args:
        n_catvar: int, the number of categorical variables.
        categories: list, the number of categories in each categorical variables.
        n_contvar: int, number of continuous variable.
        embedding_dims: list, the dimension of the latent space for each categorical variables.
        objective: str, either Classification or Regression.
        name: str, general name for each embedding layer.
    """

    assert n_catvar == len(categories), "Provide number of categories for each categorical variables"
    assert n_catvar == len(embedding_dims), "Provide embedding dimensions for each categorical variables"

    input_model = []
    output_embedding = []

    if name is None:
        name = "categorical_variable"

    for i, (c, dim) in enumerate(zip(categories, embedding_dims)):
        input_tensor, output_tensor = embedding_layer(c, dim, name + '_{}'.format(i))

        input_model.append(input_tensor)
        output_embedding.append(output_tensor)

    # Also add an input layer for the continuous variables
    continuous_input = Input(shape=(n_contvar,), name='Continuous')
    reshape_layer = Reshape(target_shape=(n_contvar,))(continuous_input)

    input_model.append(continuous_input)
    output_embedding.append(reshape_layer)

    output_model = Concatenate()(output_embedding)
    output_model = Dense(82)(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dropout(0.2)(output_model)
    output_model = Dense(41)(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dropout(0.2)(output_model)
    output_model = Dense(output_dim)(output_model)

    if activation is not None:
        output_model = Activation(activation)(output_model)

    model = Model(inputs=input_model, outputs=output_model)
    model.compile(loss=loss, optimizer='adam')

    return model

