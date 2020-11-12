import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD, FastICA
from sklearn.metrics.pairwise import euclidean_distances
from statsmodels.stats.correlation_tools import cov_nearest
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
FNAME_DIST_MATRIX = '{}/data/df_M_dist.p'.format(dir_path)

''' remove any alleles that are the same across all samples
'''
def clean(df):
    data = df.values
    mask = data.std(axis=0)!=0
    data = data[:,mask]
    return pd.DataFrame(data=data, index=df.index)

''' df: DataFrame with index of sample labels.
'''
def filter_countries_lt_n_samples(df, n, sample_lookup):
    df_countries = df.join(sample_lookup['label'])
    df_countries_cnt = df_countries.groupby('label')['label'].count()
    countries = df_countries_cnt[df_countries_cnt > n]
    df = df_countries[
            df_countries['label']\
                    .isin(countries.index)
                    ]\
            .drop('label', axis=1)
    return df

def build_matrix(data):
    if isinstance(data, pd.DataFrame):
        data = data.values
    mean, std = data.mean(axis=0), data.std(axis=0)
    M = (data-mean)/std
    return M

'''Genetic data is wide. But by an SVD change of basis, we can make it n\times n
Doin the rotation takes a minute, so here we save the result.
'''
def do_data_rotation_and_save(df, fname='{}/data/POPRES_data_rotated.p'.format(dir_path)):
    M = build_matrix(
            clean(df)
            )
    n=len(M)-1
    svd_cb  = TruncatedSVD(n_components=n).fit(M)
    V = svd_cb.components_
    M_cb = M.dot(V.T)
    print(M_cb.shape)
    np.savetxt(fname, M_cb)

'''
Assumes the input matrix is already centred and scaled
return: vector with shape (dimensions, n_components)
'''
def do_pca(M, n_components=10, random_state=0):
    if len(M)-1 < n_components:
        n_components=len(M)-1
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    PC_projection = svd.fit(M).transform(M)
    PCs = svd.components_
    eig_vals = svd.singular_values_**2
    return {
            'PC_projection':PC_projection
            , 'PCs':PCs
            ,'eig_vals':eig_vals
            }

def do_ica(M, n_components, random_state=0):
    ica = FastICA(n_components=n_components, whiten=True)
    IC_projection = ica.fit(M).transform(M)
    return IC_projection

'''Option `include_eigenvectors` is there because these vectors can have dimension
equal to the number of features.
'''
def do_pca_and_save(df, fname_prefix, n_compoments=10, include_eigenvectors=False):
    df = clean(df)
    M = build_matrix(df)
    # do PCA
    ret = do_pca(M, n_compoments)
    # save results to results/ directory
    for k, v in ret.items():
        if not include_eigenvectors and k=='PCs':
            continue
        fname = "{}/results/{}-{}.dat".format(dir_path, fname_prefix, k)
        print("Saving {}".format(fname))
        np.savetxt(fname, v)
    fname = '{}/results/{}-labels.csv'.format(dir_path, fname_prefix)
    print("Saving {}".format(fname))
    pd.Series(df.index).to_csv(fname)

'''
Take in entire dataset.
Do PCA projection. Remove any samples that are `sigma` far from the projection
'''
def outlier_detection(df, sample_lookup, sigma=6, n_iter=5, n_pca_components=10):
    print("doing outlier detection with sigma: {}, n_iter: {}, n_pca_omponents: {}"\
            .format(sigma, n_iter, n_pca_components))
    # array to store list of indices to drop
    all_drop_indxs = np.array([], dtype=type(df.index[0]))
    for i in range(n_iter):
        # remove non-variable alleles
        df = clean(df)
        # build matrix and do
        M = build_matrix(df.values)
        ret = do_pca(M, n_pca_components)
        PC_projection = ret['PC_projection']
        mean, std = PC_projection.mean(axis=0), PC_projection.std(axis=0)
        df_stds_diff = abs(PC_projection-mean)/std
        # position in array of the sample that is outl=ying
        drop_i = np.where(df_stds_diff >= sigma)[0]
        # index in the original `df` object of that group
        drop_indxs = df.iloc[drop_i].index.values
        # record the outlying samples
        all_drop_indxs = np.hstack((all_drop_indxs, drop_indxs))
        # Remove the sample
        df = df.drop(drop_indxs)
    print("Dropped the following samples")
    for i in range(len(all_drop_indxs)):
        print("\t{} country: {}".format(all_drop_indxs[i], sample_lookup.loc[all_drop_indxs[i]]['label']))
    return df

"""
This takes a long time to run. For n samlples, it's O(n^3)
So it runs and pickles the result.
"""
def build_distance_matrix(df):
    M = build_matrix(clean(df))
    M_dist = euclidean_distances(M, M)
    df_M_dist = pd.DataFrame(data=M_dist
            , index=df.index
            , columns=df.index)
    # df_M_dist.to_pickle(save_path)
    return df_M_dist

def get_supervised_t_weights(L_weight, labels, t):
    assert 0 <= t <= 1
    # array ith one row per country    
    countries = labels.loc[L_weight.index]['label'] 
    # create a different array where each integer corresponds to a particular country
    # this is so we can use a numpy function on it
    unique_countries = countries.unique()
    indx_map = dict(zip(unique_countries, range(len(unique_countries))))
    countries_indx = countries.map(indx_map).values
    
    # do 'equals' outer product. True in element (i,j) means they're from the same country
    # this has the same dimension as L_weight 
    t_mask = np.equal.outer(countries_indx, countries_indx)
    # elements wuth the same label are multiplied by t. Else multiplied by 1.
    t_multiplier = np.where(t_mask, t, 1)
    return L_weight*t_multiplier

''' fname_dist_matrix- must be a pickled dataframe st shape is square, symmetric
        is in the format saved in `build_distance_matrix`
    df - some data frame of patient data
'''
def do_normalized_pca(df, df_dist, fname_dist_matrix=FNAME_DIST_MATRIX, dist_func=lambda x: 1/x
        , supervised=False, supervised_t=0, labels=None):
    # read in distance matrix and restrict to only those samples in df.index (row and column)
    # df_dist = pd.read_pickle(fname_dist_matrix)
    df_dist = df_dist.loc[df.index][df.index]

    L_weight = dist_func(df_dist)
    L_weight[L_weight==np.inf] = 0   # send inf's to zero

    # do supervised PCA adjustment if applicable
    if supervised:
        if labels is None:
            raise ValueError("If running supervised=True, must supply labels lookup table")
        L_weight = get_supervised_t_weights(L_weight, labels, t=supervised_t)

    np.fill_diagonal(L_weight.values, -L_weight.sum())
    L_weight = cov_nearest(-L_weight)
    L = np.linalg.cholesky(L_weight)

    # clean non-variant alleles from df and build matrix
    df = clean(df)
    M = build_matrix(df)
    A = L.dot(M)

    ret_pca = do_pca(A, n_components=10)
    return ret_pca
