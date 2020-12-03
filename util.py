import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD, FastICA
from sklearn.metrics.pairwise import euclidean_distances
from statsmodels.stats.correlation_tools import cov_nearest
import os
import sys
import logging

dir_path = os.path.dirname(os.path.realpath(__file__))
FNAME_DIST_MATRIX = '{}/data/df_M_dist.p'.format(dir_path)

''' remove any alleles that are the same across all samples
'''
def clean(df):
    d=df.values
    mask = np.std(d, axis=0, dtype=np.float32)!=0
    d = d[:,mask]
    return pd.DataFrame(d, index=df.index)

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

def build_matrix(d):
    if isinstance(d, pd.DataFrame):
        d = d.values
    mean = np.mean(d, axis=0, dtype=np.float32)
    std = np.std(d, axis=0, dtype=np.float32)
    M = (d-mean)/std
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

def symmetrize(d):
    """
    Copy the lower triangle into the upper triangle, keeping diagonal the same
    Operates in place and returns d.
    """
    d *= np.tri(d.shape[0])
    d = d+d.T-np.diag(np.diag(d))
    return d

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
Important to run `clean` and `build_matrix` so that the distance 
is the same one that is actually analyzed in PCA. 
"""
def build_distance_matrix(df, save_path=None):
    M = build_matrix(clean(df))
    M_dist = euclidean_distances(M, M)
    df_M_dist = pd.DataFrame(data=M_dist
            , index=df.index
            , columns=df.index)
    df_M_dist = symmetrize(df_M_dist)
    if save_path is not None:
        logger.info("Saving distance matrix to {}".format(save_path))
        df_M_dist.to_pickle(save_path)
    return df_M_dist 

def get_supervised_t_weights(L_weight, labels, t=0):
    """
    Return a modified L_weight matrix based on data labels. 
    For L_weight[i,j] if index i,j have the same class return t,
        else return t. 
    Require 0 <= t<= 1
    Default t=0 means the 'distance' between points will be zero.
    If t=1, then the final Laplacian is unchanged (the supervising has no effect)
    We can set t anywhere in this range.
    """
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

def do_normalized_pca(df, df_dist, dist_func=lambda x: 1/x**2
        , supervised=False, supervised_t=0, labels=None):
    ''' 
    fname_dist_matrix- must be a pickled dataframe st shape is square, symmetric
        is in the format saved in `build_distance_matrix`
    df - some data frame of patient data
    '''
    # read in distance matrix and restrict to only those samples in df.index (row and column)
    # df_dist = pd.read_pickle(fname_dist_matrix)
    df.index.difference(df_dist.index).size==0 \
        and df_dist.index.difference(df.index).size==0

    L_weight = dist_func(df_dist)
    L_weight[L_weight==np.inf] = 0   # send inf's to zero

    # do supervised PCA adjustment if applicable
    if supervised:
        if labels is None:
            raise ValueError("If running supervised=True, must supply labels lookup table")
        L_weight = get_supervised_t_weights(L_weight, labels, t=supervised_t)
        L_weight=symmetrize(L_weight)

    np.fill_diagonal(L_weight.values, -L_weight.sum())
    L_weight = -L_weight
    L_weight = cov_nearest(L_weight)

    # Code to handle the case that cov_nearest gives a matrix with one very small negative
    # eigenvalue, that makes the matrix not PSD
    # Solution is to add epsilon-Identity, where epsilon is magnitude of the smallest eig
    # but only do this if the perturbation this would cause is very small, as measured by
    # the smallest diagonal. If it would cause a big perturbation, throw an error
    eigs, _ = np.linalg.eig(L_weight)
    smallest_eig = min(eigs[0], 0)
    smallest_diag = np.min(np.diag(L_weight))
    rel_perturbation = abs(smallest_eig / smallest_diag)
    if rel_perturbation > 1e-5:
        raise ValueError("L_weight is non-neglegibly far from the PSD cone")
    L_weight = L_weight + abs(smallest_eig)*10*np.identity(len(L_weight))

    L = np.linalg.cholesky(L_weight)

    # clean non-variant alleles from df and build matrix
    df = clean(df)
    M = build_matrix(df)
    A = L.T.dot(M)

    ret_pca = do_pca(A, n_components=10)
    return ret_pca
