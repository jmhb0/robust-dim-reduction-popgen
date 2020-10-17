import util, analysis
import numpy as np
import pandas as pd
import pickle
import os
import sys 
from statsmodels.stats.correlation_tools import cov_nearest
from sklearn.decomposition import TruncatedSVD

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    # data
    print("Reading label data")
    sample_lookup = pd.read_csv('{}/data/labels.csv'.format(dir_path), sep=',').set_index('indID')
    labels = sample_lookup.copy()
    print('Reading POPRES data')
    with open('{}/data/POPRES_non-reduced_phased_20.dat'.format(dir_path), 'rb') as pf:
        data = pickle.load(pf)
    print('Done reading POPRES')
    # construct dataframe 
    df = pd.DataFrame(data=data, index=sample_lookup.index)

    # read the data post change-of-basis 
    M_cb = np.loadtxt('{}/data/POPRES_data_rotated.p'.format(dir_path))
    df_cb = pd.DataFrame(M_cb, index=df.index)

    # Create a benchmark dataset - only keep countries with n samples
    n_country_sample_size_gt=4
    df = util.filter_countries_lt_n_samples(df, n_country_sample_size_gt, sample_lookup)
    
    remove_outlier = True 
    df_pre_outlier = df.copy()
    '''
    if remove_outlier:
        print('Data shape before outlier detection ', df.shape)
        d = util.outlier_detection(df, sample_lookup, sigma=6, n_iter=5)
        df_post_outlier=df.copy()
        print('Data shape after outlier detection ', df_post_outlier.shape)
   
    ### run different PCA versions
    # pca on all data points (having had outliers removed
    util.do_pca_and_save(df_post_outlier, 'pca-countries_gt_{}_n_samples-removed_outliers-no_other_filtering'.format(n_country_sample_size_gt))

    # pca on a random sample of 10 per country (this avoids overweighting countries)
    n_samples_per_group=20
    analysis.run_pca_n_p_group(df, labels, 
           fname_prefix="pca-countries_gt_{}_n_samples-removed_outliers-filter_{}_samples_p_cntry".format(n_country_sample_size_gt,  n_samples_per_group)
           , n_samples_per_group=n_samples_per_group)

    # normalized pca on all 
    fname_df_dist = '{}/data/df_M_dist.p'.format(dir_path)
    df_dist = pd.read_pickle(fname_df_dist)
    # analysis.run_normalized_pca(df_pre_outlier, df_dist, labels, fname_prefix='norm-pca-countries_gt_{}_n_samples-pca-no_removed_outliers-no_other_filters'.format(n_country_sample_size_gt))

    # run supervised PCA
    t=0
    analysis.run_normalized_pca(df_pre_outlier, df_dist, labels, fname_prefix='supervised_pca_t_{}-countries_gt_{}_n_samples-pca-no_removed_outliers-no_other_filters'.format(t, n_country_sample_size_gt), supervised=True, supervised_t=t)

    # Run normalized PCA on the rotated dataset df_cb, M_cb
    # suffix `_cb` means `change of basis`
    # print('Building new distance matrix')
    # df_dist = util.build_distance_matrix(df_cb, labels)
    df_dist = pd.read_pickle('{}/data/df_M_dist.p'.format(dir_path))
    print('Running normalized PCA')
    analysis.run_normalized_pca(df_pre_outlier, df_dist, labels, fname_prefix='cb_old_dist_matrix-norm-pca-countries_gt_{}_n_samples-pca-no_removed_outliers-no_other_filters'.format(n_country_sample_size_gt))
    '''

    # Do the weighted PCA stuff


    '''
    run_n_sample_pca=True
    if run_n_sample_pca:
        # Analysis 1 - PCA on countries with 20 samples
        df = df_post_outlier.copy()
        analysis.run_pca_n_p_group(df, sample_lookup, n)

    run_ica_averaged=False
    if run_ica_averaged:
        run_ica_averaged(df, labels)

    run_ica_not_averaged=False
    if run_ica_not_averaged:
        run_ica_not_averaged(df, labels)

    run_normalized_pca=True
    if run_normalized_pca:
        analysis.run_normalized_pca(df_pre_outlier, labels)
    '''
