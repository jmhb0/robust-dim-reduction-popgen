import util, analysis
import numpy as np
import pandas as pd
import pickle
import os
import sys 
from statsmodels.stats.correlation_tools import cov_nearest

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    # data
    print("Reading label data")
    labels = pd.read_csv('{}/data/labels.csv'.format(dir_path), sep=',').set_index('indID')
    sample_lookup = labels.copy() 
    print('Reading POPRES data')
    with open('{}/data/POPRES_non-reduced_phased_20.dat'.format(dir_path), 'rb') as pf:
        data = pickle.load(pf)
    print('Done reading POPRES')

    # construct dataframe and remove outliesrs
    d = pd.DataFrame(data=data, index=sample_lookup.index)
    
    remove_outlier = False 
    df_pre_outlier = d.copy()

    if remove_outlier:
        d = util.outlier_detection(d, sample_lookup, sigma=6, n_iter=5)
        df_post_outlier=d.copy()
    
    run_n_sample_pca=False
    if run_n_sample_pca:
        print('Data shape after outlier detection ', df_post_outlier.shape)
        # Analysis 1 - PCA on countries with 20 samples
        df = df_post_outlier.copy()
        n=20
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
