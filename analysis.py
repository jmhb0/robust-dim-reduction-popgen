import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD, FastICA
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.correlation_tools import cov_nearest
import os
import sys
import util

dir_path = os.path.dirname(os.path.realpath(__file__))
FNAME_DIST_MATRIX= '{}/data/df_M_dist.p'.format(dir_path)

def run_pca_n_p_group(df, labels, fname_prefix, n_samples_per_group=20):
        # filter out removed samples
        labels = labels.loc[labels.index.intersection(df.index)]
        gb = labels.groupby(labels['label'])[['label']].count() > n_samples_per_group
        countries_over_n_samples = gb[gb['label']==True].index.values

        # countries_over_n_samples = countries_over_n_samples[countries_over_n_samples['indID']].index
        labels = labels[labels['label'].isin(countries_over_n_samples)]
        sample_ids_n_p_country = labels.\
                reset_index()\
                .groupby('label')\
                .apply(lambda x: x.sample(n_samples_per_group))\
                ['indID']\
                .values

        print("Doing PCA on {} per country samples, saving to {}/results/".format(n_samples_per_group, dir_path))
        
        sample_countries =  labels.loc[sample_ids_n_p_country]
    
        df = df.loc[sample_ids_n_p_country]
        util.do_pca_and_save(df, fname_prefix=fname_prefix)

def run_ica_averaged(df, labels):
    # Now do ICA on the mean of the signals . Recreate the data
    df = df_post_outlier.copy()
    df = util.filter_countries_lt_n_samples(df, 5, sample_lookup)
    df = util.clean(df)
    print("ICA shape", df.shape)
    df_countries=df.join(labels)
    df_country_means = df_countries.groupby('label').mean()
    df_country_means = util.clean(df_country_means)
    M = util.build_matrix(df_country_means)
    ICA_projection = util.do_ica(M, M.shape[0])
    np.savetxt("{}/results/ICA_projection.dat".format(dir_path), ICA_projection)
    pd.DataFrame(df_country_means.index).to_csv('{}/results/ICA_countries.csv'.format(dir_path))

def run_ica_not_averaged(df, labels):
    # Do ICA not averaged
    df = df_post_outlier.copy()
    df = util.clean(df)
    M = util.build_matrix(df)
    ICA_projection = util.do_ica(M, 30)
    np.savetxt("{}/results/all_samples_ICA_projection.dat".format(dir_path), ICA_projection)
    countries = labels.loc[df.index]['label']
    pd.DataFrame(countries).to_csv('{}/results/all_samples_ICA_countries.csv'.format(dir_path))


def run_normalized_pca(df, labels, fname_prefix='norm_pca-'):
    similarity_funcs = [lambda x: 1/x
                        , lambda x: 1/x**2
                        ]
    fname_save_names = ['norm_pca-inv_pow_1-no_filter', 'norm_pca-inv_pow_2-no_filter']
    assert len(similarity_funcs) == len(fname_save_names)
    
    for i in range(len(similarity_funcs)):
        ret = util.do_normalized_pca(df, dist_func=similarity_funcs[i], fname_dist_matrix=FNAME_DIST_MATRIX,) 
        for k, v in ret.items():
            if k == 'PCs': continue
            fname = "{}/results/{}-{}-{}.dat".format(dir_path, fname_prefix, fname_save_names[i], k)
            print("Saving {}".format(fname))
            np.savetxt(fname, v)


