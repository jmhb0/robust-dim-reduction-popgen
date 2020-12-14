Implement methods from paper
> Koren, Y. and Carmel, L., 2004. Robust linear dimensionality reduction. IEEE transactions on visualization and computer graphics, 10(4), pp.459-470.
Intended for population genetic studies. 

Created of population genetics applications, for example [POPRES](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000145.v4.p2).

Also see [memory-usage.ipynb](memory-usage.ipynb) on memory usage for large genomic datasets

# Data and objects 
Take a array, `data` with with shape `(m,n)`, that is `m` samples and `n` alleles (features) that are 0 or 1. 
Take a list of sample names, `indxs` (can default to just range), and create a DataFrame:
```
import pandas as pd
df = pd.DataFrame(data, index=indxs)
```

In `util.py` the method `util.clean(df)` takes the DataFrame, and returns the same frame but removing columns that have no variation over all the samples. This reduces the data size, and won't change results. 
The method `util.build_matrix(df)` takes the DataFrame, does centering (so the matrix has mean zero), and normalizing (so the columns have std dev of 1).

## Distance matrix
Required for normalized and supervized PCA:
```
import util
df_dist = util.build_distance_matrix(df)
```
For `df` with shape `(m,n)`, the frame `df_dist` has square shape `(m,m)`. This is a DataFrame with the same index and column labels as the `df.index`.

## Return dictionary
All dimension reduction methods return a dictionary, `res`. They are paramaterized by `n_components`, which is the number of embedded dimensions computed.
- `PC_projection`: array shape `(m,n_components)`. The low-dim embedding of the `m` samples.
- `PCs`: array shape `(n_components,n)`. The vectors onto which we have linear projections. 
- `eig_vals`: array shape `(n_components,)`. The eigenvalues of the PCA eigenproblem, which is also the portion of explained variance. 

# Normalized PCA 
Do:
```
res_norm = util.do_normalized_pca(df, df_dist, dist_func=lambda x: 1/x**2)
```
This automatically centers and normalizes the data matrix (by calling `util.build_matrix`) so there's no need for the user to do this. 

The method `dist_func`, takes a float (the pairwise distance between two points, read from `df_dist`). It returns what is called d_{ij} in the paper - a dissimilarity measure. Normalized PCA tries to keep points with big d_{ij} together.

Note: The method does an estimation of a certain covariance matrix, but numerical error may cause it to be no longer PSD. This is corrected for, but if this matrix is too far from the PSD cone an exception is raised. 


# Supervized PCA 
Call the same method but with different arguments.
```
res_sup = util.do_normalized_pca(df, df_dist, dist_func=lambda x: 1/x**2, supervized=True, supervised_t=0)
```
`supervised_t` must be between 0 and 1. If using 1, this reduces to just regular normalized PCA. 

# Classic PCA 
Warning: this does not do centering or normalizing. (It doesn't make sense to do PCA without centering, but it may make sense to not scale). 
```
res_pca = util.do_pca(df)
```
