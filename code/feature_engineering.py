from pickle import load,dump

### General ###
import os
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

### Data Wrangling ###
import numpy as np
import pandas as pd
from scipy import stats

### Machine Learning ###
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans


train_features = pd.read_csv('../input/lish-moa/train_features.csv')
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]


# make transformations reproducible by setting random state
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)



#transforms features' distribution into gaussians
def make_gaussian(train_features, test_features):
    qt = QuantileTransformer(n_quantiles=100,random_state=42,output_distribution='normal')
    train_features[GENES+CELLS] = qt.fit_transform(train_features[GENES+CELLS])
    test_features[GENES+CELLS] = qt.transform(test_features[GENES+CELLS])
    return train_features, test_features


# performs Principal component analysis either on genes or cells
# and adds those columns to the initial dataset
def add_pca(train_features, test_features, kind, output_path):
    assert kind == "G" or kind == "C"
    if kind == "G":
        n_comp = 600
        cols = GENES 
        path =  output_path + 'gpca.pkl'
    else:
        n_comp = 50
        cols = CELLS
        path =  output_path + 'cpca.pkl'
    pca = PCA(n_components=n_comp, random_state=42)
    data = pd.concat([pd.DataFrame(train_features[cols]), pd.DataFrame(test_features[cols])])
    gpca= (pca.fit(data[cols]))
    train2= (gpca.transform(train_features[cols]))
    test2 = (gpca.transform(test_features[cols]))

    train_gpca = pd.DataFrame(train2, columns=["pca_" + kind + "_" + str(i) for i in range(n_comp)])
    test_gpca = pd.DataFrame(test2, columns=["pca_" + kind + "_" + str(i) for i in range(n_comp)])

    train_features = pd.concat((train_features, train_gpca), axis=1)
    test_features = pd.concat((test_features, test_gpca), axis=1)
    dump(gpca, open(path, 'wb'))
    return train_features, test_features


# delete columns that have high linear correlation
def variance_threshold(train_features, test_features, threshold = 0.85):
    c_n = [f for f in list(train_features.columns) if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
    mask = (train_features[c_n].var() >= threshold).values
    tmp = train_features[c_n].loc[:, mask]
    train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
    tmp = test_features[c_n].loc[:, mask]
    test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
    return train_features, test_features


# make clusters for either GENES or CELLS variables and add them to the dataset as features
def create_cluster(train, test, train_features2, test_features2, kind, output_path, SEED = 42):
    assert kind == "C" or kind == "G"
    if kind == 'G':
        n_clusters = 22
        features = GENES
        path = output_path  + 'kmeans_genes.pkl'
    elif kind == 'C':
        n_clusters = 4
        features = CELLS
        path = output_path + 'kmeans_cells.pkl'

    train_ = train_features2[features].copy()
    test_ = test_features2[features].copy()
    data = pd.concat([train_, test_], axis = 0)
    kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
    dump(kmeans, open(path, 'wb'))
    train[f'clusters_{kind}'] = kmeans.predict(train_.values)
    test[f'clusters_{kind}'] = kmeans.predict(test_.values)
    train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
    test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
    return train, test


# make clusters for either variables created after PCA and add them to the dataset as features
def create_cluster_pca(train, test, train_pca, test_pca, n_clusters=5, output_path = "", SEED = 42):
    data = pd.concat([train_pca,test_pca],axis=0)
    kmeans_pca = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
    dump(kmeans_pca, open(output_path + 'kmeans_pca.pkl', 'wb'))
    train[f'clusters_pca'] = kmeans_pca.predict(train_pca.values)
    test[f'clusters_pca'] = kmeans_pca.predict(test_pca.values)
    train = pd.get_dummies(train, columns = [f'clusters_pca'])
    test = pd.get_dummies(test, columns = [f'clusters_pca'])
    return train, test


# add some statistical values calculated from gene and cells variables to the dataset,
# add cross products between some cells' features
# add square of cell features
# add square for some gene features
def add_statistics_and_square(train, test):
    gsquarecols=['g-574','g-211','g-216','g-0','g-255','g-577','g-153','g-389','g-60','g-370','g-248','g-167','g-203','g-177','g-301','g-332','g-517','g-6','g-744','g-224','g-162','g-3','g-736','g-486','g-283','g-22','g-359','g-361','g-440','g-335','g-106','g-307','g-745','g-146','g-416','g-298','g-666','g-91','g-17','g-549','g-145','g-157','g-768','g-568','g-396']

    for df in train, test:
        df['g_sum'] = df[GENES].sum(axis = 1)
        df['g_mean'] = df[GENES].mean(axis = 1)
        df['g_std'] = df[GENES].std(axis = 1)
        df['g_kurt'] = df[GENES].kurtosis(axis = 1)
        df['g_skew'] = df[GENES].skew(axis = 1)
        df['c_sum'] = df[CELLS].sum(axis = 1)
        df['c_mean'] = df[CELLS].mean(axis = 1)
        df['c_std'] = df[CELLS].std(axis = 1)
        df['c_kurt'] = df[CELLS].kurtosis(axis = 1)
        df['c_skew'] = df[CELLS].skew(axis = 1)
        df['gc_sum'] = df[GENES + CELLS].sum(axis = 1)
        df['gc_mean'] = df[GENES + CELLS].mean(axis = 1)
        df['gc_std'] = df[GENES + CELLS].std(axis = 1)
        df['gc_kurt'] = df[GENES + CELLS].kurtosis(axis = 1)
        df['gc_skew'] = df[GENES + CELLS].skew(axis = 1)
        
        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-23'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']
        
        
        for feature in CELLS:
             df[f'{feature}_squared'] = df[feature] ** 2     
                
        for feature in gsquarecols:
            df[f'{feature}_squared'] = df[feature] ** 2        
        
    return train, test



def preprocess(train, test, output_path):
    set_seed(42)
    train_features2=train.copy()
    test_features2=test.copy()

    print("making gaussian distributions")
    train, test = make_gaussian(train, test)
    print("performing pca on genes")
    train, test = add_pca(train, test, kind = "G", output_path = output_path)
    print("performing pca on cells")
    train, test = add_pca(train, test, kind = "C",  output_path = output_path)

    pca_columns = [col for col in train.columns if col.startswith('pca')]
    train_pca = train[pca_columns]
    test_pca = test[pca_columns]

    print("variance threshold:", 0.85)

    train, test = variance_threshold(train, test, threshold = 0.85)

    print("adding clusters generated from KMeans as features")
    train, test = create_cluster(train, test, train_features2, test_features2, kind = "G", output_path=output_path)
    train, test = create_cluster(train, test, train_features2, test_features2, kind = "C", output_path=output_path)

    train, test = create_cluster_pca(train, test, train_pca, test_pca, n_clusters=5, output_path=output_path)

    print("adding statistics and square of columns as new features")
    init_col = train_features2.shape[1]
    stats_train, stats_test = add_statistics_and_square(train_features2, test_features2)

    #print(stats_train.columns[init_col])
    stats_train = stats_train.iloc[:, init_col:]
    stats_test = stats_test.iloc[:,init_col:]
    train = pd.concat((train, stats_train), axis = 1)
    test = pd.concat((test, stats_test), axis = 1)

    print("new number of columns:", train.shape[1])
    return train, test


if __name__ == "__main__":
    train_features = pd.read_csv('../input/lish-moa/train_features.csv')

    test_features = pd.read_csv('../input/lish-moa/test_features.csv')
    preprocess(train_features, test_features, output_path = "../")

