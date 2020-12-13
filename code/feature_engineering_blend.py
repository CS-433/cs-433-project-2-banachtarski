### General ###
import sys
import os
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../input/rank-gauss")
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

from pickle import load,dump

### Deep Learning ###
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Tabnet 
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

### Make prettier the prints ###
from colorama import Fore
c_ = Fore.CYAN
m_ = Fore.MAGENTA
r_ = Fore.RED
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN

from sklearn.preprocessing import QuantileTransformer

from feature_engineering import make_gaussian, variance_threshold, add_statistics_and_square
# make transformations reproducible by setting random state
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]



# performs Principal component analysis either on genes or cells
# and adds those columns to the initial dataset
def add_pca(train_features, test_features, kind, input_path):
    assert kind == "G" or kind == "C"
    if kind == "G":
        n_comp = 600
        cols = GENES 
        path =  input_path + 'gpca.pkl'
    else:
        n_comp = 50
        cols = CELLS
        path =  input_path + 'cpca.pkl'
    pca = load(open(path, "rb"))

    
    train2= (pca.transform(train_features[cols]))
    print("number of columns for pca", kind, "is:", train2.shape[1])
    test2 = (pca.transform(test_features[cols]))

    train_gpca = pd.DataFrame(train2, columns=["pca_" + kind + "_" + str(i) for i in range(n_comp)])
    test_gpca = pd.DataFrame(test2, columns=["pca_" + kind + "_" + str(i) for i in range(n_comp)])

    train_features = pd.concat((train_features, train_gpca), axis=1)
    test_features = pd.concat((test_features, test_gpca), axis=1)

    return train_features, test_features



# make clusters for either GENES or CELLS variables and add them to the dataset as features
# train_features2, test_features2 are the copies of the original datasets
def create_cluster(train, test, train_features2, test_features2, kind, input_path, SEED = 42):
    assert kind == "C" or kind == "G"
    if kind == 'G':
        n_clusters = 22
        features = GENES
        path = input_path  + 'kmeans_genes.pkl'
    elif kind == 'C':
        n_clusters = 4
        features = CELLS
        path = input_path + 'kmeans_cells.pkl'

    train_ = train_features2[features].copy()
    test_ = test_features2[features].copy()
    kmeans = load(open(path, "rb"))
    train[f'clusters_{kind}'] = kmeans.predict(train_.values)
    test[f'clusters_{kind}'] = kmeans.predict(test_.values)
    train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
    test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
    return train, test


# make clusters for either variables created after PCA and add them to the dataset as features
def create_cluster_pca(train, test, train_pca, test_pca, n_clusters, input_path, SEED = 42):

    kmeans_pca = load(open( input_path + "kmeans_pca.pkl", 'rb'))
    train[f'clusters_pca'] = kmeans_pca.predict(train_pca.values)
    test[f'clusters_pca'] = kmeans_pca.predict(test_pca.values)
    train = pd.get_dummies(train, columns = [f'clusters_pca'])
    test = pd.get_dummies(test, columns = [f'clusters_pca'])
    return train, test



def inference_preprocess(train, test, input_path):
    set_seed(42)
    train_features2=train.copy()
    test_features2=test.copy()

    print("making gaussian distributions")
    train, test = make_gaussian(train, test)
    print("performing pca on genes")
    train, test = add_pca(train, test, kind = "G", input_path = input_path)
    print("performing pca on cells")
    train, test = add_pca(train, test, kind = "C",  input_path = input_path)

    pca_columns = [col for col in train.columns if col.startswith('pca')]
    train_pca = train[pca_columns]
    test_pca = test[pca_columns]



    assert train.shape[1] == test.shape[1]
    print("variance threshold:", 0.85)

    train, test = variance_threshold(train, test, threshold = 0.85)
    assert train.shape[1] == test.shape[1]

    print("adding clusters generated from KMeans as features")
    train, test = create_cluster(train, test, train_features2, test_features2, kind = "G", input_path=input_path)
    assert train.shape[1] == test.shape[1]
    train, test = create_cluster(train, test, train_features2, test_features2, kind = "C", input_path=input_path)
    assert train.shape[1] == test.shape[1]

    train, test = create_cluster_pca(train, test, train_pca, test_pca, n_clusters=5, input_path=input_path)
    assert train.shape[1] == test.shape[1]

    print("adding statistics and square of columns as new features")
    init_col = train_features2.shape[1]
    stats_train, stats_test = add_statistics_and_square(train_features2, test_features2)

    #print(stats_train.columns[init_col])
    stats_train = stats_train.iloc[:, init_col:]
    stats_test = stats_test.iloc[:,init_col:]
    train = pd.concat((train, stats_train), axis = 1)
    test = pd.concat((test, stats_test), axis = 1)
    assert train.shape[1] == test.shape[1]

    print("new number of columns:", train.shape[1])
    return train, test