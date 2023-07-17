# %% [markdown]
# # Projeto

# %%
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane.optimize import AdamOptimizer
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.embeddings import AngleEmbedding
from tqdm.notebook import tqdm, trange
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import pickle

# %%

# Read the HDF5 file using pandas
data_frame_fcnc = pd.read_hdf('fcnc_pythia_sanitised_features.h5')

# Get the number of rows
num_rows = data_frame_fcnc.shape[0]

print('Number of rows: {}'.format(num_rows))

# Explore the data
data_frame_fcnc.head()


# %%
# Read the HDF5 file using pandas
data_frame_bkg = pd.read_hdf('bkg_pythia_sanitised_features.h5')

# Get the number of rows
num_rows = data_frame_bkg.shape[0]

print('Number of rows: {}'.format(num_rows))

# Explore the data
data_frame_bkg.head()


# %% [markdown]
# ## PCA:SVMS

# %%


def perform_PCA (DataFeatures, pca_n_features, train, data):
    ## Fit PCA to train data & rank components by AUC
    pca = PCA(n_components=len(DataFeatures))
    pca.fit(train[DataFeatures])

    ## Transform the desired dataset to get its principal components
    # Get ranked components by AUC from the train data
    principalComponents = pca.transform(train[DataFeatures])

    # Book will be a dictiorary with the AUC (values) of each component (keys)
    book = {}

    # Get values for AUC computation
    y_true = train['gen_label'].values
    weights = train["gen_xsec"].values

    # Renormalise weights
    weights[y_true == 1] = (weights[y_true == 1] / weights[y_true == 1].sum()) * weights.shape[0] / 2
    weights[y_true == 0] = (weights[y_true == 0] / weights[y_true == 0].sum()) * weights.shape[0] / 2

    for feature_idx in range(principalComponents.shape[1]):
        book[f"Component {feature_idx}"] = roc_auc_score(y_true=y_true, y_score=principalComponents[:, feature_idx], sample_weight=weights)

    # Give me the best features
    book = pd.DataFrame.from_dict(book, orient="index")
    book.columns = ["AUC"]
    book.sort_values(by="AUC", ascending=False, inplace=True)
    book.reset_index(inplace=True)
    book.rename(columns={"index": "Feature"}, inplace=True)

    ## Replace current data by its components ##
    # Get components for the current set we want
    principalComponents = pca.transform(data[DataFeatures])

    # Create a new dataframe with PCA data
    newdf = pd.DataFrame(principalComponents, columns=[f"Component {i}" for i in range(principalComponents.shape[1])])

    # Select the best components given their AUC performance in training data
    newdf = newdf[book["Feature"][0 : pca_n_features]]

    # Add the other relevant features
    newdf["gen_xsec"] = data["gen_xsec"].values
    newdf["gen_label"] = data["gen_label"].values
    newdf["gen_split"] = data["gen_split"].values

    # Finally, replace self.data with newdf
    data = newdf

    # Update DataFeatures
    DataFeatures = pd.Index(list(set(data.columns) - set(["gen_label", "gen_xsec", "gen_split"])))
    
    return data, DataFeatures, book


# %%
def train_val_test(n_datapoints,n_features,method,type):
    data_frame_fcnc_pca = data_frame_fcnc.copy()
    data_frame_bkg_pca = data_frame_bkg.copy()

    # Drop the categorical features except label, weights and gen_split
    data_frame_fcnc_pca.drop(['gen_decay_filter', 'gen_filter', 'gen_n_btags', 'gen_sample', 'gen_sample_filter','gen_decay2','gen_decay1'], axis=1, inplace=True)
    data_frame_bkg_pca.drop(['gen_decay_filter', 'gen_filter', 'gen_n_btags', 'gen_sample', 'gen_sample_filter','gen_decay2','gen_decay1'], axis=1, inplace=True)

    # Drop the features that are not in both dataframes
    for feature in data_frame_fcnc_pca.columns.values:
        if feature not in data_frame_bkg_pca.columns.values:
            data_frame_fcnc_pca.drop([feature], axis=1, inplace=True)

    for feature in data_frame_bkg_pca.columns.values:
        if feature not in data_frame_fcnc_pca.columns.values:
            data_frame_bkg_pca.drop([feature], axis=1, inplace=True)
        
    # Join the dataframes
    data = pd.concat([data_frame_fcnc_pca, data_frame_bkg_pca])

    # Substitute the labels "signal" and "bkg" by 1 and 0
    data = data.replace(['signal'], 1)
    data= data.replace(['bkg'], 0)

    # train set
    train = data.loc[data['gen_split'] == 'train']
    
    # which data will be used for fitting the PCA.
    # Everything except the weights, name and label
    DataFeatures = pd.Index(list(set(data.columns) - set(["gen_label", "gen_xsec", "gen_split"])))

    
    if method == "pca":
        # perform PCA on the train data
        data, DataFeatures, book = perform_PCA (DataFeatures,n_features, train, data)
        
    elif method == "sbs":
        Features = ['MissingET_MET', 'FatJet1_Tau1','FatJet1_Tau3','Jet2_PT','Jet1_PT']
        DataFeatures = Features[:n_features]
        
    if type == "ml":
        #normalize the data except the categorical features and the weights
        data [DataFeatures] = (data [DataFeatures] - data [DataFeatures].mean()) / data [DataFeatures].std()
        
    elif type == "qml":
        #normalize the data except the categorical features and the weights
        data [DataFeatures] = (((data[DataFeatures] - data[DataFeatures].min()) / (data[DataFeatures].max() - data[DataFeatures].min())) * 2 - 1) * (np.pi)
  
    # divide the new data into train, test and validation sets
    train = data.loc[data['gen_split'] == 'train']
    test = data.loc[data['gen_split'] == 'test']
    val = data.loc[data['gen_split'] == 'val']

    # divide the train data into signal and background and get 500 samples of each
    train_sgn = train.loc[train['gen_label'] == 1].sample(n=int(n_datapoints/2), random_state=42)
    train_bkg = train.loc[train['gen_label'] == 0].sample(n=int(n_datapoints/2),random_state=42)
    x_train = pd.concat([train_sgn, train_bkg])
    x_train = x_train.sample(frac=1, random_state=42)

    # divide the validation data into signal and background and get 500 samples of each
    val_sgn = val.loc[val['gen_label'] == 1].sample(n=int(n_datapoints/2), random_state=42)
    val_bkg = val.loc[val['gen_label'] == 0].sample(n=int (n_datapoints/2), random_state=42)
    x_val = pd.concat([val_sgn, val_bkg])
    x_val = x_val.sample(frac=1, random_state=42)

    # divide the test data into signal and background and get 500 samples of each
    test_sgn = test.loc[test['gen_label'] == 1].sample(n=int(n_datapoints/2), random_state=42)
    test_bkg = test.loc[test['gen_label'] == 0].sample(n=int(n_datapoints/2), random_state=42)
    x_test = pd.concat([test_sgn, test_bkg])
    x_test = x_test.sample(frac=1, random_state=42)

    # get an array with the labels for each set
    y_train = x_train['gen_label'].values
    y_val = x_val['gen_label'].values
    y_test = x_test['gen_label'].values

    # get an array with the weights for each set
    w_train = x_train["gen_xsec"].values
    w_val = x_val["gen_xsec"].values
    w_test = x_test["gen_xsec"].values

    # get an array with the features for each set
    x_train = x_train[DataFeatures].values
    x_val = x_val[DataFeatures].values
    x_test = x_test[DataFeatures].values
    
    return x_train, y_train, w_train, x_val, y_val, w_val, x_test, y_test, w_test

# %% [markdown]
# #### SVMs : Grid search to find the best hyperparameters

# %%
# get the train, test and validation sets
x_train, y_train, w_train, x_val, y_val, w_val, x_test, y_test, w_test = train_val_test(1000,3,'pca','ml')

# %%
param_grid = {
    'C': [ 0.1, 1, 10, 100],
    'gamma': [0.001,0.1,1,'scale'],
    'kernel': ['poly', 'rbf'],
    'degree': [2]
}

# Create an SVM classifier object
svm_clf = svm.SVC(probability=True)

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_clf, param_grid, cv=5)

# Fit the grid search to your training data
grid_search.fit(x_train, y_train, sample_weight=w_train)

# Print the best hyperparameters and the corresponding score on the validation set
print("Best Hyperparameters: ", grid_search.best_params_)
best_params = grid_search.best_params_
print("Best Score: ", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test, sample_weight=w_test)
print("Test Score: ", test_score)

# %%

with open('a.pickle', 'wb') as handle:
    pickle.dump(grid_search, handle)
    
# Print the best hyperparameters and the corresponding score on the validation set
print("Best Hyperparameters: ", grid_search.best_params_)
best_params = grid_search.best_params_
print("Best Score: ", grid_search.best_score_)    
    
