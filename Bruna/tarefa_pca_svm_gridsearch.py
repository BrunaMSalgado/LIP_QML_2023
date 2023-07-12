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
# ## PCA

# %% [markdown]
# ### 2 Features

# %%
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

# train, test and validation sets
train = data.loc[data['gen_split'] == 'train']
test = data.loc[data['gen_split'] == 'test']
val = data.loc[data['gen_split'] == 'val']


# %%
print(data.columns.values)

# %%
# which data will be used for fitting the PCA.
# Everything except the weights, name and label
DataFeatures = pd.Index(list(set(data.columns) - set(["gen_label", "gen_xsec", "gen_split"])))
pca_n_features = 2

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
### perform PCA on the train data
data, DataFeatures, book = perform_PCA (DataFeatures, pca_n_features, train, data)

# %%
print (book)

# %%
print (data)

# %% [markdown]
# #### SVMs

# %%
#normalize the data except the categorical features and the weights
data [DataFeatures] = (data [DataFeatures] - data [DataFeatures].mean()) / data [DataFeatures].std()

# divide the new data into train, test and validation sets
train = data.loc[data['gen_split'] == 'train']
test = data.loc[data['gen_split'] == 'test']
val = data.loc[data['gen_split'] == 'val']

# divide the train data into signal and background and get 500 samples of each
train_sgn = train.loc[train['gen_label'] == 1].sample(n=500, random_state=42)
train_bkg = train.loc[train['gen_label'] == 0].sample(n=500,random_state=42)
x_train = pd.concat([train_sgn, train_bkg])
x_train = x_train.sample(frac=1, random_state=42)

# divide the validation data into signal and background and get 500 samples of each
val_sgn = val.loc[val['gen_label'] == 1].sample(n=500, random_state=42)
val_bkg = val.loc[val['gen_label'] == 0].sample(n=500, random_state=42)
x_val = pd.concat([val_sgn, val_bkg])
x_val = x_val.sample(frac=1, random_state=42)

# divide the test data into signal and background and get 500 samples of each
test_sgn = test.loc[test['gen_label'] == 1].sample(n=500, random_state=42)
test_bkg = test.loc[test['gen_label'] == 0].sample(n=500, random_state=42)
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


# %%
print(test_sgn)

# %% [markdown]
# ##### Train

# %%
# Renormalize weights
w_train[y_train == 1] = (w_train[y_train == 1] / w_train[y_train == 1].sum()) * w_train.shape[0] / 2
w_train[y_train == 0] = (w_train[y_train == 0] / w_train[y_train == 0].sum()) * w_train.shape[0] / 2

# Train SMV
clf = svm.SVC(kernel="rbf", probability=True)   
clf.fit(x_train, y_train,sample_weight=w_train)


# %% [markdown]
# ##### Validation

# %%
# Renormalize weights
w_val[y_val == 1] = (w_val[y_val == 1] / w_val[y_val == 1].sum()) * w_val.shape[0] / 2
w_val[y_val == 0] = (w_val[y_val == 0] / w_val[y_val == 0].sum()) * w_val.shape[0] / 2

# Predict
y_val_scores = clf.predict_proba(x_val)
y_val_scores = y_val_scores[:, 1]
y_val_pred = clf.predict(x_val)

# Compute metrics
accuracy = accuracy_score(y_val, y_val_pred,sample_weight=w_val)
auc_score = roc_auc_score(y_val, y_val_scores,sample_weight=w_val)

print("Accuracy:", accuracy)
print("ROC AUC Score:", auc_score)

# %% [markdown]
# ##### Test

# %%

# Renormalize weights
w_test[y_test == 1] = (w_test[y_test == 1] / w_test[y_test == 1].sum()) * w_test.shape[0] / 2
w_test[y_test == 0] = (w_test[y_test == 0] / w_test[y_test == 0].sum()) * w_test.shape[0] / 2

# Predict
y_test_scores = clf.predict_proba(x_test)
y_test_scores = y_test_scores[:, 1]
y_pred = clf.predict(x_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
auc_score = roc_auc_score(y_test, y_test_scores, sample_weight=w_test)
print("Accuracy:", accuracy)
print("ROC AUC Score:", auc_score)

# %%
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores, sample_weight=w_test)

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--', color='orange')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# %%
# Plot signal and background distributions for the predicted scores with normalized counts and log scale

num_bins = 30  # Number of bins

# Compute the range for the bins based on the minimum and maximum values of the data
min_value = min(min(y_test_scores[y_test == 0]), min(y_test_scores[y_test == 1]))
max_value = max(max(y_test_scores[y_test == 0]), max(y_test_scores[y_test == 1]))
bin_range = (min_value, max_value)

counts, bins, _ = plt.hist(y_test_scores[y_test == 0], bins=num_bins, range=bin_range,  weights=w_test[y_test==0], alpha=0.5, color='b', label='Background')
counts2, bins2, _ = plt.hist(y_test_scores[y_test == 1], bins=num_bins, range=bin_range,weights=w_test[y_test==0], alpha=0.5, color='r', label='Signal')

print(counts)
print(counts2)

plt.xlabel('y_score')
plt.ylabel('counts')
plt.title('y_scores_bkg distribution VS y_scores_signal distribution ')
plt.legend()
plt.show()

# %%
# Plot signal and background distributions for the predicted scores with normalized counts

counts = counts / sum(counts)
counts2 = counts2 / sum(counts2)

plt.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.5, color='b', label='Background')
plt.bar(bins2[:-1], counts2, width=np.diff(bins2), alpha=0.5, color='r', label='Signal')

plt.xlabel('y_score')
plt.ylabel('normalized counts')
plt.title('y_scores_bkg distribution VS y_scores_signal distribution ')
plt.legend()
plt.show()

# %%
# Normalize the counts
counts = counts / sum(counts)
counts2 = counts2 / sum(counts2)

plt.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.5, color='b', label='Background')
plt.bar(bins2[:-1], counts2, width=np.diff(bins2), alpha=0.5, color='r', label='Signal')

plt.xlabel('y_score')
plt.ylabel('log(normalized counts)')
plt.title('y_scores_bkg distribution VS y_scores_signal distribution ')
plt.yscale('log')
plt.legend()
plt.show()

# %% [markdown]
# #### Grid search to find the best hyperparameters

# %%
# Define the hyperparameter grid
param_grid = {
    'C': [0.0001,0.0005,0.0001,0.005, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50,70, 90, 100, 110, 130, 150, 200, 500, 700, 1000],
    'gamma': [0.0001,0.0005,0.001,0.005,0.1,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,7,10, 'scale'],
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [1,2,3,4,5,7,9,10,13,14,15]
}

# Create an SVM classifier object
svm_clf = svm.SVC(probability=True)

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_clf, param_grid, cv=5)

# Fit the grid search to your training data
grid_search.fit(x_train, y_train, sample_weight=w_train)

# Print the best hyperparameters and the corresponding score on the validation set
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test, sample_weight=w_test)
print("Test Score: ", test_score)


# Save best_model using pickle
with open('gridsearch_pca_svm.pickle', 'wb') as handle:
    pickle.dump(grid_search, handle)


# %%
# Obtain predictions from the best model
y_pred = best_model.predict(x_test)

# Compute accuracy score
accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
print("Accuracy: ", accuracy)

# Compute predicted probabilities for ROC curve
y_pred_prob = best_model.predict_proba(x_test)[:, 1]

# Compute ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_prob, sample_weight=w_test)


# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, sample_weight=w_test)

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--', color='orange')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# %%
# Plot signal and background distributions for the predicted scores with normalized counts and log scale

num_bins = 30  # Number of bins

# Compute the range for the bins based on the minimum and maximum values of the data
min_value = min(min(y_pred_prob[y_test == 0]), min(y_pred_prob[y_test == 1]))
max_value = max(max(y_pred_prob[y_test == 0]), max(y_pred_prob[y_test == 1]))
bin_range = (min_value, max_value)

counts, bins, _ = plt.hist(y_pred_prob[y_test == 0], bins=num_bins, range=bin_range,  weights=w_test[y_test==0], alpha=0.5, color='b', label='Background')
counts2, bins2, _ = plt.hist(y_pred_prob[y_test == 1], bins=num_bins, range=bin_range,weights=w_test[y_test==0], alpha=0.5, color='r', label='Signal')

print(counts)
print(counts2)

plt.xlabel('y_score')
plt.ylabel('counts')
plt.title('y_scores_bkg distribution VS y_scores_signal distribution ')
plt.legend()
plt.show()

# %%
# Plot signal and background distributions for the predicted scores with normalized counts

counts = counts / sum(counts)
counts2 = counts2 / sum(counts2)

plt.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.5, color='b', label='Background')
plt.bar(bins2[:-1], counts2, width=np.diff(bins2), alpha=0.5, color='r', label='Signal')

plt.xlabel('y_score')
plt.ylabel('normalized counts')
plt.title('y_scores_bkg distribution VS y_scores_signal distribution ')
plt.yscale('log')
plt.legend()
plt.show()

# %% [markdown]
# #### 2 Features: Without PCA

# %%
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

#normalize the data except the categorical features and the weights
DataFeatures = pd.Index(list(set(data.columns) - set(["gen_label", "gen_xsec", "gen_split"])))
data [DataFeatures] = (data [DataFeatures] - data [DataFeatures].mean()) / data [DataFeatures].std()

# train, test and validation sets
train = data.loc[data['gen_split'] == 'train']
test = data.loc[data['gen_split'] == 'test']
val = data.loc[data['gen_split'] == 'val']

# divide the train data into signal and background and get 500 samples of each
train_sgn = train.loc[train['gen_label'] == 1].sample(n=500, random_state=42)
train_bkg = train.loc[train['gen_label'] == 0].sample(n=500, random_state=42)
x_train = pd.concat([train_sgn, train_bkg])
x_train = x_train.sample(frac=1, random_state=42)

# divide the validation data into signal and background and get 500 samples of each
val_sgn = val.loc[val['gen_label'] == 1].sample(n=500, random_state=42)
val_bkg = val.loc[val['gen_label'] == 0].sample(n=500, random_state=42)
x_val = pd.concat([val_sgn, val_bkg])
x_val = x_val.sample(frac=1, random_state=42)

# divide the test data into signal and background and get 500 samples of each
test_sgn = test.loc[test['gen_label'] == 1].sample(n=500, random_state=42)
test_bkg = test.loc[test['gen_label'] == 0].sample(n=500, random_state=42)
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
x_train = x_train[['MissingET_MET', 'Jet1_BTag']].values
x_val = x_val[['MissingET_MET', 'Jet1_BTag']].values
x_test = x_test[['MissingET_MET', 'Jet1_BTag']].values

# %%
print (test_sgn)
print (x_train)

# %% [markdown]
# ##### Train

# %%
# Renormalize weights
w_train[y_train == 1] = (w_train[y_train == 1] / w_train[y_train == 1].sum()) * w_train.shape[0] / 2
w_train[y_train == 0] = (w_train[y_train == 0] / w_train[y_train == 0].sum()) * w_train.shape[0] / 2

# Train SMV
clf = svm.SVC(kernel="rbf", probability=True)   
clf.fit(x_train, y_train,sample_weight=w_train)

# %% [markdown]
# ##### Validation

# %%
# Renormalize weights
w_val[y_val == 1] = (w_val[y_val == 1] / w_val[y_val == 1].sum()) * w_val.shape[0] / 2
w_val[y_val == 0] = (w_val[y_val == 0] / w_val[y_val == 0].sum()) * w_val.shape[0] / 2

# Predict
y_val_scores = clf.predict_proba(x_val)
y_val_scores = y_val_scores[:, 1]
y_val_pred = clf.predict(x_val)

# Compute metrics
accuracy = accuracy_score(y_val, y_val_pred,sample_weight=w_val)
auc_score = roc_auc_score(y_val, y_val_scores,sample_weight=w_val)

print("Accuracy:", accuracy)
print("ROC AUC Score:", auc_score)

# %% [markdown]
# ##### Test

# %%
# Renormalize weights
w_test[y_test == 1] = (w_test[y_test == 1] / w_test[y_test == 1].sum()) * w_test.shape[0] / 2
w_test[y_test == 0] = (w_test[y_test == 0] / w_test[y_test == 0].sum()) * w_test.shape[0] / 2

# Predict
y_test_scores = clf.predict_proba(x_test)
y_test_scores = y_test_scores[:, 1]
y_pred = clf.predict(x_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
auc_score = roc_auc_score(y_test, y_test_scores, sample_weight=w_test)
print("Accuracy:", accuracy)
print("ROC AUC Score:", auc_score)

# %%
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores, sample_weight=w_test)

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--', color='orange')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# %%
# Plot signal and background distributions for the predicted scores with normalized counts and log scale

num_bins = 30  # Number of bins

# Compute the range for the bins based on the minimum and maximum values of the data
min_value = min(min(y_test_scores[y_test == 0]), min(y_test_scores[y_test == 1]))
max_value = max(max(y_test_scores[y_test == 0]), max(y_test_scores[y_test == 1]))
bin_range = (min_value, max_value)

counts, bins, _ = plt.hist(y_test_scores[y_test == 0], bins=num_bins, range=bin_range,  weights=w_test[y_test==0], alpha=0.5, color='b', label='Background')
counts2, bins2, _ = plt.hist(y_test_scores[y_test == 1], bins=num_bins, range=bin_range,weights=w_test[y_test==0], alpha=0.5, color='r', label='Signal')

print(counts)
print(counts2)

plt.xlabel('y_score')
plt.ylabel('counts')
plt.title('y_scores_bkg distribution VS y_scores_signal distribution ')
plt.legend()
plt.show()

# %% [markdown]
# #### Grid search to find the best hyperparameters

# %%
# Define the hyperparameter grid
param_grid = {
    'C': [0.0001,0.0005,0.0001,0.005, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50,70, 90, 100, 110, 130, 150, 200, 500, 700, 1000],
    'gamma': [0.0001,0.0005,0.001,0.005,0.1,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,7,10, 'scale'],
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [1,2,3,4,5,7,9,10,13,14,15]
}


# Create an SVM classifier object
svm_clf = svm.SVC(probability=True)

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_clf, param_grid, cv=5)

# Fit the grid search to your training data
grid_search.fit(x_train, y_train, sample_weight=w_train)

# Print the best hyperparameters and the corresponding score on the validation set
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test, sample_weight=w_test)
print("Test Score: ", test_score)

# Save best_model using pickle
with open('gridsearch_sem_pca_svm.pickle', 'wb') as handle:
    pickle.dump(grid_search, handle)

# %%
# Obtain predictions from the best model
y_pred = best_model.predict(x_test)

# Compute accuracy score
accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
print("Accuracy: ", accuracy)

# Compute predicted probabilities for ROC curve
y_pred_prob = best_model.predict_proba(x_test)[:, 1]

# Compute ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_prob, sample_weight=w_test)

# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, sample_weight=w_test)

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--', color='orange')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# %%
# Plot signal and background distributions for the predicted scores with normalized counts and log scale

num_bins = 30  # Number of bins

# Compute the range for the bins based on the minimum and maximum values of the data
min_value = min(min(y_pred_prob[y_test == 0]), min(y_pred_prob[y_test == 1]))
max_value = max(max(y_pred_prob[y_test == 0]), max(y_pred_prob[y_test == 1]))
bin_range = (min_value, max_value)

counts, bins, _ = plt.hist(y_pred_prob[y_test == 0], bins=num_bins, range=bin_range,  weights=w_test[y_test==0], alpha=0.5, color='b', label='Background')
counts2, bins2, _ = plt.hist(y_pred_prob[y_test == 1], bins=num_bins, range=bin_range,weights=w_test[y_test==0], alpha=0.5, color='r', label='Signal')

print(counts)
print(counts2)

plt.xlabel('y_score')
plt.ylabel('counts')
plt.title('y_scores_bkg distribution VS y_scores_signal distribution ')
plt.legend()
plt.show()

# %%
# Plot signal and background distributions for the predicted scores with normalized counts

counts = counts / sum(counts)
counts2 = counts2 / sum(counts2)

plt.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.5, color='b', label='Background')
plt.bar(bins2[:-1], counts2, width=np.diff(bins2), alpha=0.5, color='r', label='Signal')

plt.xlabel('y_score')
plt.ylabel('normalized counts')
plt.title('y_scores_bkg distribution VS y_scores_signal distribution ')
plt.yscale('log')
plt.legend()
plt.show()

