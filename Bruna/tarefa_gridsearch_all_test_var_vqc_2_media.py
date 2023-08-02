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
from tqdm import tqdm, trange
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import pickle
import multiprocessing.pool
from qml_functions import square_loss, circuit, classifier, cost, train_step, validation_step, train_vqc, test_vqc, test_vqc_E 



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
def train_val_test(n_datapoints,n_batches,n_features,method,type):
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

    # divide the train data into signal and background and get n_datapoints/2*n_batches samples of each set
    train_sgn = train.loc[train['gen_label'] == 1].sample(n=int(n_datapoints/2 * n_batches),random_state=1)
    train_bkg = train.loc[train['gen_label'] == 0].sample(n=int(n_datapoints/2* n_batches),random_state=1)
    x_train_batches = [pd.concat([train_sgn[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)], train_bkg[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)]]) for i in range(n_batches)]
    for i in range(n_batches):
        x_train_batches[i] = x_train_batches[i].sample(frac=1, random_state=1)

    # divide the validation data into signal and background and get n_datapointss/2*n_batches samples of each set
    val_sgn = val.loc[val['gen_label'] == 1].sample(n=int(n_datapoints/2*n_batches), random_state=1)
    val_bkg = val.loc[val['gen_label'] == 0].sample(n=int (n_datapoints/2*n_batches), random_state=1)
    x_val_batches = [pd.concat([val_sgn[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)], val_bkg[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)]]) for i in range(n_batches)]
    for i in range(n_batches):
        x_val_batches[i] = x_val_batches[i].sample(frac=1, random_state=1)

    # divide the test data into signal and background and get n_datapoints/2*n_batches samples of each set
    test_sgn = test.loc[test['gen_label'] == 1].sample(n=int(n_datapoints/2*n_batches), random_state=1)
    test_bkg = test.loc[test['gen_label'] == 0].sample(n=int(n_datapoints/2*n_batches), random_state=1)
    x_test_batches = [pd.concat([test_sgn[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)], test_bkg[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)]]) for i in range(n_batches)]
    for i in range(n_batches):
        x_test_batches[i] = x_test_batches[i].sample(frac=1, random_state=1)

    # get an array with the labels for each set
    y_train_batches = [x_train_batches[i]['gen_label'].values for i in range(n_batches)]
    y_val_batches = [x_val_batches[i]['gen_label'].values for i in range(n_batches)]
    y_test_batches = [x_test_batches[i]['gen_label'].values for i in range(n_batches)]

    # get an array with the weights for each set
    w_train_batches = [x_train_batches[i]['gen_xsec'].values for i in range(n_batches)]
    w_val_batches = [x_val_batches[i]['gen_xsec'].values for i in range(n_batches)]
    w_test_batches = [x_test_batches[i]['gen_xsec'].values for i in range(n_batches)]

    # get an array with the features for each set
    x_train_batches = [x_train_batches[i][DataFeatures].values for i in range(n_batches)]
    x_val_batches = [x_val_batches[i][DataFeatures].values for i in range(n_batches)]
    x_test_batches = [x_test_batches[i][DataFeatures].values for i in range(n_batches)]
    
    # Renormalize weights (for each batch)
    for i in range(n_batches):
        w_train_batches[i][y_train_batches[i]==1] = w_train_batches[i][y_train_batches[i]==1] / w_train_batches[i][y_train_batches[i]==1].sum() * w_train_batches[i].shape[0] / 2
        w_train_batches[i][y_train_batches[i]==0] = w_train_batches[i][y_train_batches[i]==0] / w_train_batches[i][y_train_batches[i]==0].sum() * w_train_batches[i].shape[0] / 2
        w_test_batches[i][y_test_batches[i]==1] = w_test_batches[i][y_test_batches[i]==1] / w_test_batches[i][y_test_batches[i]==1].sum() * w_test_batches[i].shape[0] / 2
        w_test_batches[i][y_test_batches[i]==0] = w_test_batches[i][y_test_batches[i]==0] / w_test_batches[i][y_test_batches[i]==0].sum() * w_test_batches[i].shape[0] / 2
        w_val_batches[i][y_val_batches[i]==1] = w_val_batches[i][y_val_batches[i]==1] / w_val_batches[i][y_val_batches[i]==1].sum() * w_val_batches[i].shape[0] / 2
        w_val_batches[i][y_val_batches[i]==0] = w_val_batches[i][y_val_batches[i]==0] / w_val_batches[i][y_val_batches[i]==0].sum() * w_val_batches[i].shape[0] / 2
    
    return x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)
        
pool = NestablePool(processes=100)


# %% [markdown]
# ## VQC
# %%
# get the train, test and validation sets
x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(1000,5,2,'pca','qml')

# %%
n_layers = 3
n_features =2

learning_rates = [0.001, 0.005, 0.01, 0.03,0.05, 0.1, 0.5, 1.0]
# Initialize best_score and best_weights
best_score = None
best_weights = None
best_lr = None

a=[]
results = []

# Iterate over each learning rate
for lr in learning_rates:
        print (lr)
        # Random weight initialization
        weights = 0.01 * np.random.randn(n_layers, n_features, 3, requires_grad=True)

        # Train the model
        async_result = pool.apply_async(train_vqc,(n_features, n_layers,x_train_batches[0],y_train_batches[0],w_train_batches[0], x_val_batches[0],y_val_batches[0],w_train_batches[0],lr, weights, 500))

        a.append((async_result))

results = [x.get() for x in a]
for i, pair in enumerate(results):
    print (pair)
    score, weights = pair
    if best_score is None or score > best_score:
        best_score = score
        best_weights = weights
        best_lr = learning_rates [i]

# %%

dict={}

dict['best_score_pca_vqc'] = best_score
dict['best_weights_pca_vqc'] = best_weights
dict['best_lr_pca_vqc'] = best_lr
    
print (best_score)
print (best_weights)
print (best_lr)

with open('dict_vqc_pca_variabilidade_media_lr_par.pickle', 'wb') as handle:
    pickle.dump(dict, handle)

# %% [markdown]
# ### PCA:VQC - Grid search to find the best hyperparameters

# %%
x_train_batches, y_train_batches, w_train_batches, x_val_train_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(1000,5,3,'pca','qml')

# %%

a=[]
results = []
# %%
n_features_list = [1,2,3,4,5]
n_datapoints_list = [250,500,1000,2000,4000]
n_layers_list = [1,2,3,4,5]

auc_test_list_pca_vqc = []
auc_val_list_pca_vqc = []
auc_std_test_list_pca_vqc = []
auc_std_val_list_pca_vqc = []
yscores_list_E = []
ensemble_auc_test_list_pca = []
for n_features in n_features_list:
    for n_layers in n_layers_list:
        for n_datapoints in n_datapoints_list:
            weights = 0.01 * np.random.randn(n_layers, n_features, 3, requires_grad=True)
            x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(n_datapoints,5,n_features,'pca','qml')
            for i in range (5):             
                async_result = pool.apply_async(train_vqc,(n_features, n_layers,x_train_batches[i],y_train_batches[i],w_train_batches[i], x_val_batches[i],y_val_batches[i],w_train_batches[i],best_lr, weights, 500))
                
                best_score, best_weights = async_result.get()
                a.append((best_score, best_weights, n_features, n_layers, n_datapoints))
                
results = a               
batch_size = 5
num_batches = len(results) // batch_size

for batch_idx in range(num_batches):
    batch = results[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    auc_test_list = []
    y_pred_prob_val_list = []
    y_pred_prob_test_list= []
    auc_val_list = []
    best_score, best_weights, n_features, n_layers, n_datapoints = batch[0]
    x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(n_datapoints,5,n_features,'pca','qml')
    x_test = np.concatenate(x_test_batches)
    y_test = np.concatenate(y_test_batches)
    w_test= np.concatenate(w_test_batches)  
    for i, pair in enumerate(batch):
        # Your code for processing each pair in the batch goes here
        best_score, best_weights, n_features, n_layers, n_datapoints = pair                           
        auc_val_list.append(test_vqc(n_features, n_layers,x_val_batches[i],y_val_batches[i],w_val_batches[i], best_weights))
        auc_test_list.append(test_vqc(n_features, n_layers,x_test_batches[i],y_test_batches[i],w_test_batches[i], best_weights))
        yscores_list_E.append(test_vqc_E(n_features, n_layers,x_test,y_test,w_test, best_weights))
    auc_val = np.mean(auc_val_list)
    auc_test = np.mean(auc_test_list)
    auc_val_list_pca_vqc.append(auc_val)
    auc_test_list_pca_vqc.append(auc_test)
    auc_test_std = np.std(auc_test_list)
    auc_val_std = np.std(auc_val_list)
    auc_std_test_list_pca_vqc.append(auc_test_std)
    auc_std_val_list_pca_vqc.append(auc_val_std)
    print('N_components: {}'.format(n_features), 'N_datapoints: {}'.format(n_datapoints), 'N_layers: {}'.format(n_layers))
    print('AUC Val: {}'.format(auc_val))
    print ('AUC Val std: {}'.format(auc_val_std))
    print('AUC Test: {}'.format(auc_test))
    print ('AUC Test std: {}'.format(auc_test_std))
    # Ensemble
    yscores_E = np.mean(yscores_list_E, axis=0)
    ensemble_auc_test = roc_auc_score(y_test, yscores_E, sample_weight=w_test)
    ensemble_auc_test_list_pca.append(ensemble_auc_test)
    print ('Ensemble AUC Test: {}'.format(ensemble_auc_test))
    print('')

# %%

dict['auc_test_list_pca_vqc'] =auc_test_list_pca_vqc
dict['auc_val_list_pca_vqc'] =auc_val_list_pca_vqc
dict ['auc_std_test_list_pca_vqc'] =auc_std_test_list_pca_vqc
dict ['auc_std_val_list_pca_vqc'] =auc_std_val_list_pca_vqc
dict ['ensemble_auc_test_list_pca'] = ensemble_auc_test_list_pca

print (auc_test_list_pca_vqc)
print (auc_val_list_pca_vqc)
print (auc_std_test_list_pca_vqc)
print (auc_std_val_list_pca_vqc)
print (ensemble_auc_test_list_pca)

with open('dict_vqc_pca_variabilidade_media_par.pickle', 'wb') as handle:
    pickle.dump(dict, handle)




# %%
x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(1000,5,2,'pca','qml')

# ### PCA:VQC -Grid search to find the best hyperparameters


# %% [markdown]
# ### Without PCA:VQC -Grid search to find the best hyperparameters

# %%
# get the train, test and validation sets
x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(1000,5,2,'sbs','qml')

# %%
n_layers = 3
n_features =2

learning_rates = [0.001, 0.005, 0.01, 0.03,0.05, 0.1, 0.5, 1.0]
# Initialize best_score and best_weights
best_score = None
best_weights = None
best_lr = None

a=[]
results = []

# Iterate over each learning rate
for lr in learning_rates:
        print (lr)
        # Random weight initialization
        weights = 0.01 * np.random.randn(n_layers, n_features, 3, requires_grad=True)

        # Train the model
        async_result = pool.apply_async(train_vqc,(n_features, n_layers,x_train_batches[0],y_train_batches[0],w_train_batches[0], x_val_batches[0],y_val_batches[0],w_train_batches[0],lr, weights, 500))
        #score = async_result[0].get()
        
        #weights = async_result[1].get()

        a.append((async_result))

results = [x.get() for x in a]
for i, pair in enumerate(results):
    print (pair)
    score, weights = pair
    if best_score is None or score > best_score:
        best_score = score
        best_weights = weights
        best_lr = learning_rates [i]

# %%

dict={}

dict['best_score_no_pca_vqc'] = best_score
dict['best_weights_no_pca_vqc'] = best_weights
dict['best_lr_no_pca_vqc'] = best_lr
    
print (best_score)
print (best_weights)
print (best_lr)

with open('dict_vqc_no_pca_variabilidade_media_lr.pickle', 'wb') as handle:
    pickle.dump(dict, handle)


a=[]
results = []
# %%
n_features_list = [1,2,3,4,5]
n_datapoints_list = [250,500,1000,2000,4000]
n_layers_list = [1,2,3,4,5]

auc_test_list_no_pca_vqc = []
auc_val_list_no_pca_vqc = []
auc_std_test_list_no_pca_vqc = []
auc_std_val_list_no_pca_vqc = []
yscores_list_E = []
ensemble_auc_test_list_no_pca = []
for n_features in n_features_list:
    for n_layers in n_layers_list:
        for n_datapoints in n_datapoints_list:
            weights = 0.01 * np.random.randn(n_layers, n_features, 3, requires_grad=True)
            x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(n_datapoints,5,n_features,'sbs','qml')
            for i in range (5):             
                async_result = pool.apply_async(train_vqc,(n_features, n_layers,x_train_batches[i],y_train_batches[i],w_train_batches[i], x_val_batches[i],y_val_batches[i],w_train_batches[i],best_lr, weights, 500))
                
                best_score, best_weights = async_result.get()
                a.append((best_score, best_weights, n_features, n_layers, n_datapoints))
                
results = a               
batch_size = 5
num_batches = len(results) // batch_size

for batch_idx in range(num_batches):
    batch = results[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    auc_test_list = []
    y_pred_prob_val_list = []
    y_pred_prob_test_list= []
    auc_val_list = []
    best_score, best_weights, n_features, n_layers, n_datapoints = batch[0]
    x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(n_datapoints,5,n_features,'sbs','qml')
    x_test = np.concatenate(x_test_batches)
    y_test = np.concatenate(y_test_batches)
    w_test= np.concatenate(w_test_batches)  
    for i, pair in enumerate(batch):
        # Your code for processing each pair in the batch goes here
        best_score, best_weights, n_features, n_layers, n_datapoints = pair                           
        auc_val_list.append(test_vqc(n_features, n_layers,x_val_batches[i],y_val_batches[i],w_val_batches[i], best_weights))
        auc_test_list.append(test_vqc(n_features, n_layers,x_test_batches[i],y_test_batches[i],w_test_batches[i], best_weights))
        yscores_list_E.append(test_vqc_E(n_features, n_layers,x_test,y_test,w_test, best_weights))
    auc_val = np.mean(auc_val_list)
    auc_test = np.mean(auc_test_list)
    auc_val_list_no_pca_vqc.append(auc_val)
    auc_test_list_no_pca_vqc.append(auc_test)
    auc_test_std = np.std(auc_test_list)
    auc_val_std = np.std(auc_val_list)
    auc_std_test_list_no_pca_vqc.append(auc_test_std)
    auc_std_val_list_no_pca_vqc.append(auc_val_std)
    print('N_components: {}'.format(n_features), 'N_datapoints: {}'.format(n_datapoints), 'N_layers: {}'.format(n_layers))
    print('AUC Val: {}'.format(auc_val))
    print ('AUC Val std: {}'.format(auc_val_std))
    print('AUC Test: {}'.format(auc_test))
    print ('AUC Test std: {}'.format(auc_test_std))
    yscores_E = np.mean(yscores_list_E, axis=0)
    ensemble_auc_test = roc_auc_score(y_test, yscores_E, sample_weight=w_test)
    ensemble_auc_test_list_no_pca.append(ensemble_auc_test)
    
    print ('Ensemble AUC Test: {}'.format(ensemble_auc_test))
    print('')

# %%

dict['auc_test_list_no_pca_vqc'] =auc_test_list_no_pca_vqc
dict['auc_val_list_no_pca_vqc'] =auc_val_list_no_pca_vqc
dict ['auc_std_test_list_no_pca_vqc'] =auc_std_test_list_no_pca_vqc
dict ['auc_std_val_list_no_pca_vqc'] =auc_std_val_list_no_pca_vqc
dict ['ensemble_auc_test_list_no_pca'] = ensemble_auc_test_list_no_pca

print (auc_test_list_no_pca_vqc)
print (auc_val_list_no_pca_vqc)
print (auc_std_test_list_no_pca_vqc)
print (auc_std_val_list_no_pca_vqc)
print (ensemble_auc_test_list_no_pca)

with open('dict_vqc_no_pca_variabilidade_media_par.pickle', 'wb') as handle:
    pickle.dump(dict, handle)



# %%
