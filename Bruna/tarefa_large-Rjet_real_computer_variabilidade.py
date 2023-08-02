# %% [markdown]
# ## Projeto

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
import tensorflow as tf
from tensorboard import notebook
from tensorboard import program
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider
import os
import shutil
from qiskit import IBMQ, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo
from sklearn.decomposition import PCA
import pickle


# %% [markdown]
# ###  Exploring & understanding the features

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


# %%
#explore the collums of the background data
print (data_frame_bkg.columns)

# %%
#explore the collums of the fcnc data
print (data_frame_fcnc.columns)

# %%
#explore the data types of the background data
print (data_frame_bkg.info())

# %%
#explore the data types of the fcnc data
print (data_frame_fcnc.info())

# %% [markdown]
# ### Histograms of Signal vs Background

# %% [markdown]
# #### 1. Normalize the data

# %%
# Z-score normalization

# copy the data to a new dataframe
data_frame_fcnc_norm = data_frame_fcnc.copy()
data_frame_bkg_norm = data_frame_bkg.copy()

# normalize the data except the categorical features and the weights
for feature in data_frame_fcnc.columns:
    if feature in ['gen_decay_filter', 'gen_filter', 'gen_label', 'gen_n_btags', 'gen_sample', 'gen_sample_filter', 'gen_split', 'gen_decay2','gen_decay1', 'gen_xsec']:
        pass
    else: 
        data_frame_fcnc_norm[feature] = (data_frame_fcnc[feature] - data_frame_fcnc[feature].mean()) / data_frame_fcnc[feature].std()
        
for feature in data_frame_bkg.columns:
    if feature in ['gen_decay_filter', 'gen_filter', 'gen_label', 'gen_n_btags', 'gen_sample', 'gen_sample_filter', 'gen_split','gen_decay2','gen_decay1','gen_xsec']:
        pass
    else:  
        data_frame_bkg_norm[feature] = (data_frame_bkg[feature] - data_frame_bkg[feature].mean()) / data_frame_bkg[feature].std()

# %% [markdown]
# 
# #### 2. Plot histograms Signal Vs Background

# %%
# l is a list of the categorical features that are in both dataframes
l=[]
for i in data_frame_fcnc.columns:
    if  i in ['gen_decay_filter', 'gen_filter', 'gen_label', 'gen_n_btags', 'gen_sample', 'gen_sample_filter', 'gen_split','gen_xsec']:pass
    elif i not in data_frame_bkg.columns.values: pass
    else: l.append(i)

# %%
# Plot histograms signal vs bkg for each feature

num_features = len(data_frame_fcnc.columns)
num_cols = 3
num_rows = math.ceil(len(l) / 3)

plt.clf()



# %% [markdown]
# ### Data preprocessing 

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
    train_sgn = train.loc[train['gen_label'] == 1].sample(n=int(n_datapoints/2 * n_batches),random_state=46)
    train_bkg = train.loc[train['gen_label'] == 0].sample(n=int(n_datapoints/2* n_batches),random_state=46)
    x_train_batches = [pd.concat([train_sgn[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)], train_bkg[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)]]) for i in range(n_batches)]
    for i in range(n_batches):
        x_train_batches[i] = x_train_batches[i].sample(frac=1, random_state=46)

    # divide the validation data into signal and background and get n_datapointss/2*n_batches samples of each set
    val_sgn = val.loc[val['gen_label'] == 1].sample(n=int(n_datapoints/2*n_batches), random_state=46)
    val_bkg = val.loc[val['gen_label'] == 0].sample(n=int (n_datapoints/2*n_batches), random_state=46)
    x_val_batches = [pd.concat([val_sgn[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)], val_bkg[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)]]) for i in range(n_batches)]
    for i in range(n_batches):
        x_val_batches[i] = x_val_batches[i].sample(frac=1, random_state=46)

    # divide the test data into signal and background and get n_datapoints/2*n_batches samples of each set
    test_sgn = test.loc[test['gen_label'] == 1].sample(n=int(n_datapoints/2*n_batches), random_state=46)
    test_bkg = test.loc[test['gen_label'] == 0].sample(n=int(n_datapoints/2*n_batches), random_state=46)
    x_test_batches = [pd.concat([test_sgn[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)], test_bkg[i*int(n_datapoints/2):(i+1)*int(n_datapoints/2)]]) for i in range(n_batches)]
    for i in range(n_batches):
        x_test_batches[i] = x_test_batches[i].sample(frac=1, random_state=46)

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


# %% [markdown]
# ### VQC

# %%
'''
if os.path.exists('./logs/train_val'):
    shutil.rmtree('./logs/train_val')

# Create a summary writer for logging the loss
train_val_writer = tf.summary.create_file_writer('./logs/train_val')

# Create a TensorBoard server
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', './logs'])
url = tb.launch()

# Print the TensorBoard URL
print("TensorBoard URL:", url)
'''

# %%
# accuracy function
def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


# %%
# loss function
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

# quantum circuit function
def circuit(n_features, n_layers, weights, x):
        # Embedding
        
        qml.AngleEmbedding(x,range (0, n_features),rotation="X" )

        # For every layer
        for layer in range(n_layers):
            W1 = weights[layer]

            # Define Rotations
            for i in range(0,n_features):
                qml.Rot(W1[i, 0], W1[i, 1], W1[i, 2], wires=i)

            # Entanglement
            if n_features != 1:
                if n_features > 2:
                    for i in range(n_features):
                        if i == n_features - 1:
                            qml.CNOT(wires=[i, 0])
                        else:
                            qml.CNOT(wires=[i, i + 1])
                else:
                    qml.CNOT(wires=[1, 0])

        return qml.expval(qml.PauliZ(0))

# classifier function    
def classifier(n_features, n_layers, weights, x):
        #c = circuit(n_features, n_layers, weights, x)
        dev=qml.device("default.qubit", wires=n_features)
        return qml.QNode(circuit, dev)(n_features, n_layers, weights, x)
    
# cost function    
def cost(n_features, n_layers,weights,X,Y,W):  
        # Compute predictions
        y_scores = [(classifier(n_features, n_layers,weights, x) + 1) / 2 for x in X]

        loss = square_loss(Y, y_scores)
        loss = loss * W
        loss = loss.sum()
        
        return loss
    
# train step function    
def train_step(n_features, n_layers,x_train,y_train, w_train, weights, opt,desc='Training'):
        
        # Only require grad if necessary
        x_train = np.array(x_train, requires_grad=False)
        y_train = np.array(y_train, requires_grad=True)
        w_train = np.array(w_train, requires_grad=False)

        # Compute cost and update weights
        weights, loss = opt.step_and_cost(cost, n_features, n_layers,weights, X=x_train, Y=y_train, W=w_train)

        return loss, weights
    
# validation step function
def validation_step(n_features, n_layers, x_val, y_val, w_val, weights, best_score, epoch_number, best_score_epoch,best_weights,desc='Validation'):
    X_val = np.array(x_val, requires_grad=False)
    Y_val = np.array(y_val, requires_grad=False)
    W_val = np.array(w_val, requires_grad=False)

    y_scores = np.array([classifier(n_features, n_layers, weights, x) for x in X_val])
    y_scores = (y_scores + 1) / 2

    W_val[Y_val == 1] = (W_val[Y_val == 1] / W_val[Y_val == 1].sum()) * W_val.shape[0] / 2
    W_val[Y_val == 0] = (W_val[Y_val == 0] / W_val[Y_val == 0].sum()) * W_val.shape[0] / 2

    auc_score = roc_auc_score(y_true=Y_val, y_score=y_scores, sample_weight=W_val)
    loss = cost(n_features, n_layers, weights, X_val, Y_val, W_val)


    if best_score is None or auc_score > best_score:
        best_score = auc_score
        best_score_epoch = epoch_number
        best_weights = weights

    tqdm.write(f"Epoch: {epoch_number}, Validation Loss: {loss:.4f}, AUC Score: {auc_score:.4f}")

    return best_score, best_score_epoch, best_weights
        
        
# train function
def train_vqc(n_features, n_layers, x_train, y_train,w_train,x_val,y_val,w_val, learning_rate, weights, max_epochs):
    opt = AdamOptimizer(learning_rate)
    best_score = None
    best_weights = None
    best_score_epoch = None
    epoch_number = 0

    with tqdm(total=max_epochs, desc='Epoch', unit='epoch') as pbar:
        for epoch in range(epoch_number, max_epochs):
            epoch_number = epoch

            loss, nf_nl_weights = train_step(n_features, n_layers, x_train, y_train, w_train, weights, opt, desc='Training')
            
            # Log variable values using tqdm.write
            tqdm.write(f"Epoch: {epoch_number:}, Loss: {loss:.4f}")
            
            
            weights = nf_nl_weights[2:]
            weights = weights[0]

            if epoch_number == max_epochs - 1 or (epoch_number+1)%5==0:
                best_score, best_score_epoch, best_weights = validation_step(n_features, n_layers, x_val, y_val, w_val, weights, best_score, epoch_number, best_score_epoch, best_weights,desc='Validation')
                # early stopping
                if epoch_number - best_score_epoch > 20 and epoch_number > 60:
                    tqdm.write(f"Early stopping at epoch {epoch_number}")
                    break

            pbar.update(1)  # Update progress bar
        tqdm.write(f"Best Score: {best_score:.4f}")            
        
    return best_score, best_weights


def test_vqc(n_features, n_layers,x_test,y_test,w_test, weights):
        # Remove grad
        X_test = np.array(x_test, requires_grad=False)
        Y_test = np.array(y_test, requires_grad=False)
        W_test = np.array(w_test, requires_grad=False)

        # This will be between -1 and 1, we need to convert to between 0 and 1
        y_scores = np.array([classifier(n_features, n_layers,weights, x) for x in X_test])
        y_scores = (y_scores + 1) / 2

        # Renormalize weights
        W_test[Y_test == 1] = (W_test[Y_test == 1] / W_test[Y_test == 1].sum()) * W_test.shape[0] / 2
        W_test[Y_test == 0] = (W_test[Y_test == 0] / W_test[Y_test == 0].sum()) * W_test.shape[0] / 2

        # Calculate ROC
        auc_score = roc_auc_score(y_true=Y_test, y_score=y_scores, sample_weight=W_test)
        
        return auc_score
    
   
def test_vqc_E(n_features, n_layers,x_test,y_test,w_test, weights):
        # Remove grad
        X_test = np.array(x_test, requires_grad=False)
        Y_test = np.array(y_test, requires_grad=False)
        W_test = np.array(w_test, requires_grad=False)

        # This will be between -1 and 1, we need to convert to between 0 and 1
        y_scores = np.array([classifier(n_features, n_layers,weights, x) for x in X_test])
        y_scores = (y_scores + 1) / 2

        # Renormalize weights
        W_test[Y_test == 1] = (W_test[Y_test == 1] / W_test[Y_test == 1].sum()) * W_test.shape[0] / 2
        W_test[Y_test == 0] = (W_test[Y_test == 0] / W_test[Y_test == 0].sum()) * W_test.shape[0] / 2

        # Calculate ROC
        #auc_score = roc_auc_score(y_true=Y_test, y_score=y_scores, sample_weight=W_test)
        
        return y_scores 

# %%
n_features = 2
n_layers = 3
n_datapoints = 1000
x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(n_datapoints,5,n_features,'sbs','qml')
y_pred_prob_val_list = []
y_pred_prob_test_list= []
auc_val_list = []
auc_test_list = []
fpr_test_batches=[]
tpr_test_batches=[]
yscores_list_E = []
ensemble_auc_test_list_no_pca = []
x_test = np.concatenate(x_test_batches)
y_test = np.concatenate(y_test_batches)
w_test= np.concatenate(w_test_batches)  
for i in range (5):
    # Random weight initialization
    weights = 0.01 * np.random.randn(n_layers, n_features, 3, requires_grad=True)
    # We create a quantum device with n_features "wires" (or qubits)
    dev = qml.device('default.qubit', wires=n_features)
    # train the model
    best_score, best_weights = train_vqc(n_features, n_layers,x_train_batches[i],y_train_batches[i],w_train_batches[i], x_val_batches[i],y_val_batches[i],w_train_batches[i],0.01, weights, 500)
    #test
    auc_val_list.append(test_vqc(n_features, n_layers,x_val_batches[i],y_val_batches[i],w_val_batches[i], best_weights))
    auc_test_list.append(test_vqc(n_features, n_layers,x_test_batches[i],y_test_batches[i],w_test_batches[i], best_weights))
    y_scores = test_vqc_E(n_features, n_layers,x_test_batches[i],y_test_batches[i],w_test_batches[i], best_weights)
    y_pred_prob_test_list.append(y_scores)
    yscores_list_E.append(test_vqc_E(n_features, n_layers,x_test,y_test,w_test, best_weights))
    fpr, tpr, thresholds = roc_curve(y_test_batches[i],y_scores ,sample_weight=w_test_batches[i])
    fpr_test_batches.append(fpr)
    tpr_test_batches.append(tpr)
auc_val_sim = np.mean(auc_val_list)
auc_val_sim_std = np.std(auc_val_list)
auc_sim_test = np.mean(auc_test_list)
auc_test_sim_std = np.std(auc_test_list)
yscores_E = np.mean(yscores_list_E, axis=0)
ensemble_auc_test = roc_auc_score(y_test, yscores_E, sample_weight=w_test)
fpr_E, tpr_E, _ = roc_curve(y_test, yscores_E, sample_weight=w_test)

dict = { 'auc_val_sim': auc_val_sim, 'auc_val_sim_std': auc_val_sim_std  ,'auc_test_sim': auc_sim_test,'auc_test_sim_std':auc_test_sim_std ,'fpr_test_batches_sim': fpr_test_batches, 'tpr_test_batches_sim': tpr_test_batches, 'yscores_E_sim': yscores_E, 'ensemble_auc_test_sim': ensemble_auc_test, 'fpr_E_sim': fpr_E, 'tpr_E_sim': tpr_E, 'y_pred_prob_test_list_sim': y_pred_prob_test_list,'auc_test_list_sim' : auc_test_list}

print (dict)


# %%
print('N_components: {}'.format(n_features), 'N_datapoints: {}'.format(n_datapoints))
print('AUC Val: {}'.format(auc_val_sim))
print ('AUC Val std: {}'.format(auc_val_sim_std))
print('')
print('AUC Test: {}'.format(auc_sim_test))
print ('AUC Test std: {}'.format(auc_test_sim_std))
print('')

# %% [markdown]
# #### Real Device

# %%
'''
# remove existing logs
if os.path.exists('./logs/train_val'):
    shutil.rmtree('./logs/train_val')
'''

# %%
# Save your credentials on disk.
#IBMProvider.save_account(token='89d169bf0ad41245806d22338e0458557da47dab398645d14ec77cae3d506133376707f89bf393b51dcdf0a91e1bbd3efa9e680822a5e953d0fc4315ad9a491c', overwrite=True)

provider = IBMProvider(instance='ibm-q/open/main')

# %%
from qiskit.providers.ibmq import least_busy


# Load your IBM Quantum Experience account credentials
provider = IBMProvider(instance='ibm-q/open/main')


# Get a list of available backends from the provider
backends = provider.backends()

# Print the names of all available backends
print("Available Backend Names:")
for backend in backends:
    print(backend.name)


# %%
# Get a list of available backend names from the provider
backend_names = [backend.name for backend in provider.backends()]

# Get the backend objects corresponding to the available backend names
backends = [provider.get_backend(name) for name in backend_names]

# Get the least busy backend
least_busy_backend = least_busy(backends)

# Print information about the least busy backend
print("Least Busy Backend:")
print("Name:", least_busy_backend)
print("Number of active jobs:", least_busy_backend.status().pending_jobs)

# %%
'''
# Create a summary writer for logging the loss
train_val_writer = tf.summary.create_file_writer('./logs/train_val')
'''

# %%
n_features = 2
n_layers = 3
n_datapoints = 1000
# We create a quantum device with n_features "wires" (or qubits)
dev =  ibm_dev = qml.device('qiskit.ibmq', wires=n_features, backend='ibmq_quito', provider=provider, shots=2000)
x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(n_datapoints,5,n_features,'sbs','qml')
y_pred_prob_val_list = []
y_pred_prob_test_list= []
auc_val_list = []
auc_test_list = []
fpr_test_batches=[]
tpr_test_batches=[]
yscores_list_E = []
ensemble_auc_test_list_no_pca = []
x_test = np.concatenate(x_test_batches)
y_test = np.concatenate(y_test_batches)
w_test= np.concatenate(w_test_batches) 

for i in range (5):
    # Random weight initialization
    weights = 0.01 * np.random.randn(n_layers, n_features, 3, requires_grad=True)
    # We create a quantum device with n_features "wires" (or qubits)
    dev = qml.device('default.qubit', wires=n_features)
    # train the model
    best_score, best_weights = train_vqc(n_features, n_layers,x_train_batches[i],y_train_batches[i],w_train_batches[i], x_val_batches[i],y_val_batches[i],w_train_batches[i],0.01, weights, 500)
    #test
    auc_val_list.append(test_vqc(n_features, n_layers,x_val_batches[i],y_val_batches[i],w_val_batches[i], best_weights))
    auc_test_list.append(test_vqc(n_features, n_layers,x_test_batches[i],y_test_batches[i],w_test_batches[i], best_weights))
    y_scores = test_vqc_E(n_features, n_layers,x_test_batches[i],y_test_batches[i],w_test_batches[i], best_weights)
    y_pred_prob_test_list.append(y_scores)
    yscores_list_E.append(test_vqc_E(n_features, n_layers,x_test,y_test,w_test, best_weights))
    fpr, tpr, thresholds = roc_curve(y_test_batches[i],y_scores ,sample_weight=w_test_batches[i])
    fpr_test_batches.append(fpr)
    tpr_test_batches.append(tpr)    
auc_val_real = np.mean(auc_val_list)
auc_val_real_std = np.std(auc_val_list)
auc_test_real = np.mean(auc_test_list)
auc_test_real_std = np.std(auc_test_list)
yscores_E = np.mean(yscores_list_E, axis=0)
ensemble_auc_test = roc_auc_score(y_test, yscores_E, sample_weight=w_test)
fpr_E, tpr_E, _ = roc_curve(y_test, yscores_E, sample_weight=w_test)

dict['auc_val_real'] = auc_val_real
dict['auc_val_real_std'] = auc_val_real_std
dict['auc_test_real'] = auc_test_real
dict['auc_test_real_std'] = auc_test_real_std
dict['fpr_test_batches_real'] = fpr_test_batches
dict['tpr_test_batches_real'] = tpr_test_batches
dict['yscores_E_real'] = yscores_E
dict['ensemble_auc_test_real'] = ensemble_auc_test
dict['fpr_E_real'] = fpr_E
dict['tpr_E_real'] = tpr_E
dict['y_pred_prob_test_list_real'] = y_pred_prob_test_list
dict['auc_test_list_real'] = auc_test_list


# %%
print('N_components: {}'.format(n_features), 'N_datapoints: {}'.format(n_datapoints))
print('AUC Val: {}'.format(auc_val_real))
print ('AUC Val std: {}'.format(auc_val_real_std))
print('')
print('AUC Test: {}'.format(auc_test_real))
print ('AUC Test std: {}'.format(auc_test_real_std))
print('')




# %% [markdown]
# #### Simulator with noise

# %%
provider = IBMProvider(instance='ibm-q/open/main')
backend = provider.get_backend('ibmq_quito')

# Get the noise model for ibmq_quito
noise_model = NoiseModel.from_backend(backend)


# %%
n_features = 2
n_layers = 3
n_datapoints = 1000
# We create a quantum device with n_features "wires" (or qubits)
dev =  ibm_dev = qml.device('qiskit.ibmq', wires=n_features, backend='ibmq_qasm_simulator', provider=provider, shots=2000,noise_model=noise_model)
x_train_batches, y_train_batches, w_train_batches, x_val_batches, y_val_batches, w_val_batches, x_test_batches, y_test_batches, w_test_batches = train_val_test(n_datapoints,5,n_features,'sbs','qml')
y_pred_prob_val_list = []
y_pred_prob_test_list= []
auc_val_list = []
auc_test_list = []
fpr_test_batches=[]
tpr_test_batches=[]
yscores_list_E = []
ensemble_auc_test_list_no_pca = []
x_test = np.concatenate(x_test_batches)
y_test = np.concatenate(y_test_batches)
w_test= np.concatenate(w_test_batches) 
for i in range (5):
    # Random weight initialization
    weights = 0.01 * np.random.randn(n_layers, n_features, 3, requires_grad=True)
    # We create a quantum device with n_features "wires" (or qubits)
    dev = qml.device('default.qubit', wires=n_features)
    # train the model
    best_score, best_weights = train_vqc(n_features, n_layers,x_train_batches[i],y_train_batches[i],w_train_batches[i], x_val_batches[i],y_val_batches[i],w_train_batches[i],0.01, weights, 500)
    #test
    auc_val_list.append(test_vqc(n_features, n_layers,x_val_batches[i],y_val_batches[i],w_val_batches[i], best_weights))
    auc_test_list.append(test_vqc(n_features, n_layers,x_test_batches[i],y_test_batches[i],w_test_batches[i], best_weights))
    y_scores = test_vqc_E(n_features, n_layers,x_test_batches[i],y_test_batches[i],w_test_batches[i], best_weights)
    y_pred_prob_test_list.append(y_scores)
    yscores_list_E.append(test_vqc_E(n_features, n_layers,x_test,y_test,w_test, best_weights))
    fpr, tpr, thresholds = roc_curve(y_test_batches[i],y_scores ,sample_weight=w_test_batches[i])
    fpr_test_batches.append(fpr)
    tpr_test_batches.append(tpr)
auc_val_sim_noise = np.mean(auc_val_list)
auc_val_sim_noise_std = np.std(auc_val_list)
auc_test_sim_noise = np.mean(auc_test_list)
auc_test_sim_noise_std = np.std(auc_test_list)
yscores_E = np.mean(yscores_list_E, axis=0)
ensemble_auc_test = roc_auc_score(y_test, yscores_E, sample_weight=w_test)
fpr_E, tpr_E, _ = roc_curve(y_test, yscores_E, sample_weight=w_test)


dict['auc_val_sim_noise'] = auc_val_sim_noise
dict['auc_val_sim_noise_std'] = auc_val_sim_noise_std
dict['auc_test_sim_noise'] = auc_test_sim_noise
dict['auc_test_sim_noise_std'] = auc_test_sim_noise_std
dict['fpr_test_batches_sim_noise'] = fpr_test_batches
dict['tpr_test_batches_sim_noise'] = tpr_test_batches
dict['yscores_E_sim_noise'] = yscores_E
dict['ensemble_auc_test_sim_noise'] = ensemble_auc_test
dict['fpr_E_sim_noise'] = fpr_E
dict['tpr_E_sim_noise'] = tpr_E
dict['y_pred_prob_test_list_sim_noise'] = y_pred_prob_test_list
dict['auc_test_list_sim_noise'] = auc_test_list


print (dict)

with open ('dict_tarefa_variabilidade.pickle', 'wb') as f:
    pickle.dump(dict, f)

