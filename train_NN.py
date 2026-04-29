import pickle
import copy
import sys
import math
import torch
import scipy.ndimage
import scipy.interpolate

import pyvista           as pv
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import scipy             as sp 

from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from functions               import *
from READ_WRITE_FEATURES     import *
from NeuralNet               import *
from joblib                  import Parallel, delayed
from sklearn.metrics         import r2_score

def single_neural_training(x_train, x_test, y_train, y_test, seed):

    np.random.seed(seed)
    leaning_rate = 10.0**np.random.uniform(-6,-2)
    batch_size = int(2**np.random.randint(5,10))
    n_neurons = np.random.randint(50,300)
    n_layers = np.random.randint(2,10)
    lambda_l1 = 10.0**np.random.uniform(-8,-2)

    layers = [n_neurons]*n_layers

    hyperparams = {'lr': leaning_rate, 'batch_size': batch_size, 'hidden_layers': layers, 'lambda_l1': lambda_l1, 'seed':seed}

    y_pred_train, y_pred_test, model = train_neural_network(x_train, x_test, y_train, y_test, hyperparams)

    r2train = r2_score(y_train, y_pred_train)
    r2test  = r2_score(y_test,  y_pred_test)

    print(f"R2 train and test for seed {seed}: {r2train:.4f}, {r2test:.4f}")


    torch.save(model, f"model_{seed}.pth")

    with open('gridsearch.csv', 'a+') as f:
        f.write(f"{seed},{hyperparams['lr']},{hyperparams['batch_size']},{n_layers},{n_neurons},{hyperparams['lambda_l1']},{r2train},{r2test}\n")

    return


sigma_train = 0. # Standard deviation for Gaussian kernel
sigma_pred = 0.  # Standard deviation for Gaussian kernel
#
test_size=0.05
train_size = 1.- test_size
#
ML_model_choice='RFR'
latest_time_foldername ="5000"
PathToCases="../"
WhichWeights = "GaussSearchGrid"
FeaturesChoice = "Ling"
SBL_models = ['ANSJ', 'CHAN', 'SEP']
crit_RFR = ['squared_error', 'absolute_error', "friedman_mse", "poisson"]

dirModels = f"../ML_trained_models"

'''
case_type: used to set the weights (in the sense of influence) of the training points on RF training.
           The weights are set in order to give the same influence to each model on the training, even if the number of training points is different from one case to another.

selected_training_cases: used lo load the model weights, features, iterate to perform train-valid split,
                         compute the influence, and visualize the RFR model prediction

keys_weights_export: used to export the weights to the foam files, and to visualize the predicted weights vs exact weights.                 
'''

case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D", "LRN_OGV_trans"], 'SEP':[  "PH10595" , "CBFS13700", "CD12600", "LRN_OGV_trans_OP1"]} #
selected_training_cases = ["Jet_NearSonic_projected", "channel1000_2D", "PH10595", "CBFS13700", "CD12600", "LRN_OGV_trans", "LRN_OGV_trans_OP1"]
keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700", 'LRN_OGV_trans', "LRN_OGV_trans_OP1"]
cases = ["Jet_NearSonic", "CD12600", "channel1000_2D", "PH10595", "CBFS13700", 'LRN_OGV_trans', "LRN_OGV_trans_OP1"]#
models = ["Frozen", "ANSJ", "CHAN", "SEP"]# 

# case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D"], 'SEP':["PH10595" , "CBFS13700", "CD12600"]} #
# selected_training_cases = ["Jet_NearSonic_projected", "channel1000_2D", "PH10595", "CBFS13700", "CD12600"]
# keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]
# cases = ["Jet_NearSonic", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]#
# models = ["Frozen", "ANSJ", "CHAN", "SEP"]#


print ("post-process jet data, this will change the dic and add new Jet_proj")		
dic_data = create_dic_data(FeaturesChoice, cases, models)

make_symm_Jet_data_on_PIVsubdomain(FeaturesChoice, dic_data, models, whichJet="Jet_NearSonic")
export_Jet_foam_files(FeaturesChoice, models, dic_data, 'Jet_NearSonic', 'projected', 5000)
restrict_Jet_data_to_upper_half_PIV_domain_bounds(FeaturesChoice, dic_data, models, whichJet="Jet_NearSonic")
export_Jet_foam_files(FeaturesChoice, models, dic_data, 'Jet_NearSonic', 'restricted', 5000)
augment_Jet_data_to_upper_half_PIV_domain_bounds(FeaturesChoice, dic_data, models, whichJet="Jet_NearSonic")


#### Compute the weights an features for the RFR training dataset, and export them to foam files
if WhichWeights == "GaussSearchGrid":
    print ('\n--------------- Calculate Gaussian weights Grid search')
    weightsU_org, features = mixture_of_expert_Grid_search(dic_data, sigma=1e-3)

weightsU = {key: weightsU_org[key] for key in selected_training_cases if key in weightsU_org}
features = {key: features[key]['ANSJ'] for key in selected_training_cases if key in features}

C_coords = {key: dic_data[key]['CHAN']['internalMesh'].cell_centers().points for key in selected_training_cases}
domain_bounds = {key: dic_data[key]['CHAN']['internalMesh'].bounds for key in selected_training_cases}

print ('\n--------------- Export exact smoothened weights')
for case in keys_weights_export: 
    if case == "Jet_NearSonic" : case_export = "Jet_NearSonic_restricted"
    else : case_export = case
    time_folder = "5000"
    export_folder = f"./{FeaturesChoice}/{case}/ExactWeights/{WhichWeights}"
    simul_folder = os.path.join(export_folder,time_folder)
    boundary_data, nCells = read_boundary_data(f"{export_folder}/constant/polyMesh")
    
    for i_ in range(3):
        write_scalar_field(simul_folder, time_folder, f"wU{i_+1}_exact", weightsU_org[case_export][:,i_], boundary_data)


#### Perform the train-valid split
train_indices_case, test_indices_case = {}, {}	
x_input_U_train_case, x_input_U_test_case = {}, {}
y_output_U_train_case, y_output_U_test_case = {}, {}

for case_name in selected_training_cases:

    train_indices, test_indices = train_test_split(np.arange(len(C_coords[case_name])), test_size=test_size, shuffle=True, random_state=0)
    train_indices_case[case_name], test_indices_case[case_name] = train_indices, test_indices

    y_output_U_train_case[case_name], y_output_U_test_case[case_name] = weightsU[case_name][train_indices,:], weightsU[case_name][test_indices,:]
    x_input_U_train_case[case_name], x_input_U_test_case[case_name] = features[case_name][train_indices,:], features[case_name][test_indices,:]

x_input_U_train = np.vstack([x_input_U_train_case[case] for case in selected_training_cases])
x_input_U_test = np.vstack([x_input_U_test_case[case] for case in selected_training_cases])
y_output_U_train = np.vstack([y_output_U_train_case[case] for case in selected_training_cases])
y_output_U_test = np.vstack([y_output_U_test_case[case] for case in selected_training_cases])

# print(x_input_U_train.shape)
# print(x_input_U_test.shape)
# input()

# plt.figure()
# for i in range(x_input_U_test.shape[1]):
#     plt.subplot(3,4,i+1)
#     plt.hist(x_input_U_test[:,i], bins=30, alpha=0.7, label=f'feature {i+1}',density=True)
# plt.show()


#### Assigns the influence coefficients for the RFR training, in order to give the same influence to each flow case on the training
n_cells_train_case = { 
    case_name: len(train_indices_case[case_name])
    for case_name in selected_training_cases
    }
alpha_case = {
    case_name: 1.0 / n_cells_train_case[case_name]
    for case_name in selected_training_cases
    } 

mean_alpha = np.mean(list(alpha_case.values()))  #rebecca

sample_weights_list = []#
for case_name in selected_training_cases:
    n = n_cells_train_case[case_name]     # nb de points de train pour ce cas #rebecca
    w = alpha_case[case_name]/mean_alpha          # poids par point pour ce cas #rebecca
    sample_weights_list.append(w * np.ones(n)) #rebecca
rf_rms_ponderation_weights = np.concatenate(sample_weights_list) #rebecca


# results = Parallel(n_jobs=90)(delayed(single_neural_training)(x_input_U_train, x_input_U_test, y_output_U_train, y_output_U_test, i) for i in range(1000))

nn_model = torch.load("model_369.pth", weights_only=False)
nn_model.eval()
with torch.no_grad():
    y_pred_train = nn_model(torch.tensor(x_input_U_train, dtype=torch.float32))
    y_pred_test  = nn_model(torch.tensor(x_input_U_test, dtype=torch.float32))

plt.figure()
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.scatter(y_output_U_train[:,i],y_pred_train[:,i], s=5, alpha = 0.7)
    plt.subplot(2,3,i+4)
    plt.scatter(y_output_U_test[:,i],y_pred_test[:,i], s=5, alpha = 0.7)
plt.show()

models_location = f"{dirModels}/{FeaturesChoice}/{WhichWeights}"

features_dic = {'ANSJ':features, 'CHAN':features, 'SEP':features}
for case_name in selected_training_cases:

    model_name = 'ANSJ'
    
    path_save = f"{models_location}/{case_name}_{ML_model_choice}_{model_name}"
    
    print(f'Predict case {case_name}, save path: {path_save}')
    train_indices, test_indices = train_indices_case[case_name], test_indices_case[case_name]
    
    # ---------------------------------------------------------------------------------------
    weightsU_predicted = predict_weights_NN(features_dic[model_name][case_name], train_indices, test_indices, nn_model)
    
    weightsU_train_predicted, weightsU_test_predicted = weightsU_predicted[train_indices,:], weightsU_predicted[test_indices,:]
    
    plot_histograms_training_test_MLmodel(path_save, weightsU_train_predicted, weightsU[case_name][train_indices], weightsU_test_predicted, weightsU[case_name][test_indices], 'U')
                
    plot_weights_exact_predicted(path_save, C_coords[case_name], domain_bounds[case_name], weightsU[case_name], weightsU_predicted, 'U')
    
    plot_weights_predicted_vs_weights_exact(path_save, train_size, weightsU_train_predicted, weightsU[case_name][train_indices], weightsU_test_predicted, weightsU[case_name][test_indices], 'U')
        