import pickle
import copy
import sys
import math
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


sigma_train = 0. # Standard deviation for Gaussian kernel
sigma_pred = 0.  # Standard deviation for Gaussian kernel
#
test_size=0.05
train_size = 1.- test_size
#
ML_model_choice='RFR_Mourad'
latest_time_foldername ="5000"
PathToCases="../"
WhichWeights = "GaussSearchGrid"
FeaturesChoice = "Ling"
SBL_models = ['ANSJ', 'CHAN', 'SEP']
crit_RFR = ['squared_error', 'absolute_error', "friedman_mse", "poisson"]

gridsearch = False

dirModels = f"../ML_trained_models"

'''
case_type: used to set the weights (in the sense of influence) of the training points on RF training.
           The weights are set in order to give the same influence to each model on the training, even if the number of training points is different from one case to another.

selected_training_cases: used lo load the model weights, features, iterate to perform train-valid split,
                         compute the influence, and visualize the RFR model prediction

keys_weights_export: used to export the weights to the foam files, and to visualize the predicted weights vs exact weights.                 
'''

# case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D", "LRN_OGV_trans"], 'SEP':[  "PH10595" , "CBFS13700", "CD12600"]} #
# selected_training_cases = ["Jet_NearSonic_projected", "channel1000_2D", "PH10595", "CBFS13700", "CD12600", "LRN_OGV_trans"]
# keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700", 'LRN_OGV_trans']
# cases = ["Jet_NearSonic", "CD12600", "channel1000_2D", "PH10595", "CBFS13700", 'LRN_OGV_trans']#
# models = ["Frozen", "ANSJ", "CHAN", "SEP"]# 

case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D"], 'SEP':["PH10595" , "CBFS13700", "CD12600"]} #
selected_training_cases = ["Jet_NearSonic_projected", "channel1000_2D", "PH10595", "CBFS13700", "CD12600"]
keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]
cases = ["Jet_NearSonic", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]#
models = ["Frozen", "ANSJ", "CHAN", "SEP"]#

#Setup to run training on ADP & testing on OP1
case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D"], 'SEP':["PH10595" , "CBFS13700", "CD12600","LRN_OGV_ADP"]} #
selected_training_cases = ["Jet_NearSonic_projected", "channel1000_2D", "PH10595", "CBFS13700", "CD12600","LRN_OGV_ADP"]
keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700","LRN_OGV_ADP"]
#Check this line
cases = ["Jet_NearSonic", "CD12600", "channel1000_2D", "PH10595", "CBFS13700","LRN_OGV_ADP","LRN_OGV_OP1"]#
models = ["Frozen", "ANSJ", "CHAN", "SEP"]#


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


print (f'\n--------------- Train {ML_model_choice} models to learn the mapping : eta ----> w')
print (f'--------------- Saving in Models/{ML_model_choice}_log.txt')

gridsearch_results = pd.DataFrame(columns=['n_estimators', 'min_samples', 'criterion', 'max_features', 'q2_train', 'r2_train', 'q2_test', 'r2_test'])
np.random.seed(0)

best_r2 = 0

if gridsearch == True:
    for i in range(100):

        n_estim     = np.random.randint(50,301)
        min_samples = np.random.randint(1,15)
        crit_idx    = np.random.randint(0,4)
        max_feat    = np.random.randint(3,13)

        hyperparams = {'n_estimators':n_estim, 'min_samples':min_samples,'criterion':crit_RFR[crit_idx],'max_features':max_feat, 'bootstrap':True}
        temp_model_U, temp_train_metrics, temp_test_metrics = train_random_forest(x_input_U_train, x_input_U_test, y_output_U_train, y_output_U_test, rf_rms_ponderation_weights,hyperparams)
        gridsearch_results = pd.concat([gridsearch_results, pd.DataFrame([{'n_estimators':n_estim, 'min_samples':min_samples,'criterion':crit_RFR[crit_idx],'max_features':max_feat
                                                    ,'q2_train':temp_train_metrics['q2'], 'r2_train':temp_train_metrics['r2']
                                                    ,'q2_test':temp_test_metrics['q2'], 'r2_test':temp_test_metrics['r2']}])], ignore_index=True)
        if temp_test_metrics['r2'] > best_r2:
            print('\n============= New Best Model ==============')
            train_metrics = copy.deepcopy(temp_train_metrics)
            test_metrics = copy.deepcopy(temp_test_metrics)
            model_U = copy.deepcopy(temp_model_U)
            best_r2 = test_metrics['r2']
            print(f'{hyperparams}')
            print(f'Q2 test: {test_metrics['q2']:.4f}; R2 test: {best_r2:.4f}')
            print('============== Saving Model ===============')
            with open(f"./best_model.pkl", "wb") as f: pickle.dump(model_U, f)
            
        sys.stdout.write(f'{i} ')
    

    gridsearch_results.to_csv(f"./gridsearch_results.csv", index=False)
else:
    model_U, train_metrics, test_metrics = train_random_forest(x_input_U_train, x_input_U_test, y_output_U_train, y_output_U_test, rf_rms_ponderation_weights)


model_log_train = f"Q2 criterion train : {train_metrics['q2']:.3f}" 
model_log_train += f"\nR2 criterion train : {train_metrics['r2']:.3f}"

model_log_test = f"\nQ2 criterion test : {test_metrics['q2']:.3f}"
model_log_test += f"\nR2 criterion test : {test_metrics['r2']:.3f}"

models_location = f"{dirModels}/{FeaturesChoice}/{WhichWeights}"

with open(f"Models/{ML_model_choice}_log.txt", 'w') as file:	
    models_log = f"{ML_model_choice} convergence info :\n\nfeatures : {FeaturesChoice}\t weights : {WhichWeights}"
    models_log += f"\n\nU\nTrain :\n" + model_log_train + f"\nTest : " + model_log_test
#			models_log += f"\n\nvariable : all\n" + model_log_all
    file.write(models_log)

with open(f"Models/{ML_model_choice}.pickle", "wb") as f:pickle.dump(model_U ,f)

# features_dic = {'ANSJ':features, 'CHAN':features, 'SEP':features}
# for case_name in selected_training_cases:
    
#     for model_name in ['ANSJ', 'CHAN', 'SEP']:
    
#         path_save = f"{models_location}/{case_name}_{ML_model_choice}_{model_name}"
        
#         print(f'Predict case {case_name}, save path: {path_save}')
#         train_indices, test_indices = train_indices_case[case_name], test_indices_case[case_name]
        
#         # ---------------------------------------------------------------------------------------
#         weightsU_predicted = predict_smoothed_weights_training_test_MLmodel(features_dic[model_name][case_name], train_indices, test_indices, model_U, sigma_pred, C_coords[case_name], domain_bounds[case_name])
        
#         weightsU_train_predicted, weightsU_test_predicted = weightsU_predicted[train_indices,:], weightsU_predicted[test_indices,:]
        
#         plot_histograms_training_test_MLmodel(path_save, weightsU_train_predicted, weightsU[case_name][train_indices], weightsU_test_predicted, weightsU[case_name][test_indices], 'U')
                    
#         plot_weights_exact_predicted(path_save, C_coords[case_name], domain_bounds[case_name], weightsU[case_name], weightsU_predicted, 'U')
        
#         plot_weights_predicted_vs_weights_exact(path_save, train_size, model_log_train, model_log_test, weightsU_train_predicted, weightsU[case_name][train_indices], weightsU_test_predicted, weightsU[case_name][test_indices], 'U')
            
