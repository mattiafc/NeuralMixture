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
from utils_ML_orig               import *


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
FeaturesChoice = "../MouradDatabase"
SBL_models = ['ANSJ', 'CHAN', 'SEP']
crit_RFR = ['squared_error', 'absolute_error', "friedman_mse", "poisson"]

gridsearch = False

dirModels = f"../ML_trained_models"

# '''
# case_type: used to set the weights (in the sense of influence) of the training points on RF training.
#            The weights are set in order to give the same influence to each model on the training, even if the number of training points is different from one case to another.

# selected_training_cases: used lo load the model weights, features, iterate to perform train-valid split,
#                          compute the influence, and visualize the RFR model prediction

# keys_weights_export: used to export the weights to the foam files, and to visualize the predicted weights vs exact weights.                 
# '''

# # case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D", "LRN_OGV_trans"], 'SEP':[  "PH10595" , "CBFS13700", "CD12600"]} #
# # selected_training_cases = ["Jet_NearSonic_projected", "channel1000_2D", "PH10595", "CBFS13700", "CD12600", "LRN_OGV_trans"]
# # keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700", 'LRN_OGV_trans']
# # cases = ["Jet_NearSonic", "CD12600", "channel1000_2D", "PH10595", "CBFS13700", 'LRN_OGV_trans']#
# # models = ["Frozen", "ANSJ", "CHAN", "SEP"]# 

case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D"], 'SEP':["PH10595" , "CBFS13700", "CD12600"]} #
selected_training_cases = ['CD12600', 'channel1000_2D', 'PH10595', 'CBFS13700', 'Jet_NearSonic_projected']
keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]
cases = ["Jet_NearSonic", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]#
models = ["Frozen", "ANSJ", "CHAN", "SEP"]#

# case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D"], 'SEP':["PH10595" , "CBFS13700", "CD12600"]} #
# selected_training_cases = ["Jet_NearSonic_projected", "channel1000_2D", "PH10595", "CBFS13700", "CD12600"]
# keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]
# cases = ["CurvedDuct", "Channel", "CurvedBackwardFacingStep", "PeriodicHill"]#
# models = ["ANSJ", "CHAN", "SEP"]#


print ("post-process jet data, this will change the dic and add new Jet_proj")		
dic_data = create_dic_data(FeaturesChoice, cases, models)

# print(dic_data.keys())
# # Salva su disco
# with open('verify_dic_data.pkl', 'wb') as f:
#     pickle.dump(dic_data, f)

make_symm_Jet_data_on_PIVsubdomain(FeaturesChoice, dic_data, models, whichJet="Jet_NearSonic")
export_Jet_foam_files(FeaturesChoice, models, dic_data, 'Jet_NearSonic', 'projected', 5000)
restrict_Jet_data_to_upper_half_PIV_domain_bounds(FeaturesChoice, dic_data, models, whichJet="Jet_NearSonic")
export_Jet_foam_files(FeaturesChoice, models, dic_data, 'Jet_NearSonic', 'restricted', 5000)
augment_Jet_data_to_upper_half_PIV_domain_bounds(FeaturesChoice, dic_data, models, whichJet="Jet_NearSonic")


weightsU_org, features = mixture_of_expert_Grid_search(dic_data, sigma=1e-3)

weightsU = {key: weightsU_org[key] for key in selected_training_cases if key in weightsU_org}

# with open('verify_weightsU.pkl', 'wb') as f:
#     pickle.dump(weightsU, f)

features = {key: features[key]['ANSJ'] for key in selected_training_cases if key in features}
# with open('verify_features.pkl', 'wb') as f:
#     pickle.dump(features, f)

# print('**************************************************************************************')
# print(features.keys())
# print(selected_training_cases)
# for k in list(features.keys()):
#     print (f"features for case {k} have keys {features[k].keys()}")

C_coords = {key: dic_data[key]['CHAN']['internalMesh'].cell_centers().points for key in selected_training_cases}
# with open('verify_C_coords.pkl', 'wb') as f:
#     pickle.dump(C_coords, f)

domain_bounds = {key: dic_data[key]['CHAN']['internalMesh'].bounds for key in selected_training_cases}
# with open('verify_domain_bounds.pkl', 'wb') as f:
#     pickle.dump(domain_bounds, f)

# print ('\n--------------- Export exact smoothened weights')
# for case in keys_weights_export: 
#     if case == "Jet_NearSonic" : case_export = "Jet_NearSonic_restricted"
#     else : case_export = case
#     time_folder = "5000"
#     export_folder = f"./{FeaturesChoice}/{case}/ExactWeights/{WhichWeights}"
#     simul_folder = os.path.join(export_folder,time_folder)
#     boundary_data, nCells = read_boundary_data(f"{export_folder}/constant/polyMesh")
    
#     for i_ in range(3):
#         write_scalar_field(simul_folder, time_folder, f"wU{i_+1}_exact", weightsU_org[case_export][:,i_], boundary_data)


# #### Perform the train-valid split
# train_indices_case, test_indices_case = {}, {}	
# x_input_U_train_case, x_input_U_test_case = {}, {}
# y_output_U_train_case, y_output_U_test_case = {}, {}

# for case_name in selected_training_cases:

#     train_indices, test_indices = train_test_split(np.arange(len(C_coords[case_name])), test_size=test_size, shuffle=True, random_state=0)
#     train_indices_case[case_name], test_indices_case[case_name] = train_indices, test_indices

#     y_output_U_train_case[case_name], y_output_U_test_case[case_name] = weightsU[case_name][train_indices,:], weightsU[case_name][test_indices,:]
#     x_input_U_train_case[case_name], x_input_U_test_case[case_name] = features[case_name][train_indices,:], features[case_name][test_indices,:]

# x_input_U_train = np.vstack([x_input_U_train_case[case] for case in selected_training_cases])
# x_input_U_test = np.vstack([x_input_U_test_case[case] for case in selected_training_cases])
# y_output_U_train = np.vstack([y_output_U_train_case[case] for case in selected_training_cases])
# y_output_U_test = np.vstack([y_output_U_test_case[case] for case in selected_training_cases]) 
