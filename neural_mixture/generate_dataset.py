import pickle
import copy
import sys
import math
import scipy.ndimage
import scipy.interpolate
import argparse
import json

import pyvista           as pv
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import scipy             as sp 

from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils_ML               import *
from utils_OpenFOAM         import *


# sigma_train = 0. # Standard deviation for Gaussian kernel
# sigma_pred = 0.  # Standard deviation for Gaussian kernel
# #
# test_size=0.05
# train_size = 1.- test_size
#
ML_model_choice='RFR_Mourad'
latest_time_foldername ="5000"
PathToCases="../"
WhichWeights = "GaussSearchGrid"
FeaturesChoice = "../../Database"
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

# case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D"], 'SEP':["PH10595" , "CBFS13700", "CD12600"]} #
# selected_training_cases = ["Jet_NearSonic_projected", "channel1000_2D", "PH10595", "CBFS13700", "CD12600"]
# keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]
# cases = ["Jet_NearSonic", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]#
# models = ["Frozen", "ANSJ", "CHAN", "SEP"]#

# case_type = {'ANSJ':["Jet_NearSonic_projected"], 'CHAN':["channel1000_2D"], 'SEP':["PH10595" , "CBFS13700", "CD12600"]} #
# selected_training_cases = ["Jet_NearSonic_projected", "channel1000_2D", "PH10595", "CBFS13700", "CD12600"]
# keys_weights_export = ["Jet_NearSonic","Jet_NearSonic_augmented", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700"]
# cases = ["CurvedDuct", "Channel", "CurvedBackwardFacingStep", "PeriodicHill"]#
# models = ["ANSJ", "CHAN", "SEP"]#

def main():
    
    ap = argparse.ArgumentParser(
        prog="generate_dataset",
        description=(
            "Generate the dataset to be used to train the RFR model. \n"
            "You can take a look at a sample of the Json file in the examples directory. \n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", help="Path to JSON file containing the cases dict.")

    args = ap.parse_args()
    with open(args.config) as fh:
        setup_dict = json.load(fh)

    home_dict = setup_dict["simulation_home"]
    cases_dict = setup_dict["cases"]

    # print ("post-process jet data, this will change the dic and add new Jet_proj")
    dic_data = create_dic_data(home_dict, cases_dict)

    for case in cases_dict.keys():
        if cases_dict[case].get("jet", False):

            experts = cases_dict[case].get("models", {}).keys()

            make_symm_Jet_data_on_PIVsubdomain(home_dict, dic_data, experts, whichJet="Jet_NearSonic")
            export_Jet_foam_files(home_dict, experts, dic_data, 'Jet_NearSonic', 'projected', 5000)
            restrict_Jet_data_to_upper_half_PIV_domain_bounds(home_dict, dic_data, experts, whichJet="Jet_NearSonic")
            export_Jet_foam_files(home_dict, experts, dic_data, 'Jet_NearSonic', 'restricted', 5000)
            augment_Jet_data_to_upper_half_PIV_domain_bounds(home_dict, dic_data, experts, whichJet="Jet_NearSonic")
            
        query_cases = [key for key in dic_data.keys() if not(key in ["Jet_NearSonic","Jet_NearSonic_restricted","Jet_NearSonic_augmented"])]

    weightsU_org, features = mixture_of_expert_Grid_search(dic_data, setup_dict["features"])

    weightsU = {key: weightsU_org[key] for key in query_cases if key in weightsU_org}
    features = {key: features[key]['ANSJ'] for key in query_cases if key in features}
    C_coords = {key: dic_data[key]['CHAN']['internalMesh'].cell_centers().points for key in query_cases}
    domain_bounds = {key: dic_data[key]['CHAN']['internalMesh'].bounds for key in query_cases}

# print ('\n--------------- Export exact smoothened weights')
# for case in dic_data.keys():
#     if case == "Jet_NearSonic": case_export = "Jet_NearSonic_restricted"
#     else: case_export = case
#     time_folder = "5000"
#     export_folder = f"./{FeaturesChoice}/{case}/ExactWeights/{WhichWeights}"
#     simul_folder = os.path.join(export_folder,time_folder)
#     boundary_data, nCells = read_boundary_data(f"{export_folder}/constant/polyMesh")
    
#     for i_ in range(3):
#         write_scalar_field(simul_folder, time_folder, f"wU{i_+1}_exact", weightsU_org[case_export][:,i_], boundary_data)


# # #### Perform the train-valid split
# # train_indices_case, test_indices_case = {}, {}	
# # x_input_U_train_case, x_input_U_test_case = {}, {}
# # y_output_U_train_case, y_output_U_test_case = {}, {}

# # for case_name in selected_training_cases:

# #     train_indices, test_indices = train_test_split(np.arange(len(C_coords[case_name])), test_size=test_size, shuffle=True, random_state=0)
# #     train_indices_case[case_name], test_indices_case[case_name] = train_indices, test_indices

# #     y_output_U_train_case[case_name], y_output_U_test_case[case_name] = weightsU[case_name][train_indices,:], weightsU[case_name][test_indices,:]
# #     x_input_U_train_case[case_name], x_input_U_test_case[case_name] = features[case_name][train_indices,:], features[case_name][test_indices,:]

# # x_input_U_train = np.vstack([x_input_U_train_case[case] for case in selected_training_cases])
# # x_input_U_test = np.vstack([x_input_U_test_case[case] for case in selected_training_cases])
# # y_output_U_train = np.vstack([y_output_U_train_case[case] for case in selected_training_cases])
# # y_output_U_test = np.vstack([y_output_U_test_case[case] for case in selected_training_cases]) 

if __name__ == "__main__":
    main()