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

    home_directory = setup_dict["simulation_home"]
    cases_dict = setup_dict["cases"]

    # print ("post-process jet data, this will change the dic and add new Jet_proj")
    dict_data = create_dic_data(home_directory, cases_dict)

    for case in cases_dict.keys():
        if cases_dict[case].get("jet", False):

            experts = cases_dict[case].get("models", {}).keys()

            interpolate_RANS_on_HF(home_directory, dict_data, "CHAN", RANS_case="Jet_NearSonic")
            
            # export_Jet_foam_files(home_directory, experts, dict_data, 'Jet_NearSonic', 'projected', 5000)
            # restrict_Jet_data_to_upper_half_PIV_domain_bounds(home_directory, dict_data, experts, RANS_case="Jet_NearSonic")
            # export_Jet_foam_files(home_directory, experts, dict_data, 'Jet_NearSonic', 'restricted', 5000)
            # augment_Jet_data_to_upper_half_PIV_domain_bounds(home_directory, dict_data, experts, RANS_case="Jet_NearSonic")
            
        query_cases = [key for key in dict_data.keys() if not(key in ["Jet_NearSonic","Jet_NearSonic_restricted","Jet_NearSonic_augmented"])]

    weightsU_org, features = generate_labels_features(dict_data, setup_dict["features"])

    weightsU = {key: weightsU_org[key] for key in query_cases if key in weightsU_org}
    features = {key: features[key]['ANSJ'] for key in query_cases if key in features}
    C_coords = {key: dict_data[key]['CHAN']['internalMesh'].cell_centers().points for key in query_cases}
    domain_bounds = {key: dict_data[key]['CHAN']['internalMesh'].bounds for key in query_cases}

    print(weightsU["CD12600"].shape)

    for case in dict_data.keys():
        if case == "Jet_NearSonic": case_export = "Jet_NearSonic_restricted"
        else: case_export = case
        time_folder = "5000"
        # export_folder = f"./{FeaturesChoice}/{case}/ExactWeights/{WhichWeights}"
        export_folder = os.path.join(home_directory, case, "Exact")
        simul_folder = os.path.join(export_folder,time_folder)
        boundary_data, nCells = read_boundary_data(os.path.join(export_folder,"constant/polyMesh"))
        
        #### RIMUOVERE DOMANI ####
        weights = ["ANSJ", "CHAN", "SEP"]

        for i_ in range(3):
            write_scalar_field(simul_folder, time_folder, f"w_{weights[i_]}_exact", weightsU_org[case_export][:,i_], boundary_data)


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