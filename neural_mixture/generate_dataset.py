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

def mirror_symmetric_data(internalMesh):

    QoIs_list = list(dict.fromkeys(internalMesh.array_names))
    QoIs_indices = {}

    cont = 0

    for QoI in QoIs_list:

        cont_old = copy.deepcopy(cont)
        
        QoI_original_mesh = internalMesh[QoI]
        
        if len(QoI_original_mesh.shape) == 1:
            QoI_original_mesh = QoI_original_mesh.reshape(-1, 1)
            idx_flip = 0
            idx_reorder = [0]

        elif QoI_original_mesh.shape[1] == 3:
            idx_flip = -1
            idx_reorder = [0,2,1]
            
        elif QoI_original_mesh.shape[1] == 6:
            idx_flip = [2,4]
            idx_reorder = [0,2,1,5,4,3]

        QoI_mirrored = np.vstack((QoI_original_mesh, QoI_original_mesh))
        nPoints, nComp = QoI_original_mesh.shape
        
        if QoI in ['eta8','Cz'] or nComp>1:
            QoI_mirrored[nPoints:, idx_flip] *= -1

        if cont == 0:
            QoI_mirrored_all = copy.deepcopy(QoI_mirrored[:,idx_reorder])
        else:        
            QoI_mirrored_all = np.hstack((QoI_mirrored_all, QoI_mirrored[:,idx_reorder]))

        cont += nComp

        QoIs_indices[QoI] = [cont_old, cont]

    return QoI_mirrored_all, copy.deepcopy(QoIs_indices)
    
def interpolate_RANS_on_HF(home_directory, dic_data, expert_name, RANS_case, Exact_case="Exact"):
    ''' 
    Post-treat jet case:\n 
    scale * 0.0508 and rotate by 90° on x-axis
    '''

    # Transform the target mesh (PIV mesh) to match the RANS mesh orientation and scale
    targetMesh = dic_data[RANS_case][Exact_case]["internalMesh"].rotate_x(90, inplace=False)
    targetMesh.points *= 0.0508

    dic_data[f"{RANS_case}_interpolated"][expert_name] = copy.deepcopy(dic_data[RANS_case][Exact_case])

    target_mesh = targetMesh.cell_centers().points
    source_half_mesh = dic_data[RANS_case][expert_name]["internalMesh"].cell_centers().points

    nPoints = source_half_mesh.shape[0]
    source_mesh = np.vstack((source_half_mesh, source_half_mesh))
    source_mesh[nPoints:, -1] *= -1

    source_QoIs, QoI_indices = mirror_symmetric_data(dic_data[RANS_case][expert_name]["internalMesh"])
    target_QoIs = interpolate_rbf(source_mesh[:,0], source_mesh[:,2], source_QoIs, target_mesh[:,0], target_mesh[:,2], neighbors=240)

    for QoI, idx in QoI_indices.items():
        if idx[0] == idx[1]-1:
            dic_data[f"{RANS_case}_interpolated"][expert_name]["internalMesh"][QoI] = target_QoIs[:, idx[0]]
        else:
            dic_data[f"{RANS_case}_interpolated"][expert_name]["internalMesh"][QoI] = target_QoIs[:, idx[0]:idx[1]]

    return dic_data

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
    bsl_model = setup_dict["baseline_model"]

    # print ("post-process jet data, this will change the dic and add new Jet_proj")
    dict_data = create_dic_data(home_directory, cases_dict)

    for case in cases_dict.keys():
        if cases_dict[case].get("interpolate_to_floder", False):

            experts = cases_dict[case].get("models", {}).keys()

            dict_data[f"{case}_interpolated"] = {}
            target_dir = os.path.join(home_directory, cases_dict[case]["interpolate_to_floder"])
            os.system(f'rm -rf {target_dir}')
            os.mkdir(target_dir)

            for model in experts:
                dict_data[f"{case}_interpolated"][model] = {}
                dict_data[f"{case}_interpolated"][model] = interpolate_RANS_on_HF(home_directory, dict_data, model, case, setup_dict['HF_name'])[f"{case}_interpolated"][model]

                if not(model == setup_dict["HF_name"]):
                    generate_FOAM_case_from_pyvista(os.path.join(target_dir, model),
                                            dict_data[f"{case}_interpolated"][model]["internalMesh"], 
                                            dict_data[f"{case}_interpolated"][model]["boundary"], 
                                            os.path.join(home_directory, cases_dict[case]["sub_directory"], setup_dict['HF_name'], 'constant/polyMesh'),
                                            time_value=5000)
                else:
                    
                    os.system(f'cp -r {os.path.join(home_directory, cases_dict[case]["sub_directory"], setup_dict["HF_name"])} \
                          {os.path.join(target_dir, setup_dict["HF_name"])}')
                
            del dict_data[f"{case}"]


    weightsU_org, features = generate_labels_features(dict_data, setup_dict["features"], setup_dict["models_order"])

    weightsU = {key: weightsU_org[key] for key in dict_data.keys()  if key in weightsU_org}
    features = {key: features[key]['CHAN'] for key in dict_data.keys() if key in features}
    C_coords = {key: dict_data[key]['CHAN']['internalMesh'].cell_centers().points for key in dict_data.keys()}
    domain_bounds = {key: dict_data[key]['CHAN']['internalMesh'].bounds for key in dict_data.keys()}

    print(weightsU["CD12600"].shape)

    print("Weights calculated, now exporting the data in OpenFOAM format")
    for case in cases_dict.keys():
        print(case)
        if cases_dict[case].get("interpolate_to_floder", False):
            case_folder = cases_dict[case]["interpolate_to_floder"]
            case_export = f"{case}_interpolated"
        else:
            case_folder = case
            case_export = case

        time_folder = "5000"
        
        export_folder = os.path.join(home_directory, case_folder, "Exact")
        simul_folder = os.path.join(export_folder,time_folder)
        boundary_data, nCells = read_boundary_data(os.path.join(export_folder,"constant/polyMesh"))

        for i_ in range(3):
            write_scalar_field(simul_folder, time_folder, f"w_{setup_dict["models_order"][i_]}_exact", weightsU_org[case_export][:,i_], boundary_data)

    ML_dataset = pd.DataFrame()



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