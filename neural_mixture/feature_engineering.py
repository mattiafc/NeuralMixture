import copy
import argparse
import json
import numpy             as np

from neural_mixture.utils_ML       import *
from neural_mixture.utils_OpenFOAM import *

def main():
    
    ap = argparse.ArgumentParser(
        prog="feature_engineering",
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

    home_directory   = setup_dict["database_home_directory"]
    output_directory = setup_dict["output_database_directory"]
    cases_dict       = setup_dict["cases"]
    HF_model         = setup_dict["HF_name"]
    features_list    = setup_dict["features"]
    models_list      = setup_dict["models_order"]

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
                dict_data[f"{case}_interpolated"][model] = interpolate_RANS_on_HF(dict_data, model, case, HF_model)[f"{case}_interpolated"][model]

                if not(model == HF_model):
                    generate_FOAM_case_from_pyvista(os.path.join(target_dir, model),
                                            dict_data[f"{case}_interpolated"][model]["internalMesh"], 
                                            dict_data[f"{case}_interpolated"][model]["boundary"], 
                                            os.path.join(home_directory, cases_dict[case]["sub_directory"], HF_model, 'constant/polyMesh'),
                                            time_value=int(cases_dict[case]["models"][model]))
                else:
                    
                    os.system(f'cp -r {os.path.join(home_directory, cases_dict[case]["sub_directory"], HF_model)} \
                          {os.path.join(target_dir, HF_model)}')
                
            del dict_data[f"{case}"]

        if cases_dict[case].get("interpolate_to_floder", False):
            case_folder = cases_dict[case]["interpolate_to_floder"]
            case_export = f"{case}_interpolated"
        else:
            case_folder = case
            case_export = case
        
        print(f"=========================================================")

        dataset = generate_labels_features(dict_data, case_export, features_list, models_list, HF_model)
        nPoints = dataset[f"w_{HF_model}_{models_list[0]}"].shape[0]//len(models_list)

        time_folder = setup_dict["time_folder"]
        
        export_folder = os.path.join(home_directory, case_folder, HF_model)
        simul_folder = os.path.join(export_folder,time_folder)
        boundary_data, _ = read_boundary_data(os.path.join(export_folder,"constant/polyMesh"))

        for i_, model in enumerate(models_list):
            write_scalar_field(simul_folder, time_folder, f"w_{HF_model}_{model}", dataset[f"w_{HF_model}_{model}"].to_numpy()[:nPoints], boundary_data)
        
        dataset.to_csv(os.path.join(output_directory, f"{case_export}_dataset.csv"), index=False)
        
        print(f"=========================================================")



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