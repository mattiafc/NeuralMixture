import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.spatial.distance import cdist
from scipy.spatial          import cKDTree
from scipy.interpolate      import griddata

from sklearn.model_selection            import cross_val_score
from sklearn.ensemble                   import RandomForestRegressor
from sklearn.metrics                    import mean_absolute_error

def load_data(path):

    '''Load OpenFOAM data from the specified path using PyVista.'''

    with open(f"{path}/case.foam", 'w') as _ : pass  # Just create an empty file
    reader = pv.OpenFOAMReader(f"{path}/case.foam")
    time_values = reader.time_values
    reader.set_active_time_value(time_values[-1])
    mesh = reader.read()
    internalMesh = mesh["internalMesh"]
    boundaries = mesh["boundary"]
    # nbr_points = internalMesh.n_points #M
    # nbr_cells = internalMesh.n_cells #M
    
    return internalMesh, boundaries

def create_dic_data(FeaturesChoice, cases, models) :
    
    '''
    Create a dictionary to store internal mesh and boundary data for each case and model.\n
    Cases : list of flow cases
    Models : list of expert models name
    '''
    dic_data = {}

    for case in cases:
        print(f"------ load case {case}--------------------------")
        dic_data.update({case:{}})

        for model in models:
            path = f"{FeaturesChoice}/{case}/{model}"
            internalMesh, boundaries = load_data(path)
            print (f"model {model}")
            dic_data[case].update({model:{"internalMesh":internalMesh, "boundary": boundaries, "nCells":len(internalMesh.cell_centers().points)}})
            
    return dic_data

def mixture_of_expert_Grid_search(dic_data, sigma):

    features = {}
    weightsU ={}

    for case in dic_data:

        features.update({case : {}})

        # condition set to take into account the projected data Jet_NearSonic_proj
        if case != 'Jet_NearSonic'and case != 'Jet_subSonic': 

            # internal mesh, getting the U,V components of the high fidelity solution
            U_HF = dic_data[case]['Frozen']['internalMesh']['U'][:,0:2]#.reshape((-1,1)) # changed rebecca 
            weightsU_case = []
            list_U_models= []
            
            for model in dic_data[case]:
                if model != 'Frozen':
                    print(case, model)
                    U_model = dic_data[case][model]['internalMesh']['U'][:,0:2]#.reshape((-1,1)) # changed rebecca 
                    
                    if case == "LRN_OGV_trans" or case == "LRN_OGV_trans_OP1":
                        U_HF = (U_HF - U_HF.mean(axis=0))/ U_HF.std(axis=0)
                        U_model = (U_model - U_model.mean(axis=0)) / U_model.std(axis=0)
                    list_U_models.append(U_model)
                    
                    features[case].update({model : np.hstack([np.array(dic_data[case][model]['internalMesh'][f'eta_{k_}']).reshape((-1,1)) for k_ in range(1,12)]) }) 

            
            weightsU_case = select_best_weights_sigma(U_HF, list_U_models)
            weightsU.update({case : np.hstack([ np.array( w_ / sum(weightsU_case)).reshape((-1,1)) for w_ in weightsU_case]) })

            # boundaries
            ErrorUV = np.zeros(U_HF.shape); ErrorUV[:,:] = U_HF[:,:]
            n_models = len(weightsU_case)   # au lieu de range(3) en dur
            for j_ in range(n_models):
                # weightsU_case[j_] : shape (N,)
                # list_U_models[j_] : shape (N,2)
                ErrorUV -= weightsU_case[j_].reshape(-1, 1) * list_U_models[j_]
                # broadcasting → (N,1)*(N,2) = (N,2)
            norm_ErrorUV = np.linalg.norm(ErrorUV)
            # print(f'|| U,V_models * W - (U,V)_ref || = {norm_ErrorUV}')
        
    return weightsU, features

def select_best_weights_sigma(U_HF, list_U_models):

    sigma_list = [1., 0.5, 1e-1, 0.5e-1, 1e-2, 0.5e-2, 1e-3, 0.5e-3, 1e-4]

    all_weights = []
    error_sigma = []
     
    for sigma in sigma_list:
        weights_case = []
          
        for k_ in range(len(list_U_models)):
               
            diff = np.linalg.norm(U_HF - list_U_models[k_], axis=1).reshape((-1,1))
            weight_k = 1e-12 + np.exp( -0.5* diff**2 / sigma**2)
            weights_case.append( weight_k )
               
        weights_case_sum1 = [weights_case[k_] / sum(weights_case) for k_ in range(len(list_U_models))]
        all_weights.append(weights_case_sum1)
        error_sigma.append(calculate_error_(U_HF, list_U_models, weights_case_sum1))
          
    min_index = np.argmin(error_sigma)
    best_weights = all_weights[min_index]

    print(f'Best sigma is {sigma_list[np.argmin(error_sigma)]}')
     
    return best_weights

def calculate_error_(U_HF, list_U_models, weightsU_case):
     
    ErrorU = np.zeros(U_HF.shape); ErrorU[:,:] = U_HF[:,:]
     
    for j_ in range(len(list_U_models)):
        for k_ in range(U_HF.shape[1]) : ErrorU[:,k_] -= weightsU_case[j_][:,0] * list_U_models[j_][:,k_] 
          
    return np.linalg.norm(ErrorU)

def train_random_forest(X_train, X_test, Y_train, Y_test, rf_rms_ponderation_weights,
                        hyperparameters = {'n_estimators':200, 'min_samples':1,'criterion':'squared_error'
                                          ,'max_features':7, 'bootstrap':True}):
    """Perform Random Forest regression to generate surface response"""

    n_estim      = hyperparameters['n_estimators']
    split_crit   = hyperparameters['criterion']
    min_samples  = hyperparameters['min_samples']
    max_features = hyperparameters['max_features']
    boot         = hyperparameters['bootstrap']

    rf = RandomForestRegressor(
        n_estimators=n_estim,
        criterion=split_crit,
        min_samples_leaf=min_samples,
        max_features=max_features,
        random_state=0,
        bootstrap=boot,
        n_jobs=90
    )

    # rf = RandomForestRegressor(
    #     n_estimators=200,
    #     criterion='squared_error',
    #     min_samples_leaf=1,
    #     max_features=7,
    #     random_state=0,
    #     n_jobs=90
    # )

    # Fit the Random Forest model
    rf.fit(X_train, Y_train, sample_weight=rf_rms_ponderation_weights)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # Calculate R2 for the training set
    r2_score_train = rf.score(X_train, Y_train)
    
    q2_score_train = 1 - mean_absolute_error(Y_train, y_train_pred)/np.mean(Y_train)
    r2_score_train = rf.score(X_train, Y_train)
    
    q2_score_test = 1 - mean_absolute_error(Y_test, y_test_pred)/np.mean(Y_test)
    r2_score_test = rf.score(X_test, Y_test)
    
    model_log_train = f"Q2 criterion train : {q2_score_train:.3f}" 
    model_log_train += f"\nR2 criterion train : {r2_score_train:.3f}"
    
    model_log_test = f"\nQ2 criterion test : {q2_score_test:.3f}"
    model_log_test += f"\nR2 criterion test : {r2_score_test:.3f}"

    # print(f"Q2 train: {q2_score_train:.3f}, R2 train: {r2_score_train:.3f}")
    # print(f"Q2 test: {q2_score_test:.3f}, R2 test: {r2_score_test:.3f}")

    return rf, {'q2':q2_score_train, 'r2':r2_score_train}, {'q2':q2_score_test, 'r2':r2_score_test}

def plot_weights_predicted_vs_weights_exact(path_save, train_size, weightsU_train_predicted, weightsU_train_exact, weightsU_test_predicted, weightsU_test_exact, weightName):
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12)) # figsize=(16, 8)
    for k in range(3):
        ax1.scatter(weightsU_train_exact[:,k], weightsU_train_predicted[:,k], alpha=0.5, marker='D')
        ax2.scatter(weightsU_test_exact[:,k], weightsU_test_predicted[:,k], alpha=0.5, marker='o')	
    
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    # Add labels and title
    ax1.set_xlabel(fr'Exact trained, train:{train_size}$')
    ax1.set_ylabel(r'$Predicted trained weights$')
    ax2.set_xlabel(fr'$Exact trained, test:{1-train_size}$')
    ax2.set_ylabel(r'$Predicted trained weights$')
    
    # ax1.set_title(f'{model_log_U_train}')
    # ax2.set_title(f'{model_log_U_test}')
    
    plt.legend()
    
    plt.savefig(f'{path_save}_weights_predicted_vs_weights_exact_{weightName}.png')
    # plt.show()
    plt.close()
    

def predict_smoothed_weights_training_test_MLmodel(features, train_indices, test_indices, model_U):
    
    weightsU_train_predicted = model_U.predict(features[train_indices,:])
    weightsU_test_predicted = model_U.predict(features[test_indices,:])

    weightsU_predicted=np.zeros((len(weightsU_train_predicted)+len(weightsU_test_predicted),3))
    weightsU_predicted[train_indices,:] = weightsU_train_predicted
    weightsU_predicted[test_indices,:] = weightsU_test_predicted
            
    return weightsU_predicted #weightsU_train_predicted, weightsU_test_predicted
    

def predict_weights_NN(features, train_indices, test_indices, model_U):

    with torch.no_grad():
        weightsU_train_predicted = model_U(torch.tensor(features[train_indices,:], dtype=torch.float32))
        weightsU_test_predicted = model_U(torch.tensor(features[test_indices,:], dtype=torch.float32))
    
    print(f"Dimensions are: {weightsU_train_predicted.shape}, {weightsU_test_predicted.shape}")

    weightsU_predicted=np.zeros((len(weightsU_train_predicted)+len(weightsU_test_predicted),3))
    weightsU_predicted[train_indices,:] = weightsU_train_predicted
    weightsU_predicted[test_indices,:] = weightsU_test_predicted
            
    return weightsU_predicted #weightsU_train_predicted, weightsU_test_predicted
    
def plot_histograms_training_test_MLmodel(path_save,
    weightsU_train_predicted, weightsU_train_exact,
    weightsU_test_predicted,  weightsU_test_exact,
    weightName):

    bin_width = 0.025

    # Nom des modèles experts
    expert_names = [r'$ANSJ$', r'$kwSST$', r'$SEP$']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True, sharex=True)

    data_min =  1000000000
    data_max = -1000000000
    bin_width = 0.025

    for k in range(3):
        data_min = min([np.min(weightsU_train_predicted[:,k] - weightsU_train_exact[:,k]), 
                        np.min(weightsU_test_predicted[:,k] - weightsU_test_exact[:,k]),
                        data_min])
                        
        data_max = max([np.max(weightsU_train_predicted[:,k] - weightsU_train_exact[:,k]), 
                        np.max(weightsU_test_predicted[:,k] - weightsU_test_exact[:,k]),
                        data_max])
        
    min_edge = bin_width * np.floor(data_min / bin_width)
    max_edge = bin_width * np.ceil(data_max / bin_width)
    bins = np.arange(min_edge-0.5*bin_width, max_edge + bin_width*0.5, bin_width)

    for k in range(3):

        error_k_train = weightsU_train_predicted[:,k] - weightsU_train_exact[:,k]
        ax1.hist(error_k_train,
                 bins=bins,
                 density=True,
                 alpha=0.55,
                 color=colors[k],
                 label=expert_names[k],
                 edgecolor='black',
                 linewidth=0.5)
    
    for k in range(3):
        error_k_test = weightsU_test_predicted[:,k] - weightsU_test_exact[:,k]
        ax2.hist(error_k_test,
                 bins=bins,
                 density=True,
                 alpha=0.55,
                 color=colors[k],
                 label=expert_names[k],
                 edgecolor='black',
                 linewidth=0.5)
        
        
    ax1.set_title(r"$Training~set~error~distribution$", fontsize=20)
    ax1.set_xlabel(r'$Error$', fontsize=25)
    ax1.set_ylabel(r'$Density [-]$', fontsize=25)
    ax1.autoscale()
    ax1.grid(alpha=0.3)

    ax2.set_title(r"$Test~set~error~distribution$", fontsize=20)
    ax2.set_xlabel(r'$Error$', fontsize=25)
    ax2.grid(alpha=0.3)
    ax2.autoscale()

    # Légende globale
    ax1.legend(title=r"$Experts$", fontsize=20)

    plt.tight_layout()
    plt.savefig(f'{path_save}_error_histograms_{weightName}.png', dpi=300)

    # plt.show()
    plt.close()

def calculate_threshold(x, y, percentile=90):

    tree = cKDTree(np.c_[x, y])
    distances, _ = tree.query(np.c_[x, y], k=2)
    nearest_distances = distances[:, 1]  # Ignore the zero distance to the point itself
    threshold = np.percentile(nearest_distances, percentile)*2.

    return threshold, nearest_distances

def interpolate_with_threshold(x, y, values, grid_x, grid_y, threshold):

    grid_values = griddata((x, y), values, (grid_x, grid_y), method='nearest')
    tree = cKDTree(np.c_[x, y])
    distances, _ = tree.query(np.c_[grid_x.ravel(), grid_y.ravel()], k=1)
    grid_values = grid_values.ravel()
    grid_values[distances > threshold] = np.nan

    return grid_values.reshape(grid_x.shape)

def plot_weights_exact_predicted(path_save, C_coords, domain_bounds, weights_exact, weights_predicted, weightName):

    x, y = C_coords[:, 0], C_coords[:, 1]
    x_min, x_max, y_min, y_max, z_min, z_max = domain_bounds
    grid_x, grid_y = np.mgrid[x_min:x_max:1000j, y_min:y_max:1000j]
    
    # Calculate global min and max values
    global_min_value = min(np.min(weights_exact), np.min(weights_predicted))
    global_max_value = max(np.max(weights_exact), np.max(weights_predicted))
    
    fig, axs = plt.subplots(3, 3, figsize=(12,8))
    threshold, _ = calculate_threshold(x, y, percentile=90)
    models = ['ANSJ', 'kwSST', 'SEP']
    
    for j_ in range(3):

        # Interpolate the predicted weights with threshold
        grid_weight_pred_j = interpolate_with_threshold(x, y, weights_predicted[:, j_], grid_x, grid_y, threshold)

        # Interpolate the exact weights with threshold
        grid_weights_exact_j = interpolate_with_threshold(x, y, weights_exact[:, j_], grid_x, grid_y, threshold)
        
        percentage_error = 100. * np.linalg.norm(weights_predicted[:, j_] - weights_exact[:, j_]) / (np.linalg.norm(weights_exact[:, j_]) + 1e-12)
        
        # Plot the exact weights
        im1 = axs[0, j_].pcolormesh(grid_x, grid_y, grid_weights_exact_j, shading='auto', cmap='viridis', vmin=global_min_value, vmax=global_max_value)
        fig.colorbar(im1, ax=axs[0, j_], label=r'$Exact~Weights$')
        axs[0, j_].set_xlabel(r'$x$')
        axs[0, j_].set_ylabel(r'$y$')
        axs[0, j_].set_title(fr'$Exact~w${weightName}$~{models[j_]}$',  fontsize=20)
        
        # Plot the predicted weights
        im2 = axs[1, j_].pcolormesh(grid_x, grid_y, grid_weight_pred_j, shading='auto', cmap='viridis', vmin=global_min_value, vmax=global_max_value)
        fig.colorbar(im2, ax=axs[1, j_], label=r'$Predicted~Weights$')
        axs[1, j_].set_xlabel(r'$x$')
        axs[1, j_].set_ylabel(r'$y$')
        #axs[1, j_].set_title(fr'$Predicted~w${weightName}_${models[j_]} : error = {percentage_error:.1f}\%$',    fontsize=15)
        axs[1, j_].set_title(fr'$Error~{models[j_]} : {percentage_error:.1f}\%$',    fontsize=20)

        # Plot the predicted weights
        im3 = axs[2, j_].pcolormesh(grid_x, grid_y, np.abs(grid_weight_pred_j-grid_weights_exact_j), shading='auto', cmap='viridis')
        fig.colorbar(im3, ax=axs[2, j_], label=r'$Errors$')
        axs[2, j_].set_xlabel(r'$x$' )
        axs[2, j_].set_ylabel(r'$y$')
        #axs[2, j_].set_title(fr'$Error~w${weightName}_${models[j_]}$', fontsize=15)
        axs[2, j_].set_title(fr'$Error~w${weightName}$~{models[j_]}$', fontsize=20)
        
    plt.tight_layout()
    
    plt.savefig(f'{path_save}_weights_exact_predicted_{weightName}.png')
    # plt.show()
    plt.close()

def write_foam_file_with_bd(path_to_case, input_file, output_file, time_value, internalMesh, boundaries):

    # Read content of the original file
    with open(f"{path_to_case}/{time_value}/{input_file}", 'r') as fin:
        content = fin.readlines()
    
    patches = boundaries.keys()
    cells_coords_mesh = internalMesh.cell_centers().points
    
    nbr_cells = len(internalMesh[input_file])

    new_content = ''
    # Perform modifications on the internal field values
    for _, line in enumerate(content):
        
        if line.startswith('    location'):
            new_content += f'    location    \"{time_value}\";\n'
        elif line.startswith('    class'):
            new_content += line
            if line.split()[1] == 'volScalarField;' :
                nbr_comp = 1
                type_comp = 'scalar'
            elif line.split()[1] == 'volVectorField;' :
                nbr_comp = 3
                type_comp = 'vector'
            elif line.split()[1] == 'volSymmTensorField;' :
                nbr_comp = 6
                type_comp = 'symmTensor'
            
        elif line.startswith('    object'):
            new_content += f'    object    {output_file};\n'
            
        elif line.startswith('internalField   nonuniform List<scalar>'):
            new_content += f'internalField   nonuniform List<scalar>\n{nbr_cells}\n(\n'
            for value in list(internalMesh[input_file]):
                new_content += f'{value}\n' 
            new_content +=  ');\n\n'
            break

        elif line.startswith('internalField   nonuniform List<vector>'):
            new_content += f'internalField   nonuniform List<vector>\n{nbr_cells}\n(\n'
            for values in internalMesh[input_file]:
                v1, v2, v3 = values
                new_content += f'({v1} {v2} {v3})\n' 
            new_content +=  ');\n\n'
            break

        elif line.startswith('internalField   nonuniform List<symmTensor>'):
            new_content += f'internalField   nonuniform List<symmTensor>\n{nbr_cells}\n(\n'
            for values in internalMesh[input_file]:
                v1, v2, v3, v4, v5, v6 = values
                new_content += f'({v1} {v2} {v3} {v4} {v5} {v6})\n' 
            new_content +=  ');\n\n'
            break

        else :
            new_content += line	
    
    new_content += "boundaryField\n{\n"
    
    for patch_name in patches:

        if patch_name in ['front', 'back', 'axis', 'frontAndBack', 'sides1_half0', 'sides1_half1', 'sides2_half0', 'sides2_half1', 'sideRight_half0', 'sideRight_half1', 'sideLeft_half0', 'sideLeft_half1']:
            new_content += f"\t{patch_name}\n\t{{\n\t\ttype\t\t\tempty;\n\t}}\n"
        
        elif patch_name in ['wedge1', 'wedge2'] :
            new_content += f"\t{patch_name}\n\t{{\n\t\ttype\t\t\twedge;\n\t}}\n"
        
        elif patch_name in ['inout1_half0', 'inout1_half1', 'inout2_half0', 'inout2_half1', 'inlet_half0', 'inlet_half1', 'outlet_half0', 'outlet_half1']:
            new_content += f"\t{patch_name}\n\t{{\n\t\ttype\t\t\tcyclic;\n\t\tvalue\t\tuniform 0;\n\t}}\n"
        
        elif patch_name == 'symmetry':
            new_content += f"\t{patch_name}\n\t{{\n\t\ttype\t\t\tsymmetryPlane;\n\t}}\n"
        
        else :
            patch_values_array = boundaries[patch_name][input_file]
            bd_points = boundaries[patch_name].cell_centers().points
            nbr_values_patch = len(patch_values_array)			
            new_content += f"\t{patch_name}\n\t{{\n\t\ttype\t\t\tcalculated;\n\t\tvalue\t\t\tnonuniform List<{type_comp}>\n{nbr_values_patch}\n(\n"

            for i_, _ in enumerate(patch_values_array):
                idx = np.argmin(np.linalg.norm( bd_points[i_] - cells_coords_mesh, axis=1))
                nearest_value = internalMesh[input_file][idx]
                if nbr_comp == 1:	
                    new_content += f'{nearest_value}\n' #f'{values_bd}\n' 
                else :
                    new_content += '('
                    for val in nearest_value : new_content += f'{val} '#f'{val} '
                    new_content += ')\n' 
            new_content +=  ");\n\t}\n"
    new_content +=  "}"

    # Write modified content to a new file
    with open(f'{path_to_case}/{time_value}/{output_file}', 'w') as fout:
        fout.writelines(new_content)
    
def make_symm_Jet_data_on_PIVsubdomain(FeaturesChoice, dic_data, models, whichJet):
    ''' 
    Post-treat jet case:\n 
    scale * 0.0508 and rotate by 90° on x-axis
    '''
      
    internalMeshRestricted = dic_data[whichJet]["Frozen"]["internalMesh"]
    #indices of points 0 ymax
    bounds = internalMeshRestricted.bounds
    ymin, ymax = bounds[2], bounds[3]
      
    # cell centers within the bounds (y_min + y_max)/2. and y_max
    cell_centers = internalMeshRestricted.cell_centers().points
    indices_upper_half = np.where((cell_centers[:, 1] >= (ymin + ymax)/2.) & (cell_centers[:, 1] <= ymax))[0]
    Upper_HalfRestricted_cell_centers = cell_centers[indices_upper_half,:]
    
    indices_lower_half = np.where((cell_centers[:, 1] <= (ymin + ymax)/2.) & (cell_centers[:, 1] >= ymin))[0]
    Lower_HalfRestricted_cell_centers = cell_centers[indices_lower_half,:]
    #
    dic_data.update({f"{whichJet}_projected":{}})
    #
    for model in models:
     
        mesh_Jet=dic_data[whichJet][model]["internalMesh"].rotate_x(-90, inplace=False)
        mesh_Jet.points *= 1./0.0508

        internalMesh = mesh_Jet
        # boundaries = dic_data[whichJet][model]["boundary"] #M
        cell_centers_Mesh = internalMesh.cell_centers().points
        
        
        distances = cdist(Upper_HalfRestricted_cell_centers, cell_centers_Mesh)

        if model=='Frozen' : upper_nearest_indices = indices_upper_half
        else : upper_nearest_indices = np.argmin(distances, axis=1)
        
        
        mesh_Jet = mesh_Jet.rotate_x(-180, inplace=False)
        internalMesh = mesh_Jet
        cell_centers_Mesh = internalMesh.cell_centers().points
        
        distances = cdist(Lower_HalfRestricted_cell_centers, cell_centers_Mesh)
        if model=='Frozen' : lower_nearest_indices = indices_lower_half
        else : lower_nearest_indices = np.argmin(distances, axis=1)
        
        # load new restricted mesh
        internalMesh_new, boundaries_new = load_data(f'{FeaturesChoice}/{whichJet}/Frozen')
        
        for key in set(internalMesh_new.array_names) :
            internalMesh_new[key] *= 0.
            internalMesh_new[key][indices_upper_half,] = internalMesh[key][upper_nearest_indices,]
            internalMesh_new[key][indices_lower_half,] = internalMesh[key][lower_nearest_indices,]
        
        dic_data[f'{whichJet}_projected'].update({model:{"internalMesh":internalMesh_new, "boundary": boundaries_new, "nCells":len(internalMesh_new.cell_centers().points)}})
        
def export_Jet_foam_files(FeaturesChoice, models, dic_data, whichJet, whichDoamin, latest_time_value):
    for model in models:
        path_to_case = f'{FeaturesChoice}/{whichJet}_{whichDoamin}/{model}'
        if not os.path.exists(path_to_case):os.makedirs(path_to_case)
        internalMesh_Jet = dic_data[f'{whichJet}_{whichDoamin}'][model]['internalMesh']
        boundaries_Jet = dic_data[f'{whichJet}_{whichDoamin}'][model]['boundary']
        
        for fieldname in set(internalMesh_Jet.array_names):
            write_foam_file_with_bd(path_to_case, fieldname, fieldname, latest_time_value, internalMesh_Jet, boundaries_Jet)

def restrict_Jet_data_to_upper_half_PIV_domain_bounds(FeaturesChoices, dic_data, models, whichJet):

    PIVmesh = dic_data[whichJet]["Frozen"]["internalMesh"].rotate_x(90, inplace=False)
    PIVmesh.points *= 0.0508
    #indices of points 0 zmax
    bounds = PIVmesh.bounds
    xmin, xmax, _, _, zmin, zmax = bounds

    cell_centers_PIV = PIVmesh.cell_centers().points
    # indices_upper_half = np.where((cell_centers_PIV[:, 2] >= (zmin + zmax)/2.) & (cell_centers_PIV[:, 2] <= zmax))[0]
    
    internalMesh_ref = dic_data[whichJet]['CHAN']["internalMesh"]
    cell_centers_Mesh_ref = internalMesh_ref.cell_centers().points
    indices_upper_half_phisical = np.where((cell_centers_Mesh_ref[:, 2] >= (zmin + zmax)/2.) & (cell_centers_Mesh_ref[:, 2] <= zmax) & (cell_centers_Mesh_ref[:, 0] >= xmin) & (cell_centers_Mesh_ref[:, 0] <= xmax))[0]
    HalfRestricted_cell_centers_ref = cell_centers_Mesh_ref[indices_upper_half_phisical,:]
    
    dic_data.update({f"{whichJet}_restricted":{}})
    
    for model in models:
        
        internalMesh = dic_data[whichJet][model]["internalMesh"]
        internalMesh_new, boundaries_new = load_data(f'{FeaturesChoices}/{whichJet}/CHAN')
        
        if model=='Frozen' : 			
            distances = cdist(HalfRestricted_cell_centers_ref, cell_centers_PIV)
            nearest_indices_model = np.argmin(distances, axis=1)
        else : 
            nearest_indices_model = indices_upper_half_phisical

        for key in set(internalMesh_new.array_names) :
            internalMesh_new[key] *= 0.
            internalMesh_new[key][indices_upper_half_phisical,] = internalMesh[key][nearest_indices_model,]
        
        
        dic_data[f'{whichJet}_restricted'].update({model:{"internalMesh":internalMesh_new, "boundary": boundaries_new, "nCells":len(internalMesh_new.cell_centers().points)}})

def augment_Jet_data_to_upper_half_PIV_domain_bounds(FeaturesChoices, dic_data, models, whichJet):

    PIVmesh = dic_data[whichJet]["Frozen"]["internalMesh"].rotate_x(90, inplace=False)
    PIVmesh.points *= 0.0508
    #indices of points 0 zmax
    bounds = PIVmesh.bounds
    xmin, xmax, _, _, zmin, zmax = bounds

    cell_centers_PIV = PIVmesh.cell_centers().points
    indices_upper_half = np.where((cell_centers_PIV[:, 2] >= (zmin + zmax)/2.) & (cell_centers_PIV[:, 2] <= zmax))[0]
     
    internalMesh_ref = dic_data[whichJet]['CHAN']["internalMesh"]
     
    cell_centers_Mesh_ref = internalMesh_ref.cell_centers().points
    indices_upper_half_phisical = np.where((cell_centers_Mesh_ref[:, 2] >= (zmin + zmax)/2.) & (cell_centers_Mesh_ref[:, 2] <= zmax) & (cell_centers_Mesh_ref[:, 0] >= xmin) & (cell_centers_Mesh_ref[:, 0] <= xmax))[0]
    HalfRestricted_cell_centers_ref = cell_centers_Mesh_ref[indices_upper_half_phisical,:]
     
    dic_data.update({f"{whichJet}_augmented":{}})
     
    for model in models:
        
        internalMesh = dic_data[whichJet][model]["internalMesh"]
          
        
        # load new restricted mesh (ANSJ in particular and fill it with
        if model=='Frozen' : 
            distances = cdist(HalfRestricted_cell_centers_ref, cell_centers_PIV)
            nearest_indices_model = np.argmin(distances, axis=1)
                 
            internalMesh_new, boundaries_new = load_data(f'{FeaturesChoices}/{whichJet}/ANSJ')
               
        else :
            nearest_indices_model = indices_upper_half_phisical
            internalMesh_new, boundaries_new = load_data(f'{FeaturesChoices}/{whichJet}/{model}')
            if model !='ANSJ':
                for key in ['U', 'bijDelta', 'ReskOmega'] :
                    internalMesh_new[key][:,] = 1e8
        
        
        for key in set(internalMesh_new.array_names) :
            internalMesh_new[key][indices_upper_half_phisical,] = internalMesh[key][nearest_indices_model,]
            
        dic_data[f'{whichJet}_augmented'].update({model:{"internalMesh":internalMesh_new, "boundary": boundaries_new, "nCells":len(internalMesh_new.cell_centers().points)}})



# import numpy					as np
# import matplotlib.pyplot		as plt
# import scipy					as sp 
# import pickle

# from scipy.spatial.distance import cdist


# from sklearn.metrics                    import mean_squared_error,\
#       mean_absolute_error
# from sklearn.neighbors                  import KNeighborsRegressor
# from sklearn.ensemble                   import RandomForestRegressor
# from sklearn.decomposition              import PCA
# from sklearn.cross_decomposition        import PLSRegression


# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, DotProduct, RationalQuadratic
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_absolute_error

# import os

# import scipy.ndimage
# import scipy.interpolate



# try:
#     from skimage.restoration import denoise_tv_chambolle
# except ImportError:
#     # skimage < 0.12
#     from skimage.filters import denoise_tv_chambolle
  
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas as pd
# from scipy.ndimage import gaussian_filter
# # plot params
# plt.rcParams.update({
#     "font.size": 25, "font.family": "serif", "mathtext.fontset": "cm",
#     "axes.grid": False, "grid.color": "1", "grid.linestyle": "-", "grid.linewidth": 0.9,
# })  
  
# def CG_block_II(A, rhs):
# 	norm_Rk = 1e5
# 	X0 = np.zeros(rhs.shape)
# 	Xk = X0
# 	Rk = rhs - A @ Xk 
# 	Pk = Rk
# 	Zero = np.zeros(X0.shape)
# 	iter_CG = 1
# 	while norm_Rk > 1e-6 and iter_CG < 20:
# 		MPk = A @ Pk
# 		normF_Rk_squared = np.trace(Rk.T@Rk)
# 		alphak = normF_Rk_squared / np.trace(Pk.T@MPk)
        
# 		Xk_new = Xk + alphak * Pk
# 		Rk_new = Rk - alphak * MPk
        
# 		betak = np.trace(Rk_new.T@Rk_new) / normF_Rk_squared
        
# 		Pk_new = Rk_new + betak * Pk
        
# 		Rk = Rk_new
# 		Pk = Pk_new
# 		Xk = Xk_new
        
# 		norm_Rk = np.sqrt(np.trace(Rk.T@Rk))
# 		iter_CG +=1
# #		print("CG iter : "+ repr(iter_CG) +" ==>   || Rk ||_F = ", norm_Rk)
        
    
# 	# Return the solution	
# 	X = Xk_new
# 	return X
    
# def mahalanobis_distance(x, cluster, cov_matrix):
# #	print("""Calculate the Mahalanobis distance between a vector and a cluster.""")
# 	# Calculate the covariance matrix of the cluster
# #	cov_matrix = np.cov(cluster, rowvar=False) 
# #	cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-5
    
# 	# Ensure the covariance matrix is symmetric and positive definite
# 	if not np.allclose(cov_matrix, cov_matrix.T) or np.any(np.linalg.eigvalsh(cov_matrix) <= 0):
# 		# If the covariance matrix is not positive definite, return None
# 		return None
    
# 	# Calculate the difference between x and each vector in the cluster
# 	diff = cluster - x
    
# 	# Solve the linear systems using conjugate gradient method
    
# 	residuals = CG_block_II(cov_matrix, diff.T)
# 	# Calculate the Mahalanobis distance for each vector in the cluster
# 	distances = np.sqrt(np.sum(residuals ** 2)) / residuals.size
    
# 	return distances
    
    

# def post_treat_weights_fluctuations(weights):
# 	for case in weights.keys():
# 		for i_ in range(len(weights[case])):
# 			if (abs(weights[case][i_,0]-1./3.)<1e-6 and abs(weights[case][i_,1]-1./3.)<1e-6 and abs(weights[case][i_,2]-1./3.)<1e-6):
# 				if case.startswith('Jet') : weights[case][i_,:] = [1,0,0]
# 				elif case.startswith('channel') : weights[case][i_,:] = [0,1,0]
# 				else : weights[case][i_,:] = [0,0,1]


# from scipy.stats import gaussian_kde 
# def compute_alpha(x,y):
# 	# Calculate KDE
# 	kde = gaussian_kde(np.vstack([x, y]))
# 	density = kde(np.vstack([x, y]))

# 	# Normalize density values to range [0, 1]
# 	density = (density - density.min()) / (density.max() - density.min())

# 	# Adjust alpha based on density
# 	alpha = 1 - density  # inverse relationship: higher density -> lower alpha
    
# 	return alpha


# def generate_random_color():
#     r = random.random()
#     g = random.random()
#     b = random.random()
#     return (r, g, b)
    
# def disk_radius_scattered_data(x, y):
# 	# Calculate the center of the data
# 	center_x, center_y = np.mean(x), np.mean(y)

# 	# Calculate the maximum distance from the center to any data point
# 	distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
# 	disk_radius = np.max(distances)
    
# 	return center_x, center_y, disk_radius

# def make_input_output_data(weights, features_list, train_indices, test_indices, x_input_train, x_input_test, y_output_train, y_output_test):
# #	y_output_train, y_output_test = weights[train_indices,:], weights[test_indices,:]
    
    
# 	y_output_train = np.vstack([weights[train_indices,:] for i_ in range(3)])
# 	y_output_test = np.vstack([weights[test_indices,:] for i_ in range(3)])
    
    
# 	features_ANSJ, features_CHAN, features_SEP = features_list
    
# 	x_input_train = np.vstack([features_ANSJ[train_indices,:], features_CHAN[train_indices,:], features_SEP[train_indices,:]])
# 	x_input_test = np.vstack([features_ANSJ[test_indices,:], features_CHAN[test_indices,:], features_SEP[test_indices,:]])
    

# from scipy import fftpack 
# import cv2  
# def smoothen_weights(sigma, C_coords, domain_bounds, weights):
# 	x_min, x_max, y_min, y_max, z_min, z_max = domain_bounds
# 	coords = C_coords
# 	x, y = coords[:,0], coords[:,1]
# 	# Create a uniform grid for interpolation
# 	grid_x, grid_y = np.mgrid[x_min:x_max:1000j, y_min:y_max:1000j]
# 	threshold, nearest_distances = calculate_threshold(x, y, percentile=90)  # You can adjust the percentile if needed
# #	nearest_distances_grid = scipy.interpolate.griddata((x, y), nearest_distances, (grid_x, grid_y), method='nearest')
# 	for j_ in range(3):
# 		weights_j = weights[:,j_]
# 		# Interpolate the non-uniform data onto the uniform grid
# 		grid_weight_j = scipy.interpolate.griddata((x, y), weights_j, (grid_x, grid_y), method='nearest')
# 		# Apply Gaussian filter to smooth the interpolated data
# 		smoothed_grid_weight_j = scipy.ndimage.gaussian_filter(grid_weight_j, sigma=sigma)
        
# 		#------------------------------------------------------------------------------------
# 		# Apply filter to smooth the interpolated data
# #		smoothed_grid_weight_j = denoise_tv_chambolle(grid_weight_j, weight=100)
        
# 		#------------------------------------------------------------------------------------
# #		smoothed_grid_weight_j = cv2.bilateralFilter(grid_weight_j.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
        
# 		#------------------------------------------------------------------------------------
# 		# Compute FFT
# #		grid_weight_j_fft = fftpack.fft2(grid_weight_j)
# #		# Example: Create a simple high-pass filter
# #		rows, cols = grid_weight_j.shape
# #		center_row, center_col = rows // 2, cols // 2
# #		radius = 30  # Adjust this radius based on the desired cutoff frequency

# #		mask = np.ones((rows, cols))
# #		mask[center_row - radius:center_row + radius, center_col - radius:center_col + radius] = 0
# #		grid_weight_j_fft_filtered = grid_weight_j_fft * mask
# #		smoothed_grid_weight_j = np.abs(fftpack.ifft2(grid_weight_j_fft_filtered))



# 		# Interpolate smoothed grid data back to the original coordinates
# 		smoothed_weight_j = scipy.interpolate.griddata( (grid_x.ravel(), grid_y.ravel()), smoothed_grid_weight_j.ravel(), (x, y), method='nearest')
        
        
# 		print('error smoothing % = ', 100 * np.linalg.norm(weights_j-smoothed_weight_j) / np.linalg.norm(weights_j) )
# 		weights[:,j_] = smoothed_weight_j[:]

# 	# sum weights to one
# 	weights /= np.sum(weights, axis = 1).reshape((-1,1))	
    

# from scipy.spatial import cKDTree
    
    
# def plot_weights_exact_predicted(path_save, C_coords, domain_bounds, weights_exact, weights_predicted, weightName):
#     x, y = C_coords[:, 0], C_coords[:, 1]
#     x_min, x_max, y_min, y_max, z_min, z_max = domain_bounds
#     grid_x, grid_y = np.mgrid[x_min:x_max:1000j, y_min:y_max:1000j]
    
#     # Calculate global min and max values
#     global_min_value = min(np.min(weights_exact), np.min(weights_predicted))
#     global_max_value = max(np.max(weights_exact), np.max(weights_predicted))
    
#     fig, axs = plt.subplots(3, 3, figsize=(20, 12), dpi=360)
#     threshold, nearest_distances = calculate_threshold(x, y, percentile=90)
#     models = ['ANSJ', 'kwSST', 'SEP']
    
#     for j_ in range(3):
#         # Interpolate the predicted weights with threshold
#         grid_weight_pred_j = interpolate_with_threshold(x, y, weights_predicted[:, j_], grid_x, grid_y, threshold)
#         # Interpolate the exact weights with threshold
#         grid_weights_exact_j = interpolate_with_threshold(x, y, weights_exact[:, j_], grid_x, grid_y, threshold)
        
#         percentage_error = 100. * np.linalg.norm(weights_predicted[:, j_] - weights_exact[:, j_]) / (np.linalg.norm(weights_exact[:, j_]) + 1e-12)
        
#         # Plot the exact weights
#         im1 = axs[0, j_].pcolormesh(grid_x, grid_y, grid_weights_exact_j, shading='auto', cmap='viridis', vmin=global_min_value, vmax=global_max_value)
#         fig.colorbar(im1, ax=axs[0, j_], label=r'$Exact~Weights$')
#         axs[0, j_].set_xlabel(r'$x$')
#         axs[0, j_].set_ylabel(r'$y$')
#         axs[0, j_].set_title(fr'$Exact~w${weightName}$~{models[j_]}$',  fontsize=20)
        
#         # Plot the predicted weights
#         im2 = axs[1, j_].pcolormesh(grid_x, grid_y, grid_weight_pred_j, shading='auto', cmap='viridis', vmin=global_min_value, vmax=global_max_value)
#         fig.colorbar(im2, ax=axs[1, j_], label=r'$Predicted~Weights$')
#         axs[1, j_].set_xlabel(r'$x$')
#         axs[1, j_].set_ylabel(r'$y$')
#         #axs[1, j_].set_title(fr'$Predicted~w${weightName}_${models[j_]} : error = {percentage_error:.1f}\%$',    fontsize=15)
#         axs[1, j_].set_title(fr'$Error~{models[j_]} : {percentage_error:.1f}\%$',    fontsize=20)

#         # Plot the predicted weights
#         im3 = axs[2, j_].pcolormesh(grid_x, grid_y, np.abs(grid_weight_pred_j-grid_weights_exact_j), shading='auto', cmap='viridis')
#         fig.colorbar(im3, ax=axs[2, j_], label=r'$Errors$')
#         axs[2, j_].set_xlabel(r'$x$' )
#         axs[2, j_].set_ylabel(r'$y$')
# 		#axs[2, j_].set_title(fr'$Error~w${weightName}_${models[j_]}$', fontsize=15)
#         axs[2, j_].set_title(fr'$Error~w${weightName}$~{models[j_]}$', fontsize=20)
        
#     plt.tight_layout()
    
#     plt.savefig(f'{path_save}_weights_exact_predicted_{weightName}.png')
#     plt.clf()
    
# #    plt.show()
    
    
# def predict_smoothed_weights_training_test_MLmodel(features, train_indices, test_indices, model_U, sigma_pred, C_coords, domain_bounds):	
# 	weightsU_train_predicted = model_U.predict(features[train_indices,:])
# 	weightsU_test_predicted = model_U.predict(features[test_indices,:])
    
# 	weightsU_predicted=np.zeros((len(weightsU_train_predicted)+len(weightsU_test_predicted),3))
# 	weightsU_predicted[train_indices,:] = weightsU_train_predicted
# 	weightsU_predicted[test_indices,:] = weightsU_test_predicted
    
# #	smoothen_weights(sigma_pred, C_coords, domain_bounds, weightsU_predicted)
    
# #	weightsU_train_predicted = weightsU_predicted[train_indices,:]
# #	weightsU_test_predicted = weightsU_predicted[test_indices,:]
            
# 	return weightsU_predicted #weightsU_train_predicted, weightsU_test_predicted
    
# def plot_histograms_training_test_MLmodel(path_save,
#     weightsU_train_predicted, weightsU_train_exact,
#     weightsU_test_predicted,  weightsU_test_exact,
#     weightName):

#     # Nom des modèles experts
#     expert_names = [r'$ANSJ$', r'$kwSST$', r'$SEP$']
#     colors = ['tab:blue', 'tab:orange', 'tab:green']

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

#     # -- TRAIN PLOT --
#     for k in range(3):
#         error_k_train = weightsU_train_predicted[:,k] - weightsU_train_exact[:,k]
#         ax1.hist(error_k_train,
#                  bins=40,
#                  alpha=0.55,
#                  color=colors[k],
#                  label=expert_names[k],
#                  edgecolor='black',
#                  linewidth=0.5)
        
#     ax1.set_title(r"$Training~set~error~distribution$", fontsize=20)
#     ax1.set_xlabel(r'$Error$', fontsize=25)
#     ax1.set_ylabel(r'$Frequency~[-]$', fontsize=25)
# 	#ax1.relim()
#     ax1.autoscale()
#     ax1.grid(alpha=0.3)
    

#     # -- TEST PLOT --
#     for k in range(3):
#         error_k_test = weightsU_test_predicted[:,k] - weightsU_test_exact[:,k]
#         ax2.hist(error_k_test,
#                  bins=40,
#                  alpha=0.55,
#                  color=colors[k],
#                  label=expert_names[k],
#                  edgecolor='black',
#                  linewidth=0.5)
        
#     ax2.set_title(r"$Test~set~error~distribution$", fontsize=20)
#     ax2.set_xlabel(r'$Error$', fontsize=25)
#     ax2.grid(alpha=0.3)
# 	#ax2.relim()
#     ax2.autoscale()

#     # Légende globale
#     ax1.legend(title=r"$Experts$", fontsize=20)

#     plt.tight_layout()
#     plt.savefig(f'{path_save}_error_histograms_{weightName}.png', dpi=300)
#     plt.close()


# #	plt.show()

# def plot_weights_predicted_vs_weights_exact(path_save, train_size, model_log_U_train, model_log_U_test, weightsU_train_predicted, weightsU_train_exact, weightsU_test_predicted, weightsU_test_exact, weightName):
    
    
# 	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12)) # figsize=(16, 8)
# 	for k in range(3):
# 		ax1.scatter(weightsU_train_exact[:,k], weightsU_train_predicted[:,k], alpha=0.5, marker='D')
# 		ax2.scatter(weightsU_test_exact[:,k], weightsU_test_predicted[:,k], alpha=0.5, marker='o')	
    
# 	ax2.set_xlim(ax1.get_xlim())
# 	ax2.set_ylim(ax1.get_ylim())

# 	# Add labels and title
# 	ax1.set_xlabel(fr'$Exact~trained,~train:{(100*train_size):.1f}%$')
# 	ax1.set_ylabel(r'$Predicted~trained~weights$')
# 	ax2.set_xlabel(fr'$Exact~trained,~test:{(100*(1-train_size)):.1f}$%')
# 	ax2.set_ylabel(r'$Predicted~trained~weights$')
    
# 	ax1.set_title(f'{model_log_U_train}')
# 	ax2.set_title(f'{model_log_U_test}')
    
# #			plt.title(f'case : {case_name}')
# 	plt.legend()
    
# 	plt.savefig(f'{path_save}_weights_predicted_vs_weights_exact_{weightName}.png')
# 	plt.clf()
    
# #	plt.show()

    
# """ Clustering data by k-means"""
# from sklearn.cluster import KMeans
# def reduce_array(array, nClusters):
#     # Fit K-means clustering
#     kmeans = KMeans(n_clusters=nClusters)
#     kmeans.fit(array)

#     # Identify representative rows from each cluster and store their indices
#     unique_labels = np.unique(kmeans.labels_)
#     reduced_rows = []
#     kept_indices = []
#     for label in unique_labels:
#         cluster_indices = np.where(kmeans.labels_ == label)[0]
#         representative_index = cluster_indices[0]  # Choose the first index as representative
# #        representative_index = random.choice(cluster_indices)
#         reduced_rows.append(array[representative_index])
#         kept_indices.append(representative_index)

# #   return np.array(reduced_rows), np.array(kept_indices)
#     return np.array(kept_indices)
    
    
# def clustering_data(array, nClusters):
#     # Fit K-means clustering
#     kmeans = KMeans(n_clusters=nClusters)
#     kmeans.fit(array)
#     min_len_cluster = 20
#     # Identify representative rows from each cluster and store their indices
#     unique_labels = np.unique(kmeans.labels_)
#     reduced_rows = []
#     kept_indices = []
#     list_cluster_indices =[]
#     for label in unique_labels:
#         cluster_indices = np.where(kmeans.labels_ == label)[0]
#         if len(cluster_indices) > min_len_cluster : list_cluster_indices.append(cluster_indices)
# #        cluster_indices = list_cluster_indices[-1]
#         representative_index = cluster_indices[0]  # Choose the first index as representative
# #        representative_index = random.choice(cluster_indices)
#         reduced_rows.append(array[representative_index])
#         kept_indices.append(representative_index)
    
# #   return np.array(reduced_rows), np.array(kept_indices)
#     return list_cluster_indices



# import random
# def cluster_data_indices(array, nClusters):
#     # Fit K-means clustering
#     kmeans = KMeans(n_clusters=nClusters)
#     kmeans.fit(array)
#     # Identify representative rows from each cluster and store their indices
#     unique_labels = np.unique(kmeans.labels_)
#     reduced_rows = []
#     kept_indices = []
#     for label in unique_labels:
#         cluster_indices = np.where(kmeans.labels_ == label)[0]
# #        cluster_indices = list_cluster_indices[-1]
# #        representative_index = cluster_indices[0]  # Choose the first index as representative
#         representative_index = random.choice(cluster_indices)
#         reduced_rows.append(array[representative_index])
#         kept_indices.append(representative_index)
    
#     return np.array(kept_indices) #np.array(reduced_rows), 









# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import numpy as np




# from sklearn.model_selection import cross_val_score

# def train_random_forest(X_train, X_test, Y_train, Y_test, rf_rms_ponderation_weights):
#     """Perform Random Forest regression to generate surface response"""
#     # Define the RandomForestRegressor with tuned parameters
#     rf = RandomForestRegressor(
#         n_estimators=100,
#         criterion='squared_error',
#         min_samples_leaf=1,
#         max_features=7,        
#         random_state=0
#     )
    
#     # Fit the Random Forest model
#     rf.fit(X_train, Y_train, sample_weight=rf_rms_ponderation_weights)
    
#     # Predictions for train and test sets
#     y_train_pred = rf.predict(X_train)
#     y_test_pred = rf.predict(X_test)
    
#     # Calculate R2 for the training set
#     r2_score_train = rf.score(X_train, Y_train)
    
#     # Perform 5-fold cross-validation on the entire dataset
# 	##reb : change cv en 5
#     q2_scores = cross_val_score(rf, X_train, Y_train, cv=5, scoring='r2')
#     q2_score_mean = q2_scores.mean()
    
#     # Log the results for training and test sets
#     model_log_train = f"R2 criterion train : {r2_score_train:.3f}"
#     model_log_test = f"Q2 (5-fold CV) criterion test : {q2_score_mean:.3f}"

#     return rf, model_log_train, model_log_test





# #def train_random_forest(X_train, X_test, Y_train, Y_test, rf_rms_ponderation_weights):
# #    """Perform Random Forest regression to generate surface response"""
# #    # Define the RandomForestRegressor with tuned parameters
# #    rf = RandomForestRegressor(
# #        n_estimators=100,           # Start with 100 and increase
# #        criterion='squared_error',  # Default for regression
# #        min_samples_leaf=1,         # Control leaf size
# #        max_features=7,        
# #        random_state=0              # For reproducibility
# #    )
# #    
# #    # Fit the Random Forest model
# #    rf.fit(X_train, Y_train, sample_weight=rf_rms_ponderation_weights)
# #    
# #    # Predictions for train and test sets
# #    y_train_pred = rf.predict(X_train)
# #    y_test_pred = rf.predict(X_test)
# #    
# #    # Calculate R2 for the training set
# #    r2_score_train = rf.score(X_train, Y_train)
# #    
# #    # Calculate Q2 for the test set
# #    tss_test = np.sum((Y_test - np.mean(Y_test)) ** 2)
# #    rss_test = mean_squared_error(Y_test, y_test_pred) * len(Y_test)
# #    q2_score_test = 1 - (rss_test / tss_test)
# #    
# #    # Log the results for training and test sets
# #    model_log_train = f"R2 criterion train : {r2_score_train:.3f}"
# #    model_log_test = f"Q2 criterion test : {q2_score_test:.3f}"

# #    return rf, model_log_train, model_log_test


    
    
        
# #print(boundaries['frontAndBack']['U_ANSJ'])
# # ------ Random Forest:{'friedman_mse', 'poisson', 'absolute_error', 'squared_error'}
# def train_random_forest_(X_train, X_test, Y_train, Y_test, rf_rms_ponderation_weights):
# 	"""Perform Random Forest regression to generate surface response"""
# 	# Define the RandomForestRegressor with tuned parameters
# 	rf = RandomForestRegressor(
# 		n_estimators=25,           # Start with 100 and increase
# 		criterion='squared_error',  # Default for regression
# #		max_depth=20,               # Control tree depth
# #		min_samples_split=5,        # Control splits
# 		min_samples_leaf=1,         # Control leaf size
# 		max_features=7,        
# #		bootstrap=True,             # Bootstrap sampling
# 		random_state=0            # For reproducibility
# 	)
    


# 	# Split data
# #	X_train, X_test, Y_train, Y_test = train_test_split(input_param, results_doe, test_size=0.2, shuffle=True)

# 	#
# 	rf.fit(X_train, Y_train, sample_weight=rf_rms_ponderation_weights)
# 	y_train_pred = rf.predict(X_train)
# 	y_test_pred = rf.predict(X_test)
    
# 	q2_score_train = 1 - mean_absolute_error(Y_train, y_train_pred)/np.mean(Y_train)
# 	r2_score_train = rf.score(X_train, Y_train)
    
# 	q2_score_test = 1 - mean_absolute_error(Y_test, y_test_pred)/np.mean(Y_test)
# 	r2_score_test = rf.score(X_test, Y_test)
    
# 	model_log_train = f"Q2 criterion train : {q2_score_train:.3f}" 
# 	model_log_train += f"\nR2 criterion train : {r2_score_train:.3f}"
    
# 	model_log_test = f"\nQ2 criterion test : {q2_score_test:.3f}"
# 	model_log_test += f"\nR2 criterion test : {r2_score_test:.3f}"
    
    
# #	print(model_log)
    
# #	# Fit model to training set:	
# #	rf.fit(X_train, Y_train)
# #	y_prediction = rf.predict(X_test)
# #	print("Q2 of Random Forest:", 1 - mean_absolute_error(Y_test, y_prediction)/np.mean(Y_test))
# #	print("R2 of Random Forest:", rf.score(X_test, Y_test))
# #	model_log = "Q2 of Random Forest : " + repr( 1 - mean_absolute_error(Y_test, y_prediction)/np.mean(Y_test))
# #	model_log += "\nR2 of Random Forest : "+ repr(rf.score(X_test, Y_test))
        
# 	return rf, model_log_train, model_log_test

# #from scipy.stats import randint
# #from sklearn.model_selection import RandomizedSearchCV, train_test_split
# #def train_random_forest(input_param, results_doe):
# #    """Perform Random Forest regression to generate surface response"""
# #    
# #    # Define the parameter grid for Randomized Search
# #    param_distributions = {
# #        'n_estimators': randint(100, 1000),
# #        'max_depth': randint(10, 50),
# #        'min_samples_split': randint(2, 10),
# #        'min_samples_leaf': randint(1, 10),
# #        'max_features': ['auto', 'sqrt', 'log2']
# #    }

# #    # Initialize the RandomForestRegressor with criterion as 'absolute_error'
# #    rf = RandomForestRegressor(
# #        criterion='absolute_error',  # Use absolute error for regression
# #        bootstrap=True,              # Bootstrap sampling
# #        random_state=100             # For reproducibility
# #    )

# #    # Initialize the RandomizedSearchCV with cross-validation
# #    random_search = RandomizedSearchCV(
# #        rf, 
# #        param_distributions=param_distributions,
# #        n_iter=100,  # Number of parameter settings that are sampled
# #        cv=5,        # 5-fold cross-validation
# #        verbose=2,   # Output progress
# #        n_jobs=-1,   # Use all available cores
# #        random_state=100
# #    )

# #    # Split the data
# #    X_train, X_test, Y_train, Y_test = train_test_split(input_param, results_doe, test_size=0.25, random_state=100)

# #    # Fit the model
# #    random_search.fit(X_train, Y_train)

# #    # Get the best estimator
# #    best_rf = random_search.best_estimator_

# #    # Predict using the test set
# #    y_prediction = best_rf.predict(X_test)

# #    # Calculate Q2 and R2
# #    q2 = 1 - mean_absolute_error(Y_test, y_prediction) / np.mean(Y_test)
# #    r2 = best_rf.score(X_test, Y_test)

# #    print("Best Parameters:", random_search.best_params_)
# #    print("Q2 of Random Forest:", q2)
# #    print("R2 of Random Forest:", r2)

# #    model_log = "Best Parameters: " + str(random_search.best_params_)
# #    model_log += "\nQ2 of Random Forest: " + str(q2)
# #    model_log += "\nR2 of Random Forest: " + str(r2)
# #    
# #    return best_rf, model_log_ReskOmega
    
    

# import warnings
# from sklearn.exceptions import ConvergenceWarning
# import logging

# # Suppress all warnings from the specific modules
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=ConvergenceWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Additionally, configure logging to suppress warnings
# logging.getLogger('sklearn').setLevel(logging.ERROR)

# def train_gaussian_process_best(X_train, Y_train, nClusters):

#     print ("clustering ...")
#     kept_indices_clusters = reduce_array(np.hstack([X_train, Y_train]), nClusters)
# #    kept_indices_clusters = reduce_array(Y_train, nClusters)
    
#     X, y = X_train[kept_indices_clusters, :], Y_train[kept_indices_clusters, :]

#     # Define the extended parameter grid
#     param_grid = {
#         'kernel': [
#             RBF(length_scale=ls) + WhiteKernel(noise_level=nl) for ls in [0.1, 0.5, 1.0] for nl in [0.001, 0.01]
#         ] + [
#             RationalQuadratic(length_scale=ls, alpha=alpha) + WhiteKernel(noise_level=nl) for ls in [0.1, 0.5, 1.0] for alpha in [0.01, 0.1, 1.0] for nl in [0.001, 0.01]
#         ]
#     }

#     # Define the GPR model
#     gpr = GaussianProcessRegressor(random_state=0)

#     grid_search = GridSearchCV(estimator=gpr, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
#     grid_search.fit(X, y)


#     # Get the best model
#     best_gpr = grid_search.best_estimator_

#     y_prediction = best_gpr.predict(X)
#     print("Q2 of GPR training: ", 1 - mean_absolute_error(y, y_prediction) / np.mean(y))
#     print("Score of GPR training: ", best_gpr.score(X, y))

#     # Test GPR on the whole set of features data
#     y_all_prediction = best_gpr.predict(X_train)
#     print("Q2 of GPR test: ", 1. - mean_absolute_error(Y_train, y_all_prediction) / np.mean(Y_train))
#     print("Score of GPR test: ", best_gpr.score(X_train, Y_train))

#     print("Best parameters found: ", grid_search.best_params_)
    
#     model_log = "Q2 of GPR training: " + repr(1 - mean_absolute_error(y, y_prediction) / np.mean(y))
#     model_log += "\nScore of GPR training: " + repr(best_gpr.score(X, y))
#     model_log += "\nQ2 of GPR test: " + repr( 1. - mean_absolute_error(Y_train, y_all_prediction) / np.mean(Y_train))
#     model_log += "\nScore of GPR test: " + repr(best_gpr.score(X_train, Y_train))
#     model_log += "\nBest parameters found: " + str(grid_search.best_params_)
    
#     return best_gpr, model_log

# def train_gaussian_process(X_train, Y_train, nClusters):
    
# 	print ("clustering ...")
# #	kept_indices_clusters = reduce_array(np.hstack([features_concat, weightsU_concat]), nClusters)
# 	kept_indices_clusters = reduce_array(Y_train, nClusters)
            
# 	X, y = X_train[kept_indices_clusters,:], Y_train[kept_indices_clusters,:]


# 	klen = 0.01

# 	kernel = RBF(length_scale=0.1) + WhiteKernel(noise_level=0.001)
# 	#kernel = Matern(length_scale=klen, nu=1.5)
# 	#kernel = DotProduct(sigma_0=1.0)
# 	#kernel = RationalQuadratic(length_scale=0.005, alpha=0.01) 
# 	gpr = GaussianProcessRegressor(kernel=kernel ,random_state=0).fit(X, y)



# 	y_prediction = gpr.predict(X)
# 	print("Q2 of GPR training : ", 1 - mean_absolute_error(y, y_prediction)/np.mean(y))
# 	print("score of GPR training : ", gpr.score(X, y))
# 	# Test GPR on the whole set of features data
# 	y_all_prediction = gpr.predict(X_train)
# 	print("Q2 of GPR +test :", 1. - mean_absolute_error(Y_train, y_all_prediction)/np.mean(Y_train))
# 	print("score of GPR +test :", gpr.score(X_train, Y_train))
    
# 	return gpr


# def restrict_Jet_data_to_upper_half_PIV_domain_bounds(FeaturesChoices, dic_data, models, whichJet):
# 	# Post-treat jet case : 
# 	# scale * 0.0508  && rotate by 90° on x-axis
# #	dic_data[whichJet]["Frozen"]["internalMesh"] = dic_data[whichJet]["Frozen"]["internalMesh"].scale([0.0508, 0.0508, 0.0508], inplace=False)
# #	dic_data[whichJet]["Frozen"]["internalMesh"].points *= 0.0508
# #	dic_data[whichJet]["Frozen"]["internalMesh"] = dic_data[whichJet]["Frozen"]["internalMesh"].rotate_x(90, inplace=False)
# #	dic_data[whichJet]["Frozen"]["internalMesh"] = dic_data[whichJet]["Frozen"]["internalMesh"].translate([0., -0.06, 0.], inplace=False)
# 	#
# 	#
# 	PIVmesh = dic_data[whichJet]["Frozen"]["internalMesh"].rotate_x(90, inplace=False)
# 	PIVmesh.points *= 0.0508
# 	#indices of points 0 zmax
# 	bounds = PIVmesh.bounds
# 	xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
# #	print(bounds), input()
# #	p = pv.Plotter()
# #	p.add_mesh(PIVmesh, scalars='eta_1')
# #	p.show_bounds(grid='front', location='outer', all_edges=True, ticks='both', xlabel='X Axis', ylabel='Y Axis', zlabel='Z Axis')
# #	p.show_axes()
# #	p.show()

# 	#
# 	# cell centers within the bounds (z_min + z_max)/2. and z_max
# 	cell_centers_PIV = PIVmesh.cell_centers().points
# 	indices_upper_half = np.where((cell_centers_PIV[:, 2] >= (zmin + zmax)/2.) & (cell_centers_PIV[:, 2] <= zmax))[0]
# 	HalfRestricted_cell_centers = cell_centers_PIV[indices_upper_half,:]
# 	#
# 	internalMesh_ref = dic_data[whichJet]['CHAN']["internalMesh"]
# 	boundaries_ref = dic_data[whichJet]['CHAN']["boundary"]
# 	cell_centers_Mesh_ref = internalMesh_ref.cell_centers().points
# 	indices_upper_half_phisical = np.where((cell_centers_Mesh_ref[:, 2] >= (zmin + zmax)/2.) & (cell_centers_Mesh_ref[:, 2] <= zmax) & (cell_centers_Mesh_ref[:, 0] >= xmin) & (cell_centers_Mesh_ref[:, 0] <= xmax))[0]
# 	HalfRestricted_cell_centers_ref = cell_centers_Mesh_ref[indices_upper_half_phisical,:]
    
# 	#
# 	dic_data.update({f"{whichJet}_restricted":{}})
# 	#
# 	for model in models:
# 		# create projected data
# 		internalMesh = dic_data[whichJet][model]["internalMesh"]
# 		boundaries = dic_data[whichJet][model]["boundary"]
# 		cell_centers_Mesh = internalMesh.cell_centers().points
        
        
        
        
# 		# load new restricted mesh
# 		internalMesh_new, boundaries_new = load_data(f'{FeaturesChoices}/{whichJet}/CHAN')
        
        
# 		if model=='Frozen' : 			
# 			distances = cdist(HalfRestricted_cell_centers_ref, cell_centers_PIV)
# 			nearest_indices_model = np.argmin(distances, axis=1)
# 		else : 
# 			nearest_indices_model = indices_upper_half_phisical
        
        
        
# 		for key in set(internalMesh_new.array_names) :
# 			internalMesh_new[key] *= 0.
# 			internalMesh_new[key][indices_upper_half_phisical,] = internalMesh[key][nearest_indices_model,]
        
        
# 		dic_data[f'{whichJet}_restricted'].update({model:{"internalMesh":internalMesh_new, "boundary": boundaries_new, "nCells":len(internalMesh_new.cell_centers().points)}})
        
        
# #		dic_data[f'{whichJet}_restricted'].update({model:{"nCells":len(internalMesh_new.cell_centers().points)}})
        
        
# #		internalMesh['U_mag'] = np.linalg.norm(internalMesh['U'], axis = 1)	
# #		internalMesh_new['U_mag'] = np.linalg.norm(internalMesh_new['U'], axis = 1)	
# #		p = pv.Plotter()
# #		p.add_mesh(internalMesh_new, scalars='U_mag')
# ##		p.add_mesh(internalMesh, scalars='U_mag')
# #		# Add a bounding box
# #		p.show_bounds(grid='front', location='outer', all_edges=True, ticks='both', xlabel='X Axis', ylabel='Y Axis', zlabel='Z Axis')

# #		p.show_axes()
# #		p.show()



# def export_Jet_foam_files(FeaturesChoice, models, dic_data, whichJet, whichDoamin, latest_time_value):
# 	for model in models:
# 		path_to_case = f'{FeaturesChoice}/{whichJet}_{whichDoamin}/{model}'
# 		if not os.path.exists(path_to_case):os.makedirs(path_to_case)
# 		internalMesh_Jet = dic_data[f'{whichJet}_{whichDoamin}'][model]['internalMesh']
# 		boundaries_Jet = dic_data[f'{whichJet}_{whichDoamin}'][model]['boundary']
        
# 		for fieldname in set(internalMesh_Jet.array_names):
# #			write_foam_file(path_to_case, fieldname, fieldname, latest_time_value, internalMesh_Jet[fieldname])
# 			write_foam_file_with_bd(path_to_case, fieldname, fieldname, latest_time_value, internalMesh_Jet, boundaries_Jet)


# def make_symm_Jet_data_on_PIVsubdomain(FeaturesChoice, dic_data, models, whichJet):
# 	# Post-treat jet case : 
# 	# scale * 0.0508  && rotate by 90° on x-axis
# 	#
# 	internalMeshRestricted = dic_data[whichJet]["Frozen"]["internalMesh"]
# 	#indices of points 0 ymax
# 	bounds = internalMeshRestricted.bounds
# 	ymin, ymax = bounds[2], bounds[3]
# 	#
# 	# cell centers within the bounds (y_min + y_max)/2. and y_max
# 	cell_centers = internalMeshRestricted.cell_centers().points
# 	indices_upper_half = np.where((cell_centers[:, 1] >= (ymin + ymax)/2.) & (cell_centers[:, 1] <= ymax))[0]
# 	Upper_HalfRestricted_cell_centers = cell_centers[indices_upper_half,:]
    
# 	indices_lower_half = np.where((cell_centers[:, 1] <= (ymin + ymax)/2.) & (cell_centers[:, 1] >= ymin))[0]
# 	Lower_HalfRestricted_cell_centers = cell_centers[indices_lower_half,:]
# 	#
# 	dic_data.update({f"{whichJet}_projected":{}})
# 	#
# 	for model in models:
# 		# don't do this for the Frozen
# #		if model!='Frozen' : 
# 		# create projected data
# #			dic_data[whichJet][model]["internalMesh"] = dic_data[whichJet][model]["internalMesh"].scale([1./0.0508, 1./0.0508, 1./0.0508], inplace=False)
        
# 		mesh_Jet=dic_data[whichJet][model]["internalMesh"].rotate_x(-90, inplace=False)
# 		mesh_Jet.points *= 1./0.0508
        
# #		dic_data[whichJet][model]["internalMesh"].points *= 1./0.0508
# #		dic_data[whichJet][model]["internalMesh"] = dic_data[whichJet][model]["internalMesh"].rotate_x(-90, inplace=False)
# #		dic_data[whichJet][model]["internalMesh"] = dic_data[whichJet][model]["internalMesh"].translate([0., 0., 0.06], inplace=False)
        
        
# 		internalMesh = mesh_Jet
# 		boundaries = dic_data[whichJet][model]["boundary"]
# 		cell_centers_Mesh = internalMesh.cell_centers().points
        
        
# 		distances = cdist(Upper_HalfRestricted_cell_centers, cell_centers_Mesh)
# 		if model=='Frozen' : upper_nearest_indices = indices_upper_half
# 		else : upper_nearest_indices = np.argmin(distances, axis=1)
        
        
# 		mesh_Jet = mesh_Jet.rotate_x(-180, inplace=False)
# 		internalMesh = mesh_Jet
# 		boundaries = dic_data[whichJet][model]["boundary"]
# 		cell_centers_Mesh = internalMesh.cell_centers().points
        
# 		distances = cdist(Lower_HalfRestricted_cell_centers, cell_centers_Mesh)
# 		if model=='Frozen' : lower_nearest_indices = indices_lower_half
# 		else : lower_nearest_indices = np.argmin(distances, axis=1)
        
        
# 		# load new restricted mesh
# 		internalMesh_new, boundaries_new = load_data(f'{FeaturesChoice}/{whichJet}/Frozen')
        
# 		for key in set(internalMesh_new.array_names) :
# 			internalMesh_new[key] *= 0.
# 			internalMesh_new[key][indices_upper_half,] = internalMesh[key][upper_nearest_indices,]
# 			internalMesh_new[key][indices_lower_half,] = internalMesh[key][lower_nearest_indices,]
        
# 		dic_data[f'{whichJet}_projected'].update({model:{"internalMesh":internalMesh_new, "boundary": boundaries_new, "nCells":len(internalMesh_new.cell_centers().points)}})
        
# #		dic_data[f'{whichJet}_projected'].update({model:{"nCells":len(internalMesh_new.cell_centers().points)}})
            
# #		internalMesh['U_mag'] = np.linalg.norm(internalMesh['U'], axis = 1)
# #		internalMesh_new['U_mag'] = np.linalg.norm(internalMesh_new['U'], axis = 1)
# #		
# #		internalMesh['bijDelta_mag'] = np.linalg.norm(internalMesh['bijDelta'], axis = 1)
# #		internalMesh_new['bijDelta_mag'] = np.linalg.norm(internalMesh_new['bijDelta'], axis = 1)
# #		
# #		p = pv.Plotter()
# #		p.show_grid()
# #		p.add_mesh(internalMesh_new, scalars='U_mag')
# #		p.add_mesh(internalMesh, scalars='U_mag')
# ##		p.add_mesh(internalMesh_new, scalars='bijDelta_mag')
# #		p.show_axes()
# #		p.show()



    
# def write_foam_file(path_to_case, input_file, output_file, time_value, values_table):
# 	# Read content of the original file
# 	with open(f"{path_to_case}/{time_value}/{input_file}", 'r') as fin:
# 		content = fin.readlines()

# 	# Determine the size of the array
# 	array_size = len(values_table)
# 	collect_lines = False
# 	new_content = ''
# 	# Perform modifications on the internal field values
# 	for i, line in enumerate(content):
        
        
# 		if line.startswith('    location'):
# 			new_content += f'    location    \"{time_value}\";\n'
# 		elif line.startswith('    object'):
# 			new_content += f'    object    {output_file};\n'
# 		elif line.startswith('internalField   nonuniform List<scalar>'):
# 			new_content += f'internalField   nonuniform List<scalar>\n{array_size}\n(\n'
# 			for value in list(values_table):
# 				new_content += f'{value}\n' 
# 			new_content +=  ');\n\n'
# 			break
# 		elif line.startswith('internalField   nonuniform List<vector>'):
# 			new_content += f'internalField   nonuniform List<vector>\n{array_size}\n(\n'
# 			for values in values_table:
# 				v1, v2, v3 = values
# 				new_content += f'({v1} {v2} {v3})\n' 
# 			new_content +=  ');\n\n'
# 			break
# 		elif line.startswith('internalField   nonuniform List<symmTensor>'):
# 			new_content += f'internalField   nonuniform List<symmTensor>\n{array_size}\n(\n'
# 			for values in values_table:
# 				v1, v2, v3, v4, v5, v6 = values
# 				new_content += f'({v1} {v2} {v3} {v4} {v5} {v6})\n' 
# 			new_content +=  ');\n\n'
# 			break
        
# 		else :
# 			new_content += line	
    
# 	# collect the boundary conditions from the original file 
    
# 	for i, line in enumerate(content):	
        
# 		# Check if the current line matches the starting sentence
# 		if line.strip() == 'boundaryField':
# 			collect_lines = True
        
# 		# If collect_lines is True, collect the line
# 		if collect_lines:
# 			new_content += line

# 	# Write modified content to a new file
# 	with open(f'{path_to_case}/{time_value}/{output_file}', 'w') as fout:
# 		fout.writelines(new_content)	

# # ------- Exact waits: mixture of experts
# def mixture_of_expert(dic_data, sigma):
# 	weights = {}
# 	features = {}
# 	weightsU ={}
# 	for case in dic_data:
# 		# condition set to take into account the projected data Jet_NearSonic_proj
# 		if case != 'Jet_NearSonic'and case != 'Jet_subSonic': 
# 			# internal mesh 
# 			#U_HF = dic_data[case]['Frozen']['internalMesh']['U'][:,0:1].reshape((-1,1))
# 			U_HF = dic_data[case]['Frozen']['internalMesh']['U'][:,0:2]
# 			weightsU_case, list_U_models = [], []
# 			for model in dic_data[case]:
# 				if model != 'Frozen':
# 					print(case, model)
# 					#U_model = dic_data[case][model]['internalMesh']['U'][:,0:1].reshape((-1,1))
# 					U_model = dic_data[case][model]['internalMesh']['U'][:,0:2]
# 				    #
# 					if case == "LRN_OGV_trans" or case == "LRN_OGV_trans_OP1" :
# 						U_HF = (U_HF - U_HF.mean(axis=0)) / U_HF.std(axis=0)
# 						U_model = (U_model - U_model.mean(axis=0)) / U_model.std(axis=0)

# 					weightsU_case.append( 1e-12 + np.exp(-0.5*np.sum( (U_HF - U_model)**2. , axis=1)/sigma**2) )
# 					#
# 					list_U_models.append(U_model)
# 					#
# 			        #
# 			weightsU.update({case : np.hstack([ np.array( w_ / sum(weightsU_case)).reshape((-1,1)) for w_ in weightsU_case]) })
# 			features.update({case : np.hstack([np.array(dic_data[case][model]['internalMesh'][f'eta_{k_}']).reshape((-1,1)) for k_ in range(1,12)]) })
# 			#
# 			#
# 			#
# 			ErrorUV = np.zeros(U_HF.shape); ErrorUV[:,:] = U_HF[:,:]
# 			#for j_ in range(3):
# 			#	for k_ in range(U_HF.shape[1]) : ErrorU[:,k_] -= weightsU_case[j_] * list_U_models[j_][:,k_]
# 			#
# 			#
# 			#norm_ErrorU = np.linalg.norm(ErrorU)
# 			#print(f'|| U_models * W - Uref || = {norm_ErrorU}') 
# 			n_models = len(weightsU_case)   # au lieu de range(3) en dur
# 			for j_ in range(n_models):
# 				# weightsU_case[j_] : shape (N,)
#                 # list_U_models[j_] : shape (N,2)
# 				ErrorUV -= weightsU_case[j_].reshape(-1, 1) * list_U_models[j_]
#                 # broadcasting → (N,1)*(N,2) = (N,2)
# 			norm_ErrorUV = np.linalg.norm(ErrorUV)
# 			print(f'|| U,V_models * W - (U,V)_ref || = {norm_ErrorUV}')
# 			# boundaries
# 			# ...		
    
    
# 	return weightsU, features






# "---------------------------------------------------------------------------------------------------------------------"
# "--------------------------------------- optimized weights on sigma  ------------------------------------------------"
# "---------------------------------------------------------------------------------------------------------------------"

# #e_i**2*w_i/(sigma**3*sum_w) - w_i*Sum(e_k**2*w_k/sigma**3, (k, 1, n))/sum_w**2
# from sympy import symbols, exp, Sum
# #
# def w_derivatives_wrt_sigma(sigma, x_HF, x_list):
# 	e_list, w_list_aux, w_k_e_k_power2_list, w_k_e_k_power4_list = [], [], [], []
# 	for x_k in x_list :
# 		e_k = np.linalg.norm(x_HF - x_k, axis=1) 
# 		e_list.append(e_k)
# 		w_k =np.exp(- e_k**2. / (2.*sigma**2.) )  + 1e-8
# 		w_list_aux.append(w_k)
# 		w_k_e_k_power2 = w_k * e_k**2.
# 		w_k_e_k_power2_list.append(w_k_e_k_power2)
# 		w_k_e_k_power4 = w_k * e_k**4.
# 		w_k_e_k_power4_list.append(w_k_e_k_power4)
    
# 	sum_w = sum(w_list_aux)
# 	sum_w_e2 = sum(w_k_e_k_power2)
# 	sum_w_e4 = sum(w_k_e_k_power4)
    
# 	regCst = 1e-2
# 	dw_list, d2w_list = [], []
# 	for i_, w_i in enumerate(w_list_aux) :
# 		e_i = e_list[i_]
# 		dw_i = w_i / (sigma**3. * sum_w**2.) * ( e_i**2. * sum_w - sum_w_e2 ) + regCst * sigma
        
        
# 		d2w_i = - 3. * w_i * e_i**2. / (sigma**4. * sum_w) \
# 				+ w_i * e_i**4. / (sigma**6. * sum_w) \
# 				- 2. * w_i * e_i**2. * sum_w_e2 / (sigma**6. * sum_w**2.) \
# 				+ 2. * w_i * sum_w_e2**2. / (sigma**6. * sum_w**3.)\
# 				- w_i / sum_w**2. * ( - 3.* sum_w_e2/sigma**4. + sum_w_e4/sigma**6.)\
# 				+ regCst
# 		dw_list.append(dw_i)
# 		d2w_list.append(d2w_i)
    
    
# 	w_list = [w_/sum_w for w_ in w_list_aux]
    
# 	return w_list, dw_list, d2w_list


# def J_derivatives_wrt_sigma(sigma, x_HF, x_list):
# 	w_list, dw_list, d2w_list = w_derivatives_wrt_sigma(sigma, x_HF, x_list)
    
# 	sum_w_x = sum([ np.column_stack([w_list[i_]]*x_list[i_].shape[1]) * x_list[i_] for i_ in range(len(w_list))])
# 	sum_dw_x = sum([ np.column_stack([dw_list[i_]]*x_list[i_].shape[1]) * x_list[i_] for i_ in range(len(w_list))])
# 	sum_d2w_x = sum([ np.column_stack([d2w_list[i_]]*x_list[i_].shape[1]) * x_list[i_] for i_ in range(len(w_list))])
    
# 	J = np.trace( (sum_w_x - x_HF).T @ (sum_w_x - x_HF) )
    
# 	dJ = np.trace( (sum_w_x - x_HF).T @ sum_dw_x )
    
# 	d2J = np.trace( sum_dw_x.T @ sum_dw_x + (sum_w_x - x_HF).T @ sum_d2w_x )
    
# 	return w_list, J, dJ, d2J



    
# #def Gauss_weight(sigma, x_list, x_HF):
# #    e_list, w_list = [], []
# #	for x_k in x_list :
# #		e_k = np.linalg.norm(x_HF - x_k, axis=1)
# #		e_list.append(e_k)
# #		w_k =np.exp(- e_k**2. / (2.*sigma**2.) )
# #		w_list.append(w_k)
# #	
# #    return np.exp(- np.sum( (x_HF - x_list[i_])**2. , axis=1) / (2.*sigma**2.) ) 
# #    


# "---------------------------------------------------------------------------------------------------------------------"
# "---------------------------------------------------------------------------------------------------------------------"

# def weights_with_optimal_sigma(x_HF, x_list):
    
# #	res = minimize(Gauss_weight, x0, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
    
# 	# Newton raphson
# 	sigma = 0.01
# 	dJ = 1.
# 	ite_ = 1
# 	J_old = 1e5
# 	errJ = 1. 
# 	while abs(dJ) > 1e-8 and ite_ < 3000 :#and errJ > 1e-15 :
# 		w_list, J, dJ, d2J = J_derivatives_wrt_sigma(sigma, x_HF, x_list)
# #		if dJ > 1e-10 and d2J > 1e-10 : 
# 		if abs(d2J) > 1e-10 : sigma -= dJ / d2J
# 		else : 
# 			break;
# #		else : break; # to avoid sigma = nan
# 		errJ = abs(J-J_old)/abs(J)
# 		J_old = J
# #		print(f'iter = {ite_} \t sigma = {sigma} \t J(.) = {J}\t |dJ| = ', abs(dJ))#, input() #\t w_list = {w_list} 
# 		ite_ +=1
    
# #	print(f'ite_ = {ite_} \t sigma = {sigma} \t J(.) = {J}\t |dJ| = ', abs(dJ)), input()
# 	return w_list
    


    
    
    
# # ------- Exact waits: mixture of experts
# def mixture_of_expert_sigOptimal(dic_data, sigma):
# 	weights = {}
# 	features = {}
# 	weightsU ={}
# 	for case in dic_data:
# 		# condition set to take into account the projected data Jet_NearSonic_proj
# 		if case != 'Jet_NearSonic'and case != 'Jet_subSonic': 
# 			features.update({case : {}})
# 			# internal mesh 
# 			U_HF = dic_data[case]['Frozen']['internalMesh']['U'][:,0].reshape((-1,1))
# 			weightsU_case = []
# 			list_U_models = []
# 			for model in dic_data[case]:
# 				if model != 'Frozen':
# 					print(case, model)
# 					U_model = dic_data[case][model]['internalMesh']['U'][:,0].reshape((-1,1))		
# 					list_U_models.append(U_model)			
# 					features[case].update({model : np.hstack([np.array(dic_data[case][model]['internalMesh'][f'eta_{k_}']).reshape((-1,1)) for k_ in range(1,12)]) }) 			
# 			weightsU_case = weights_with_optimal_sigma(U_HF, list_U_models)
# 			wU_array = np.hstack([ np.array( w_ / sum(weightsU_case)).reshape((-1,1)) for w_ in weightsU_case])
# 			weightsU.update({case : wU_array })	
# #			features_array = np.hstack([np.array(dic_data[case][model]['internalMesh'][f'eta_{k_}']).reshape((-1,1)) for k_ in range(1,12)])
# #			features.update({case : features_array })
            
            
# 			#
# 			ErrorU = np.zeros(U_HF.shape); ErrorU[:,:] = U_HF[:,:]
# 			for j_ in range(3):
# 				for k_ in range(U_HF.shape[1]) : ErrorU[:,k_] -= weightsU_case[j_] * list_U_models[j_][:,k_]
# 			norm_ErrorU = np.linalg.norm(ErrorU)
# 			print(f'|| U_models * W - Uref || = {norm_ErrorU}') 
# 			# boundaries
# 			# ...		
            
# 			#Export exact weights
# #			np.savetxt(f'{case}/wU.txt', wU_array)
            
            
# 	return weightsU, features


# "---------------------------------------------------------------------------------------------------------------------"
# "--------------------------------------- no constraint  ------------------------------------------------"
# "---------------------------------------------------------------------------------------------------------------------"

# #	
# #def cacluate_Exact_weights(QoI_HF, list_QoI_models, weightsU_Gauss):
# #	gamma_ = 1e-8
# #	nbr_cells, nbr_components = QoI_HF.shape
# #	nbr_weights = len(list_QoI_models)
# #	Weights = np.zeros((nbr_cells,nbr_weights))
# #	Id_ = np.eye(nbr_weights)
# #	ones = np.zeros((nbr_components, 1))+1.
# #	A_ = np.zeros((nbr_components, nbr_weights))
# #	b_ = np.zeros((nbr_components, 1))
# #	ATA_ = np.zeros((nbr_weights, nbr_weights))
# #	ATb_ = np.zeros((nbr_weights, 1))
# #	Hess = np.zeros((nbr_weights + 1 , nbr_weights + 1 ))
# #	GradJ = np.zeros((nbr_weights + 1 , 1))
# #	xi_ = np.zeros((nbr_weights, 1))
# #	W_ = np.zeros((nbr_weights, 1))
# #	for i_ in range(nbr_cells):
# #		for k_ in range(nbr_weights) : A_[:,k_] = list_QoI_models[k_][i_,:]
# #		b_[:,0] = QoI_HF[i_,:]
# #		ATA_[:,:] = A_.T @ A_ + gamma_ * Id_
# #		ATb_[:,0] = (A_.T @ b_)[:,0] 
# #		# solve optimization problem for each cell : Newton-Raphson
# #		W_[:,0] = np.linalg.solve(ATA_, ATb_)[:,0]
# #		Weights[i_,:] = W_[:,0]
# #		
# #		sum_sqr_weights = sum(W_)-1.
# #		WW = W_[:,0]
# #		J_value = np.linalg.norm(A_@ W_ - b_ )
# ##		print(f'J(.) = {J_value} \t weights = {WW} \t sum w^2 = {sum_sqr_weights}')
# ##		input()
# #		
# #	return Weights
# #	
    
    
    
# "---------------------------------------------------------------------------------------------------------------------"
# "--------------------------------------- constraint sum of weights = 1 ------------------------------------------------"
# "---------------------------------------------------------------------------------------------------------------------"
# #def update_Hessian(Hess, ATA_, Id_, W_, xi_, s_):
# #	nbr_weights = len(ATA_)
# #	Hess[:nbr_weights, :nbr_weights] = ATA_
# #	Hess[:nbr_weights, nbr_weights] = 1.
# #	Hess[nbr_weights, :nbr_weights] = 1.
# ##	np.savetxt("Hess.txt", Hess), input()
# #	
# #	
# #def update_GradJ(GradJ, ATA_, ATb_, W_, ones, q_, xi_, s_):
# #	nbr_weights = len(W_)
# #	GradJ[:nbr_weights, 0] =  ATb_[:,0]
# #	GradJ[nbr_weights, 0] = 1.

# #	
# #def cacluate_Exact_weights(QoI_HF, list_QoI_models, weightsU_Gauss):
# #	gamma_ = 1e-7
# #	nbr_cells, nbr_components = QoI_HF.shape
# #	nbr_weights = len(list_QoI_models)
# #	Weights = np.zeros((nbr_cells,nbr_weights))
# #	Id_ = np.eye(nbr_weights)
# #	zerr = np.zeros((nbr_components, 1))
# #	ones = np.zeros((nbr_components, 1))+1.
# #	A_ = np.zeros((nbr_components, nbr_weights))
# #	b_ = np.zeros((nbr_components, 1))
# #	ATA_ = np.zeros((nbr_weights, nbr_weights))
# #	ATb_ = np.zeros((nbr_weights, 1))
# #	Hess = np.zeros((nbr_weights + 1 , nbr_weights + 1 ))
# #	GradJ = np.zeros((nbr_weights + 1 , 1))
# #	xi_ = np.zeros((nbr_weights, 1))
# #	W_ = np.zeros((nbr_weights, 1))
# #	for i_ in range(nbr_cells):
# #		for k_ in range(nbr_weights) : A_[:,k_] = list_QoI_models[k_][i_,:]
# #		b_[:,0] = QoI_HF[i_,:]
# #		ATA_[:,:] = A_.T @ A_ + gamma_ * Id_
# #		ATb_[:,0] = (A_.T @ b_)[:,0] 
# #		# solve optimization problem for each cell : Newton-Raphson
# #		xi_ = 0.
# #		q_ = 0.
# #		s_ = 0.
# #			
# #		update_GradJ(GradJ, ATA_, ATb_, W_, ones, q_, xi_, s_)
# #		update_Hessian(Hess, ATA_, Id_, W_, xi_, s_)
# #		delta_W = np.linalg.solve(Hess, GradJ)
# #		W_[:,0] = delta_W[:nbr_weights,0]
# #		
# #		Weights[i_,:] = W_[:,0]
# #		
# #		sum_sqr_weights = sum(W_)-1.
# #		WW = W_[:,0]
# #		J_value = np.linalg.norm(A_@ W_ - b_ )
# ##		print(f'J(.) = {J_value} \t weights = {WW} \t sum w = {sum_sqr_weights}')
# ##		input()
# #		
# #	return Weights




# "---------------------------------------------------------------------------------------------------------------------"
# "--------------------------------------- constraint sum of weights = 1 projected gradient w \in [-1,1] ----------------"
# "---------------------------------------------------------------------------------------------------------------------"
# def update_Hessian(Hess, ATA_, Id_, W_, W_tilda, xi_, gamma_):
# 	nbr_weights = len(ATA_)
# 	Hess[:nbr_weights, :nbr_weights] = ATA_+ gamma_ * Id_#(ATA_ + 2. * xi_ * Id_ )
# 	Hess[:nbr_weights, nbr_weights] = 1.
# 	Hess[nbr_weights, :nbr_weights] = 1.
    
# #	Hess[:nbr_weights, nbr_weights+1] = 2. * gamma_ * W_[:,0]
# #	Hess[nbr_weights+1, :nbr_weights] = 2. * gamma_ * W_[:,0]
    
    
# def update_GradJ(GradJ, ATA_, Id_, ATb_, W_, W_tilda, ones, xi_, gamma_):
# 	nbr_weights = len(W_)
# 	GradJ[:nbr_weights, :] = ATA_ @ W_ - ATb_ + xi_ + gamma_ * (W_ - W_tilda)# 2.*xi_* W_
# 	GradJ[nbr_weights, :] = sum(W_) - 1.
# #	GradJ[nbr_weights+1, :] = gamma_ * (W_.T@ W_)[0][0]
    
# def cacluate_Exact_weights(QoI_HF, list_QoI_models, weights_Gauss):
# 	gamma_ = 5e-3
# 	nbr_cells, nbr_components = QoI_HF.shape
# 	nbr_weights = len(list_QoI_models)
# 	Weights = np.zeros((nbr_cells,nbr_weights))
# 	Id_ = np.eye(nbr_weights)
# 	ones = np.zeros((nbr_components, 1))+1.
# 	A_ = np.zeros((nbr_components, nbr_weights))
# 	b_ = np.zeros((nbr_components, 1))
# 	ATA_ = np.zeros((nbr_weights, nbr_weights))
# 	ATb_ = np.zeros((nbr_weights, 1))
    
# 	Hess = np.zeros((nbr_weights + 1 , nbr_weights + 1 ))
# 	GradJ = np.zeros((nbr_weights + 1 , 1))
    
# 	LHS = np.zeros((nbr_weights, nbr_weights))
# 	RHS = np.zeros((nbr_weights, 1))
# 	for i__ in range(nbr_weights):
# 		RHS[i__, 0] = np.sum(list_QoI_models[i__] * QoI_HF)
# 		for j__ in range(nbr_weights):
# 			LHS[i__, j__] = np.sum(list_QoI_models[i__] * list_QoI_models[j__])
    
# 	W_tilda = 0.*np.linalg.solve(LHS+1e-8 * np.eye(nbr_weights), RHS)
    
# 	min_weights, max_weights = -2., 2.
    
# 	xi_ = np.zeros((nbr_weights, 1))
# 	W_ = np.zeros((nbr_weights, 1))
# 	for i_ in range(nbr_cells):
# 		for k_ in range(nbr_weights) : A_[:,k_] = list_QoI_models[k_][i_,:]
# 		b_[:,0] = QoI_HF[i_,:]
# 		ATA_[:,:] = A_.T @ A_ #+ gamma_ * Id_
# 		ATb_[:,0] = (A_.T @ b_)[:,0] 
# 		# solve optimization problem for each cell : Newton-Raphson
        
# 		xi_ = 0.
        
# 		sum_weights_Gauss = 0.
# 		for k_ in range(nbr_weights): sum_weights_Gauss += weights_Gauss[k_][i_]
# 		for k_ in range(nbr_weights):W_[k_,0] = weights_Gauss[k_][i_] / sum_weights_Gauss

        
# 		Weights[i_,:] = W_[:,0]
# 		J_value_init = np.linalg.norm(A_@ W_ - b_ )
# #		print('initial weight = ',W_[:,0], '\t J(.) = ', J_value_init)
# 		normGradJ = 1.
# 		iter_ = 1
# 		while normGradJ > 1e-6 and iter_ < 50:	
# 			update_GradJ(GradJ, ATA_, Id_, ATb_, W_, W_tilda, ones, xi_, gamma_)
# 			update_Hessian(Hess, ATA_, Id_, W_, W_tilda, xi_, gamma_)
# 			delta_W = np.linalg.solve(Hess, GradJ)
# 			xi_ -= delta_W[nbr_weights, 0]
            
# #			W_[:,0] -= delta_W[:nbr_weights,0]
# 			for k_ in range(nbr_weights):
# 				W_[k_,0] = min(max( min_weights, W_[k_,0] - delta_W[k_, 0]) , max_weights)
# #			
# 			normGradJ = np.linalg.norm(GradJ)
# 			J_value = np.linalg.norm(A_@ W_ - b_ )
# 			iter_+=1
        
# 		if J_value < J_value_init and abs(sum(W_[:,0])-1.)<1e-6:	
# #			print("true")
# 			Weights[i_,:] = W_[:,0]
# 		else :
# 			J_value = J_value_init
        
        
# #		WW = Weights[i_,:]
# #		sum_sqr_weights = sum(WW)
# #		print(f'iter : {iter_}\t J(.) = {J_value} \t norm gradJ = {normGradJ} \t weights = {WW} \t sum w = {sum_sqr_weights}')
# #		input()
# 	return Weights
    
    
# "---------------------------------------------------------------------------------------------------------------------"
# "--------------------------------------- constraint sum of weights ^2 = 1  (unite sphere) -------------------------"
# "---------------------------------------------------------------------------------------------------------------------"

# #def update_Hessian(Hess, ATA_, Id_, W_, xi_, s_):
# #	nbr_weights = len(ATA_)
# #	Hess[:nbr_weights, :nbr_weights] = (ATA_ + 2. * xi_ * Id_ )
# #	Hess[:nbr_weights, nbr_weights] = (2. * W_)[:,0]
# #	Hess[nbr_weights, :nbr_weights] = (2. * W_)[:,0]
# #	
# #	
# #def update_GradJ(GradJ, ATA_, ATb_, W_, ones, q_, xi_, s_):
# #	nbr_weights = len(W_)
# #	GradJ[:nbr_weights, :] = ATA_ @ W_ - ATb_ + 2.*xi_ * W_
# #	GradJ[nbr_weights, :] = (W_.T @ W_)[0][0]-1.

# #	
# #def cacluate_Exact_weights(QoI_HF, list_QoI_models, Gauss_weights):
# #	gamma_ = 1e-6
# #	nbr_cells, nbr_components = QoI_HF.shape
# #	nbr_weights = len(list_QoI_models)
# #	Weights = np.zeros((nbr_cells,nbr_weights))
# #	Id_ = np.eye(nbr_weights)
# #	ones = np.zeros((nbr_components, 1))+1.
# #	A_ = np.zeros((nbr_components, nbr_weights))
# #	b_ = np.zeros((nbr_components, 1))
# #	ATA_ = np.zeros((nbr_weights, nbr_weights))
# #	ATb_ = np.zeros((nbr_weights, 1))
# #	Hess = np.zeros((nbr_weights + 1 , nbr_weights + 1 ))
# #	GradJ = np.zeros((nbr_weights + 1 , 1))
# #	xi_ = np.zeros((nbr_weights, 1))
# #	W_ = np.zeros((nbr_weights, 1))
# #	for i_ in range(nbr_cells):
# #		for k_ in range(nbr_weights) : A_[:,k_] = list_QoI_models[k_][i_,:]
# #		b_[:,0] = QoI_HF[i_,:]
# #		ATA_[:,:] = A_.T @ A_ + gamma_ * Id_
# #		ATb_[:,0] = (A_.T @ b_)[:,0] 
# #		# solve optimization problem for each cell : Newton-Raphson
# #		xi_ = 0.
# #		q_ = 0.
# #		s_ = 0.
# #		W_[:,0] = 1./nbr_weights
# #		normGradJ = 1.
# #		iter_ = 1
# #		while normGradJ > 1e-6 and iter_ < 200:	
# #			update_GradJ(GradJ, ATA_, ATb_, W_, ones, q_, xi_, s_)
# #			update_Hessian(Hess, ATA_, Id_, W_, xi_, s_)
# #			np.savetxt("Hess.txt", Hess)
# #			delta_W = np.linalg.solve(Hess, GradJ)
# #			W_[:,0] -= delta_W[:nbr_weights,0]
# #			xi_ -= delta_W[nbr_weights, 0]
# #			
# #			normGradJ = np.linalg.norm(GradJ)
# #			sum_sqr_weights = (W_.T @ W_)[0][0]-1.
# #			WW = W_[:,0]
# #			J_value = np.linalg.norm(A_@ W_ - b_ )
# #			iter_+=1
# #		
# #		Weights[i_,:] = W_[:,0]
# ##		print(f'iter : {iter_}\t J(.) = {J_value} \t norm gradJ = {normGradJ} \t weights = {WW} \t sum w^2 = {sum_sqr_weights}')
# ##		input()
# #		
# #	return Weights



# "---------------------------------------------------------------------------------------------------------------------"
# "----------------------- combination of constraints sum=1 and sum of ^2 = 1  (unite sphere) ------------------------"
# "---------------------------------------------------------------------------------------------------------------------"

# #def update_Hessian(Hess, ATA_, Id_, W_, xi_, s_):
# #	Hess*=0.
# #	nbr_weights = len(ATA_)
# #	Hess[:nbr_weights, :nbr_weights] = (ATA_ + 2. * xi_ * Id_ ) / 2.
# #	Hess[nbr_weights+2, nbr_weights+2] = (2. * xi_ ) /2.	
# #	
# #	Hess[:nbr_weights, nbr_weights] = 1.
# #	Hess[:nbr_weights, nbr_weights+1] = (2. * W_)[:,0]
# #	Hess[nbr_weights+1, nbr_weights+2] = 2. * s_
# #	
# #	
# #	Hess[nbr_weights, :nbr_weights] = 1.
# #	Hess[nbr_weights+1, :nbr_weights] = (2. * W_)[:,0]
# #	Hess[nbr_weights+2, nbr_weights+1] = 2. * s_
# #	
# ##	np.savetxt("Hess.txt", Hess), input()
# #	
# #	
# #def update_GradJ(GradJ, ATA_, ATb_, W_, ones, q_, xi_, s_):
# #	nbr_weights = len(W_)
# #	GradJ[:nbr_weights, :] = ATA_ @ W_ - ATb_ + q_ * ones + 2.*xi_ * W_
# #	GradJ[nbr_weights, :] = sum(W_) - 1.
# #	GradJ[nbr_weights+1, :] = W_.T @ W_ - 1. + s_**2.
# #	GradJ[nbr_weights+2, :] = 2. * s_ * xi_
# #	
# #def cacluate_Exact_weights(QoI_HF, list_QoI_models, Gauss_weights):
# #	gamma_ = 1e-6
# #	nbr_cells, nbr_components = QoI_HF.shape
# #	nbr_weights = len(list_QoI_models)
# #	Id_ = np.eye(nbr_weights)
# #	ones = np.zeros((nbr_components, 1))+1.
# #	A_ = np.zeros((nbr_components, nbr_weights))
# #	b_ = np.zeros((nbr_components, 1))
# #	ATA_ = np.zeros((nbr_weights, nbr_weights))
# #	ATb_ = np.zeros((nbr_weights, 1))
# #	Hess = np.zeros((nbr_weights+3 , nbr_weights+3 ))
# #	GradJ = np.zeros((nbr_weights+3 , 1))
# #	xi_ = np.zeros((nbr_weights, 1))
# #	W_ = np.zeros((nbr_weights, 1))
# #	for i_ in range(nbr_cells):
# #		for k_ in range(nbr_weights) : A_[:,k_] = list_QoI_models[k_][i_,:]
# #		b_[:,0] = QoI_HF[i_,:]
# #		ATA_[:,:] = A_.T @ A_ + gamma_ * Id_
# #		ATb_[:,0] = (A_.T @ b_)[:,0] 
# #		# solve optimization problem for each cell : Newton-Raphson
# #		q_ = 0.
# #		xi_ = 1e-3
# #		s_ = 1e-1
# #		W_[:,0] = 0.#1./nbr_weights
# #		normGradJ = 1.
# #		J_value = np.linalg.norm(A_@ W_ - b_ )
# ##		print(f'J(.) = {J_value}')
# #		iter_ = 1
# #		while normGradJ > 1e-6 and iter_ < 100:	
# #			update_GradJ(GradJ, ATA_, ATb_, W_, ones, q_, xi_, s_)
# #			update_Hessian(Hess, ATA_, Id_, W_, xi_, s_)
# #			delta_W = np.linalg.solve(Hess, GradJ)
# #			W_[:,0] -= delta_W[:nbr_weights,0]
# #			q_ -= delta_W[nbr_weights, 0]
# #			xi_ -= delta_W[nbr_weights+1, 0]
# ##			s_ -= delta_W[nbr_weights+2, 0]
# #			s_ = min(max( 0., s_ - delta_W[nbr_weights+2, 0]), 1.)
# #			
# #			normGradJ = np.linalg.norm(GradJ)
# #			sum_weights = sum(W_)-1.
# #			sum_sqr_weights = (W_.T @ W_)[0][0]-1.+s_**2.
# #			WW = W_[:,0]
# #			J_value = np.linalg.norm(A_@ W_ - b_ )
# #			iter_+=1
# ##		print(s_)	
# ##		print(f'iter : {iter_}\t J(.) = {J_value} \t norm gradJ = {normGradJ} \t weights = {WW} \t sum w = {sum_weights}\t sum w^2 = {sum_sqr_weights}')	
# ##		input()
# #	return weights

# "---------------------------------------------------------------------------------------------------------------------"
# "---------------------------------------------------------------------------------------------------------------------"
# "---------------------------------------------------------------------------------------------------------------------"

# # ------- Exact waits: mixture of experts
# def Optimal_model_weights(dic_data, sigma):
# 	weights = {}
# 	features = {}
# 	weightsU ={}
    
# 	for case in dic_data:
# 		if case != 'Jet_NearSonic' and case != 'Jet_subSonic':
# 			# internal mesh 
# 			features.update({case : {}})
# 			U_HF = dic_data[case]['Frozen']['internalMesh']['U'][:,0].reshape((-1,1))
# 			list_U_models = [], [], []
# 			weightsU_Gauss_case = [], [], []
            
# 			for model in dic_data[case]:
# 				if model != 'Frozen':
# 					print(case, model)
# 					U_model = dic_data[case][model]['internalMesh']['U'][:,0].reshape((-1,1))
                    
# 					# Gather data for all models
# 					list_U_models.append(U_model)
                    
# 					weightsU_Gauss_case.append(1e-12 + np.exp(-0.5*np.sum( (U_HF - U_model)**2., axis=1)/sigma**2))
# 					features[case].update({model : np.hstack([np.array(dic_data[case][model]['internalMesh'][f'eta_{k_}']).reshape((-1,1)) for k_ in range(1,12)]) }) 
            
# 			weightsU_case = cacluate_Exact_weights(U_HF, list_U_models, weightsU_Gauss_case)
# 			#idx_bD = [1, 2, 4]			
# 			#
# 			ErrorU = np.zeros(U_HF.shape); ErrorU[:,:] = U_HF[:,:]

# 			for j_ in range(3):
# 				for k_ in range(U_HF.shape[1]) : ErrorU[:,k_] -= weightsU_case[:,j_] * list_U_models[j_][:,k_]
# 			norm_ErrorU = np.linalg.norm(ErrorU)
# 			print(f'|| U_models * W - Uref || = {norm_ErrorU}') 
            
# 			weightsU.update({case : weightsU_case })
        
# #			features.update({case : np.hstack([np.array(dic_data[case][model]['internalMesh'][f'eta_{k_}']).reshape((-1,1)) for k_ in range(1,12)]) })
# 			#
# 			# boundaries
# 			# ...	
                
            
            
            
# 			#Export exact weights
# #			np.savetxt(f'{case}/wU.txt', weightsU_case)
            
            
# 	return weightsU, features




# def select_best_weights_sigma_gradients(U_HF, grad_U_HF, list_U_models, list_grad_U_models):
# 	sigma_list = [25., 20., 15., 10., 5., 1., 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
# 	eps_ins=1e-5
    
# 	alpha=0.01
# 	all_weights = []
# 	error_sigma = []
# 	for sigma in sigma_list :
# 		weights_case = []
# 		for k_ in range(len(list_U_models)):
# 			#diff = epsilon_insensible(U_HF, list_U_models[k_], sigma, eps_ins)
# 			diff = np.linalg.norm(U_HF - list_U_models[k_], axis=1).reshape((-1,1))
# 			diff_grad = np.linalg.norm(grad_U_HF - list_grad_U_models[k_], axis=1).reshape((-1,1))
# 			diff_sqr_total = diff**2. + alpha * diff_grad**2.
# 			sigma_grad = np.linalg.norm(grad_U_HF, axis=1).reshape((-1,1))
# 			sigma_grad/=np.linalg.norm(grad_U_HF)
# 			sigma_grad= np.exp(-sigma_grad**2./1e-1)
# #			diff_sqr_total = np.where(diff_sqr_total < eps_ins , 0., diff_sqr_total)
# 			weights_case.append( 1e-12 + np.exp( -  0.5 * diff_sqr_total / (sigma)**2. ) )
# 		weights_case_sum1 = [weights_case[k_] / sum(weights_case) for k_ in range(len(list_U_models))]
# 		all_weights.append(weights_case_sum1)
# 		error_sigma.append(calculate_error_(U_HF, list_U_models, weights_case_sum1))	
# 	min_index = np.argmin(error_sigma)
# 	best_weights = all_weights[min_index]
# 	return best_weights



# def predit_weights(path):
# 	features = np.hstack([np.array(dic_data[case][model]['internalMesh'][f'eta_{k_}']).reshape((-1,1)) for k_ in range(1,12)])
# 	weights_pred = rf.predict(features)


# def gradient_of(mesh, fieldname):
# 	mesh_g = mesh.compute_derivative(scalars=fieldname)
# 	gradient_array = mesh_g["gradient"]
# 	return gradient_array



# def epsilon_insensible(x_hf, x_model, sigma, eps):
# #	diff = np.sum( np.abs(x - y)**2., axis=1) / (2.*sigma**2.)
# 	diff = np.linalg.norm(x_hf - x_model, axis=1).reshape((-1,1))
# # / np.linalg.norm(1e-18+x_hf, axis=1).reshape((-1,1))
# 	# set 0 when diff < eps
# #	diff2 = np.linalg.norm(x - y, axis=1) / np.linalg.norm(x, axis=1)
# #	diff2 = np.linalg.norm((x+1e12 - y)/(1e-12+x), axis=1).reshape((-1,1))
# 	diff = np.where(diff < eps, 0., diff)
# #	for i_ in range(diff.shape[0]):
# #		if diff[i_]<eps : diff[i_] = 0. 
# 	sigma_selectif = 1e-8
# 	gauss_term = np.where(diff < eps, 0, 0.5*diff**2./ sigma**2.)
# 	return gauss_term.reshape((-1,1))
        





























