import os
import copy
import torch

import pyvista           as pv
import numpy             as np
import matplotlib.pyplot as plt

from scipy.spatial     import cKDTree
from scipy.interpolate import griddata
from scipy.optimize    import minimize_scalar
from scipy.interpolate import RBFInterpolator

from sklearn.model_selection import cross_val_score
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error

from utils_OpenFOAM          import *

def create_dic_data(home_directory, cases) :
    
    '''
    Create a dictionary to store internal mesh and boundary data for each case and model.\n
    Cases : list of flow cases
    Models : list of expert models name
    '''
    dic_data = {}

    print(f"============================================")
    for case in cases:

        print(f"Loading case {case}")
        dic_data.update({case:{}})

        for model in cases[case]["models"].keys():
            path = os.path.join(home_directory, cases[case]["sub_directory"], model)
            internalMesh, boundaries = load_OpenFOAM_data(path)
            dic_data[case].update({model:{"internalMesh":internalMesh, "boundary": boundaries, "nCells":len(internalMesh.cell_centers().points)}})
    
    print(f"============================================")
            
    return dic_data

def generate_labels_features(dic_data, feature_names, model_order):

    print(f"=============================================================================")

    features = {}
    weightsU ={}

    for case in dic_data:

        features.update({case : {}})

        # internal mesh, getting the U,V components of the high fidelity solution
        U_HF = dic_data[case]['Exact']['internalMesh']['U'][:,0:2]#.reshape((-1,1)) # changed rebecca
        weightsU_case = []
        list_U_models= []

        idx = 0;
        for model in dic_data[case]:
            if model != 'Exact':

                assert(model == model_order[idx])
                idx+=1

                U_model = dic_data[case][model]['internalMesh']['U'][:,0:2]#.reshape((-1,1)) # changed rebecca
                list_U_models.append(U_model)

                features[case].update({model : np.hstack([np.array(dic_data[case][model]['internalMesh'][feat]).reshape((-1,1)) for feat in feature_names]) })


        weightsU_case, best_sigma = find_optimal_weights(U_HF, list_U_models)
        weightsU.update({case : np.hstack([ np.array( w_ / sum(weightsU_case)).reshape((-1,1)) for w_ in weightsU_case]) })

        print(f"Computed weights for case {case}; best sigma is {best_sigma:.3f}")

            # # boundaries
            # ErrorUV = np.zeros(U_HF.shape); ErrorUV[:,:] = U_HF[:,:]
            # n_models = len(weightsU_case)   # au lieu de range(3) en dur
            # for j_ in range(n_models):
            #     # weightsU_case[j_] : shape (N,)
            #     # list_U_models[j_] : shape (N,2)
            #     ErrorUV -= weightsU_case[j_].reshape(-1, 1) * list_U_models[j_]
            #     # broadcasting → (N,1)*(N,2) = (N,2)
            # norm_ErrorUV = np.linalg.norm(ErrorUV)

    print(f"=============================================================================")
        
    return weightsU, features

def find_optimal_weights(U_HF, list_U_models):

    def compute_weights(sigma):
        return [1e-12 + np.exp(-0.5 * np.linalg.norm(U_HF - U_k, axis=1).reshape((-1, 1))**2 / sigma**2) for U_k in list_U_models]

    def optimize_weight_loss(sigma):
        weights = compute_weights(sigma)
        w_norm = [w / sum(weights) for w in weights]
        return calculate_error_(U_HF, list_U_models, w_norm)

    res = minimize_scalar(optimize_weight_loss, bounds=(1e-5, 1.), method='bounded')
    sigma_opt = res.x
    weights = compute_weights(sigma_opt)
    best_weights = [w / sum(weights) for w in weights]

    return best_weights, sigma_opt

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


def interpolate_rbf(x_src, y_src, values, x_tgt, y_tgt,
                    kernel='thin_plate_spline', neighbors=None):
    """
    Interpolate field(s) from source to target mesh using local RBF.

    Parameters
    ----------
    x_src, y_src : (N,)      source coordinates
    values       : (N,) or (N, K)  field values at source
    x_tgt, y_tgt : (M,)     target coordinates
    kernel       : RBF kernel ('thin_plate_spline', 'linear', 'cubic', ...)
    neighbors    : number of nearest neighbors for local RBF (50-100 is good)

    Returns
    -------
    (M,) or (M, K) interpolated values
    """

    vals = np.atleast_2d(values).T if values.ndim == 1 else values

    src_pts = np.column_stack([x_src, y_src]).astype(np.float32)
    tgt_pts = np.column_stack([x_tgt, y_tgt]).astype(np.float32)

    # Remove duplicates
    _, idx = np.unique(src_pts, axis=0, return_index=True)
    src_pts = src_pts[idx]
    vals = vals[idx]

    # Normalize coordinates
    scale = np.std(src_pts, axis=0)
    src_pts /= scale
    tgt_pts /= scale

    rbf = RBFInterpolator(src_pts, vals, neighbors=neighbors, kernel=kernel, epsilon = 1e-6)

    out = rbf(tgt_pts)

    return out[:, 0] if values.ndim == 1 else out

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

    

















