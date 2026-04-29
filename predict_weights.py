import numpy as np
import pickle
import sys, os
import scipy.ndimage
import scipy.interpolate
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/mciarlatani/DataDrivenML/dummy_XMA')

from READ_WRITE_FEATURES import *

ML_model_FW_ = sys.argv[1]
Model_directory = ""
current_directory = os.getcwd()

print(f'\n--------------- {ML_model_FW_} - Predict weights ...')

# Chargement du modèle U uniquement
with open(f"{ML_model_FW_}.pickle", "rb") as f:
    model_U = pickle.load(f)

# Lecture des features
nFeatures = 11
SIMULATION = "serial"

if SIMULATION == 'parallel':
    # Liste des sous-dossiers type "processor"
    all_items = os.listdir(current_directory)
    processor_folders = [
        item for item in all_items
        if os.path.isdir(os.path.join(current_directory, item)) and item.startswith('processor')
    ]
elif SIMULATION == 'serial':
    processor_folders = ['./']

# Parcours des dossiers processeurs
for pid, ps_folder in enumerate(processor_folders):

    time_folder = get_last_modified_folder(f'{ps_folder}')
    simul_folder = os.path.join(ps_folder, time_folder)

    boundary_data, nCells = read_boundary_data(f"{ps_folder}/constant/polyMesh")

    # Construction de la matrice des features
    for k_ in range(nFeatures):
        eta_k = read_volScalar_internalField(simul_folder, f'eta_{k_+1}', nCells)
        eta_k_array = np.array(eta_k).reshape((-1, 1))
        if k_ == 0:
            features = eta_k_array
        else:
            features = np.hstack((features, eta_k_array))

    # Prédiction des poids avec le modèle
    w_U = model_U.predict(features)
    w_U /= np.sum(w_U, axis=1, keepdims=True)  # Normalisation ligne à ligne

    # Écriture des champs scalaires prédits
    for i_ in range(3):
        write_scalar_field(simul_folder, time_folder, f"w_model_{i_}", w_U[:, i_], boundary_data)
        print("on a écrit scalar")

    # Si tu veux activer d'autres modèles, décommente ci-dessous :
    # write_scalar_field(simul_folder, time_folder, f"wbD{i_+1}", w_bDelta[:,i_], boundary_data)
    # write_scalar_field(simul_folder, time_folder, f"wR{i_+1}", w_ReskOmega[:,i_], boundary_data)

