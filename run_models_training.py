import pyvista as pv
import numpy					as np
import matplotlib.pyplot		as plt
import scipy					as sp 
from sklearn.ensemble import RandomForestRegressor
import pickle
import copy
from functions import *

from READ_WRITE_FEATURES import *

Modelname='RFR'

latest_time_foldername ="5000"

PathToCases="Ling/"

WeightsList = ["GaussSearchGrid"]
FeaturesChoices = ["Ling"]

dirModels = f"../ML_trained_models"

keys_to_select = ["Jet_NearSonic_projected", "CD12600", "channel1000_2D", "PH10595", "CBFS13700", "LRN_OGV_trans",  "LRN_OGV_trans_OP1"]

keys_weights_export = ["Jet_NearSonic", "Jet_NearSonic_projected", "Jet_NearSonic_restricted", "CD12600", "channel1000_2D", "PH10595", "CBFS13700", "LRN_OGV_trans", "LRN_OGV_trans_OP1"]

for FeaturesChoice in FeaturesChoices:
	
	cases = ["Jet_NearSonic", "CD12600", "channel1000_2D", "PH10595", "CBFS13700", "LRN_OGV_trans", "LRN_OGV_trans_OP1"]#
	models = ["Frozen", "ANSJ", "CHAN", "SEP"]# 


	print ("post-process jet data, this will change the dic and add new Jet_proj")		
	dic_data = create_dic_data(FeaturesChoice, cases, models)		
#	Jetdomain = 'restricted'#'projected' #


	"""This project the data from numerical simulation domain to PIV domain (symmetry is applied for cases ANSJ CHAN SEP)"""
#	if Jetdomain == 'projected':
	make_symm_Jet_data_on_PIVsubdomain(FeaturesChoice, dic_data, models, whichJet="Jet_NearSonic")
	export_Jet_foam_files(FeaturesChoice, models, dic_data, 'Jet_NearSonic', 'projected', 5000)

	## make predictions in python directly for the external XMA and save to the PIV domain 

	"""This takes the data from the simulation mesh and restricts it to the upper half PIV domain bounds """
#	if Jetdomain == 'restricted':
	restrict_Jet_data_to_upper_half_PIV_domain_bounds(FeaturesChoice, dic_data, models, whichJet="Jet_NearSonic")
	export_Jet_foam_files(FeaturesChoice, models, dic_data, 'Jet_NearSonic', 'restricted', 5000)
	
#	export_Jet_foam_files_Frozen_restricted(FeaturesChoice, models, dic_data, 'Jet_NearSonic', 'restricted', 5000)
	
	print('Jet foam data exported')#, input()
	
	
	
	for WhichWeights in WeightsList:
		if WhichWeights == "GaussSearchGrid":
			print ('\n\n\n\n--------------- Calculate Gaussian weights Grid search')
			weightsU, features = mixture_of_expert_Grid_search(dic_data, sigma=1e-3)


		
		print ('\n\n\n\n--------------- Export exact weights')
		for case in keys_weights_export: 
			if case == "Jet_NearSonic" : case_export = "Jet_NearSonic_restricted"
			else : case_export = case
			print(case)
			time_folder = "5000"
			export_folder = f"./{FeaturesChoice}/{case}/ExactWeights/{WhichWeights}"
			simul_folder = os.path.join(export_folder,time_folder)
			boundary_data, nCells = read_boundary_data(f"{export_folder}/constant/polyMesh")
			
			for i_ in range(3):
				write_scalar_field(simul_folder, time_folder, f"wU{i_+1}_exact", weightsU[case_export][:,i_], boundary_data)

		# Selected cases for training
		weightsU = {key: weightsU[key] for key in keys_to_select if key in weightsU}
		features = {key: features[key] for key in keys_to_select if key in features}
		print('exact weights exported !')
		
		
		
		
		weightsU_concat = np.vstack([weightsU[case] for case in weightsU])
		features_concat = np.vstack([ features[case]  for case in features])

		weights_all_concat = np.hstack([weightsU_concat])



		print (f'\n\n\n\n--------------- Train {Modelname} models')
		if Modelname == 'RFR':
			print("training ...")
			model_U, model_log_U = train_random_forest(features_concat, weightsU_concat) # changed rebecca 
			
		with open(f"{dirModels}/{FeaturesChoice}/{WhichWeights}/{Modelname}_log.txt", 'w') as file:	
			models_log = f"{Modelname} convergence info :\n\nfeatures : {FeaturesChoice}\t weights : {WhichWeights}"
			models_log += f"\n\nvariable : U\n" + model_log_U
			file.write(models_log)
		
		with open(f"{dirModels}/{FeaturesChoice}/{WhichWeights}/{Modelname}_model_U.pickle", "wb") as f:pickle.dump(model_U ,f)

#		weightsU_concat_pred = model_U.predict(features_concat)
#		weightsbDelta_concat_pred = model_bDelta.predict(features_concat)
#		weightsReskOmega_concat_pred = model_ReskOmega.predict(features_concat)
#		for k in range(3):
#				plt.figure()
#				plt.scatter(weightsU_concat[:,k], weightsU_concat_pred[:,k], alpha=0.5, label = "U")
#				plt.scatter(weightsbDelta_concat[:,k], weightsbDelta_concat_pred[:,k], alpha=0.5, label = "bD")
#				plt.scatter(weightsReskOmega_concat[:,k], weightsReskOmega_concat_pred[:,k], alpha=0.5, label = "R")
#				
#				plt.xlim(0,1)  # Set x-axis range from 1 to 5
#				plt.ylim(0,1) # Set y-axis range from 0 to 12

#				# Add labels and title
#				plt.xlabel('data')
#				plt.ylabel('predictions')
#				plt.title('weight '+repr(k+1) )
#				plt.legend()
#				plt.show()



#print ('\n\n\n\n--------------- Predict weights')

#weights_all_concat_pred = RF_model_all.predict(features_concat)

#print('||w_pred - w_exact|| = ', np.linalg.norm(weights_all_concat_pred - weights_all_concat))

#N_,Q_in = features_concat.shape
#N_, Q_out = weights_all_concat.shape

#for k in range(Q_out):
#	plt.figure()
#	
#	plt.plot( weights_all_concat[:,k], weights_all_concat_pred[:,k], alpha=0.5)

#	plt.title('mode_'+repr(k+1) )
#	plt.legend()
#	plt.show()









