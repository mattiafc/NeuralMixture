# NeuralMixture
Neural network to find the optimal mixture of expert to inform RANS turbulence modeling

To install the package, clone the repo and then do the following
cd NeuralMixturv
python3 -m -env .venv
source .venv/bin/activate
pip install -e .

Once the procedure is finished. You'll be able to use the NeuralMixture commands specified in the pyproject.toml file.

- Simulations can be setup by running the command setup_simulation with the JSONs/setupSimulation.json file by specifying a model name (folder name), the model coefficients (theta and thetaR). The JSON can also specify commands to be execute: mesh_command, preprocess_command, decompose_command, simulation_command, and postprocess_command can be used to specify commands that the python script will run as from terminal. you can also create a custom ./execute_something.sh file to be execute by any _command.

- Exact (or HF) results can be donwload from ADDLINK.

- Features can be extracted from simulation results by running the command extract_features with the JSONs/generateDataset.json files. To do so, you need to copy the Exact results on an OF mesh in the case/Exact folder. At this point, you need to specify the simulation_home containing all the directories where you store the cases
