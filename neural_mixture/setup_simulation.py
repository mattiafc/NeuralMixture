import numpy as np
import pandas as pd
import argparse
import json
import os
import subprocess

def write_model_coefficients(theta_coeff, thetaR_coeff, file_stored):
    file_stored += '\n\n'
    file_stored += f'       Theta  {theta_coeff};\n'
    file_stored += f'       ThetaR {thetaR_coeff};\n'
    file_stored += '\n       }\n}'
    return file_stored

def update_turbulence_properties(setup_dict, simulation_setup):
        
    sim_dir = os.path.join(setup_dict["home_directory"], simulation_setup["model"])

    with open(f'{os.path.join(sim_dir,"constant/turbulenceProperties")}', 'r', encoding='utf-8') as file:
        turbulence_properties = file.read()

    if simulation_setup["blending"]:
        turbulence_properties += '       Blending        true;\n\n'
        for model in simulation_setup["blending"]:
            turbulence_properties += '\n\n'
            turbulence_properties += f'       Theta_{model}  {simulation_setup[f"theta_{model}"]};\n'
            turbulence_properties += f'       ThetaR_{model} {simulation_setup[f"thetaR_{model}"]};\n'
    else:
        turbulence_properties += '\n\n'
        turbulence_properties += '       Blending        false;\n\n'
        turbulence_properties += f'       Theta  {simulation_setup["theta"]};\n'
        turbulence_properties += f'       ThetaR {simulation_setup["thetaR"]};\n'

    if('kOmegaSSTLM' in turbulence_properties):
        turbulence_properties += '\n       }\n}'
    else:
        turbulence_properties += '\n}'
                
    with open(f'{os.path.join(sim_dir,"constant/turbulenceProperties")}', 'w', encoding='utf-8') as file:
        file.write(turbulence_properties)

def main():
    
    ap = argparse.ArgumentParser(
        prog="setup_simulation",
        description=(
            "Setup an OpenFOAM simulation by copying a baseline case and updating the turbulence properties file with each model coefficients. \n"
            "You can take a look at a sample of the Json file in the examples directory. \n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", help="Path to JSON file containing the cases dict.")

    args = ap.parse_args()
    with open(args.config) as fh:
        setup_dict = json.load(fh)
    
    base_dir = os.path.join(setup_dict["home_directory"], setup_dict["baseline_directory"])
    simulations_entries = setup_dict["simulations"]

    for sim in simulations_entries:

        simulation_setup = simulations_entries[sim]

        # Copy the directory
        target_dir = os.path.join(setup_dict["home_directory"], simulation_setup["model"])
        os.system(f"cp -r {base_dir} {target_dir}")

        update_turbulence_properties(setup_dict, simulation_setup)

        # Update the turbulence properties file with the new coefficients

        # Mesh, decompose and run the simulation  
        if not(simulation_setup.get("mesh_command", False) == False):
            os.system(f"cd {target_dir} && {simulation_setup['mesh_command']}")

        # Mesh, decompose and run the simulation  
        if not(simulation_setup.get("preprocess_command", False) == False):
            os.system(f"cd {target_dir} && {simulation_setup['preprocess_command']}")
        
        if not(simulation_setup.get("decompose_command", False) == False):
            os.system(f"cd {target_dir} && {simulation_setup['decompose_command']}")

        if not(simulation_setup.get("simulation_command", False) == False):
            subprocess.Popen( simulation_setup['simulation_command'], shell=True, 
                cwd=target_dir, stdout=None, stderr=None)


if __name__ == "__main__":
    main()