import numpy as np
import pandas as pd
import argparse
import json
import os

def OpenFOAM_header():
    header = "/*--------------------------------*- C++ -*----------------------------------*\\"
    header += "/*--------------------------------*- C++ -*----------------------------------*\\"
    header += "| =========                 |                                                 |"
    header += "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |"
    header += "|  \\    /   O peration     | Version:  2.3.0                                 |"
    header += "|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |"
    header += "|    \\/     M anipulation  |                                                 |"
    header += "\\*---------------------------------------------------------------------------*/"
    header += "FoamFile"
    header += "{"
    header += "    version     2.0;"
    header += "    format      ascii;"
    header += "    class       dictionary;"
    header += "    object      blockMeshDict;"
    header += "}"
    header += "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"
    return header

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

    turbulence_properties += '\n       }\n}'
                
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
    sims_dict = setup_dict["simulations"]

    for sim in sims_dict:

        simulation_setup = sims_dict[sim]

        # Copy the directory
        target_dir = os.path.join(setup_dict["home_directory"], simulation_setup["model"])
        os.system(f"cp -r {base_dir} {target_dir}")

        update_turbulence_properties(setup_dict, simulation_setup)

        # Update the turbulence properties file with the new coefficients

        # Mesh, decompose and run the simulation  
        if simulation_setup["mesh_command"]:
            os.system(f"cd {target_dir} && {simulation_setup['mesh_command']}")
        
        if simulation_setup["decompose_command"]:
            os.system(f"cd {target_dir} && {simulation_setup['decompose_command']}")

        # if simulation_setup["simulation_command"]:
        #     os.system(f"cd {target_dir} && {simulation_setup['simulation_command']} &&")


if __name__ == "__main__":
    main()