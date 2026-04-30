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

def update_turbulence_properties(setup_dict, sims_dict, thetaR_coeff):
        with open(f'{os.path.join(setup_dict["home_directory"], setup_dict["baseline_directory"],"constant/turbulenceProperties")}.txt', 'r', encoding='utf-8') as file:
            turbulence_properties = file.read()

        if sims_dict[sim]["blending"]:
            pass
        else:
            with open(f'{os.path.join(target_dir,"constant/turbulenceProperties")}.txt', 'w', encoding='utf-8') as file:
                file.write(write_model_coefficients(sims_dict[sim]["theta_coeff"], sims_dict[sim]["thetaR_coeff"], turbulence_properties))

def main():
    
    ap = argparse.ArgumentParser(
        prog="read_mach_numb",
        description=(
            "Plot isentropic Mach and total-pressure loss for OpenFOAM RANS cases vs LES.\n\n"
            "JSON cases format:\n"
            '  {"tag": {"case": "case.foam", "fName": "Output",\n'
            '           "color": "tab:green", "label": "sim_1"}}'
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

        # Copy the directory
        target_dir = os.path.join(setup_dict["home_directory"], sims_dict[sim]["model"])
        os.system(f"cp -r {base_dir} {target_dir}")

        # Update the turbulence properties file with the new coefficients

        # Mesh, decompose and run the simulation  
        if sims_dict[sim]["mesh_command"]:
            os.system(f"cd {target_dir} && {sims_dict[sim]['mesh_command']}")
        
        if sims_dict[sim]["decompose_command"]:
            os.system(f"cd {target_dir} && {sims_dict[sim]['decompose_command']}")
    
        if sims_dict[sim]["simulation_command"]:
            os.system(f"cd {target_dir} && {sims_dict[sim]['simulation_command']} &&")