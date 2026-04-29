import os

def get_last_modified_folder(case_dir):
    simulation_folders = [folder for folder in os.listdir(case_dir) if os.path.isdir(os.path.join(case_dir, folder)) and any(char.isdigit() for char in folder)]
    if not simulation_folders:
        return None
    return max(simulation_folders, key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    
def read_volScalar_internalField(FolderPath, field_name, nCells):
    array = []
    try:
        with open(f"{FolderPath}/{field_name}", "r") as file:
            lines = file.readlines()
            internal_data_started = False
            
            for line in lines:
                if "internalField" in line.split() and "uniform" in line.split():
                    value = float(line.split()[2][:-1])
                    array = [value for iC in range(nCells)]
                    break;

                if not internal_data_started:
                    if line.strip() == "(":  
                        internal_data_started = True
                    continue
                
                if line.strip().startswith(")"): 
                    break
                
                array.append(float(line.strip()))
                
    except FileNotFoundError:
        print(f"Error: Could not find {field_name} file in '{FolderPath}'.")
        
    return array


def read_owner(FolderPath):
    owner = []
    try:
        with open(f"{FolderPath}/owner", "r") as file:
            lines = file.readlines()
            internal_data_started = False
            for line in lines:
                
                if line.startswith("    note"):
                    nCells = int(line.split()[2].split(":")[1])
                if not internal_data_started:
                    if line.strip() == "(":  
                        internal_data_started = True
                    continue
                if line.strip().startswith(")"): 
                    break
                owner.append(int(line.strip()))
    except FileNotFoundError:
        print(f"Error: Could not find the 'owner' file in '{FolderPath}'.")
    return owner, nCells


def add_nCells_for_boundary_faces_from_owner(polyMesh_dir, boundary_data):
    owner, nCells = read_owner(polyMesh_dir)
    for boundary_name, data in boundary_data.items():
        start_face = int(data["startFace"])
        n_faces = int(data["nFaces"])
        cells = []
        for i in range(start_face, start_face + n_faces):
            face_index = i - 1  # Face indices in OpenFOAM are 0-based
            owner_cell = owner[face_index]
            cells.append(owner_cell)		
        boundary_data[boundary_name].update({'bd_cells':cells})
    return nCells	

def read_boundary_data(polyMesh_dir):
    boundary_data = {}
    try:
        with open(f"{polyMesh_dir}/boundary", "r") as file:
            lines = file.readlines()
            boundary_data_started = False
            for line in lines:
                if not boundary_data_started:
                    if line.strip() == "(":  # Start of boundary definitions
                        boundary_data_started = True
                    continue
                if line.strip().startswith(")"):  # End of boundary definitions
                    break
                
                words = line.split()  # Split the line into words
                
                if len(words)==1 and words[0]!='{' and words[0]!='}' :
                    boundary_name = words[0]
                    boundary_data.update({boundary_name:{}})
                else :
                    if words[0] == "type":
                        boundary_type=words[1][:-1]
                        boundary_data[boundary_name].update({'type':boundary_type})
                    elif words[0] == "nFaces":
                        nFaces=words[1][:-1]
                        boundary_data[boundary_name].update({'nFaces':nFaces})
                    elif words[0] == "startFace": 
                        startFace=words[1][:-1]
                        boundary_data[boundary_name].update({'startFace':startFace})

                    

    except FileNotFoundError:
        print(f"Error: Could not find 'boundary' file in '{polyMesh_dir}'.")
    
    nCells = add_nCells_for_boundary_faces_from_owner(polyMesh_dir, boundary_data)
    
    return boundary_data, nCells

def write_scalar_field(simul_folder, time_folder, field_name, internal_field, boundary_data):

    internal_field_file_path = f"{simul_folder}/{field_name}"
    with open(internal_field_file_path, "w") as file:

        
        file.write(f"/*--------------------------------*- C++ -*----------------------------------* \n")
        file.write(f"| =========                 |                                                 |\n")
        file.write(f"| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
        file.write(f"|  \\    /   O peration     | Version:  2306                                  |\n")
        file.write(f"|   \\  /    A nd           | Website:  www.openfoam.com                      |\n")
        file.write(f"|    \\/     M anipulation  |                                                 |\n")
        file.write(f"\*---------------------------------------------------------------------------*/\n")

        
        file.write("FoamFile\n")
        file.write("{\n")
        file.write("\tversion     2.0;\n")
        file.write("\tformat      ascii;\n")
        file.write("\tarch        \"LSB;label=32;scalar=64\";\n")
        file.write("\tclass       volScalarField;\n")
        file.write(f"\tlocation    \"{time_folder}\";\n")
        file.write(f"\tobject      {field_name};	\n")
        file.write("}\n")
        file.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")

        
        nCells = len(internal_field)
        file.write(f"dimensions  \t[0 0 0 0 0 0 0];\n\ninternalField\tnonuniform List<scalar>\n{nCells}\n(\n")
        for value in internal_field:
            file.write(f"{value}\n")
        file.write(")\n;\n\n")
        
        file.write("boundaryField\n{\n")
        
        for boundary_name, patch_data in boundary_data.items():
            patch_type = patch_data['type']
            nFaces = patch_data['nFaces']
            ids_adjacent_cells = patch_data['bd_cells']
            
            if patch_type == 'empty' or patch_type == 'symmetryPlane' or patch_type == 'cyclicAMI' or patch_type == 'wedge':
                file.write(f"\t{boundary_name}\n\t{{\n\t\ttype\t\t\t{patch_type};\n\t}}\n")
            else :
                if patch_type == 'processor' : calcType = 'processor'
                else :calcType = 'calculated'
                
                if nFaces == '0':
                    file.write(f"\t{boundary_name}\n\t{{\n\t\ttype\t\t\t{calcType};\n\t\tvalue\t\t\tnonuniform List<scalar> {nFaces}();\n")
                    file.write("\t}\n")
                else :
                    file.write(f"\t{boundary_name}\n\t{{\n\t\ttype\t\t\t{calcType};\n\t\tvalue\t\t\tnonuniform List<scalar>\n{nFaces}\n(\n")
                    for id_adjacent_cell in ids_adjacent_cells:
                        value = internal_field[id_adjacent_cell]
                        file.write(f"{value}\n")
                    file.write(")\n;\n\t}\n")
                
        file.write("}\n\n\n// ************************************************************************* //")
