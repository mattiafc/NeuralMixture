import os
import shutil
import numpy as np
import pyvista as pv

def OpenFOAM_header(OF_class, OF_object):
    header = "/*--------------------------------*- C++ -*----------------------------------*\\\n"
    header += "| =========                 |                                                 |\n"
    header += "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
    header += "|  \\    /   O peration     | Version:  2.3.0                                 |\n"
    header += "|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n"
    header += "|    \\/     M anipulation  |                                                 |\n"
    header += "\\*---------------------------------------------------------------------------*/\n"
    header += "FoamFile\n"
    header += "{\n"
    header += "    version     2.0;\n"
    header += "    format      ascii;\n"
    header += f"    class       {OF_class};\n"
    header += f"    object      {OF_object};\n"
    header += "}\n"
    header += "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    return header

def get_last_modified_folder(case_dir):
    simulation_folders = [folder for folder in os.listdir(case_dir) if os.path.isdir(os.path.join(case_dir, folder)) and any(char.isdigit() for char in folder)]
    if not simulation_folders:
        return None
    return max(simulation_folders, key=lambda x: int(''.join(filter(str.isdigit, x))))

def parse_scalar_field(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    start = content.index('(\n', content.index('nonuniform')) + 2
    end   = content.index('\n)', start)
    return np.array([float(v) for v in content[start:end].split()])

def read_openfoam_centers_2D(path, time_folder="0"):
    """Read OpenFOAM cell center coordinates from Cx, Cy files."""
    x = parse_scalar_field(os.path.join(path, time_folder, 'Cx'))
    y = parse_scalar_field(os.path.join(path, time_folder, 'Cy'))
    return x, y
    
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

        
        file.write(OpenFOAM_header("volScalarField",field_name))

        
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

def load_OpenFOAM_data(path):

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

def write_OpenFOAM_with_boundaries(path_to_case, input_file, output_file, time_value, internalMesh, boundaries):

    # Read content of the original file
    with open(os.path.join(path_to_case, time_value, input_file), 'r') as fin:
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


def generate_FOAM_case_from_pyvista(path_output, internalMesh, boundaries, source_polymesh_path, time_value=5000):
    """
    Generate an OpenFOAM case directory from PyVista mesh data.

    Parameters
    ----------
    path_output          : str   output case root directory
    internalMesh         : pv.UnstructuredGrid  internal mesh with field arrays
    boundaries           : pv.CompositeDataSet  boundary patches (keys used for boundaryField)
    source_polymesh_path : str   path to an existing constant/polyMesh to copy
    time_value           : int   time step folder name (default 5000)
    """
    time_dir     = os.path.join(path_output, str(time_value))
    polymesh_dir = os.path.join(path_output, 'constant', 'polyMesh')
    os.makedirs(time_dir,     exist_ok=True)
    os.makedirs(polymesh_dir, exist_ok=True)

    # --- Copy polyMesh from source case ---
    for fname in os.listdir(source_polymesh_path):
        src = os.path.join(source_polymesh_path, fname)
        dst = os.path.join(polymesh_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # --- Write fields ---
    QoIs_list = list(dict.fromkeys(internalMesh.cell_data.keys()))

    for QoI in QoIs_list:
        data = np.array(internalMesh[QoI])
        n_cells = len(data)

        if data.ndim == 1:
            field_class, field_type = 'volScalarField',     'scalar'
        elif data.shape[1] == 3:
            field_class, field_type = 'volVectorField',     'vector'
        elif data.shape[1] == 6:
            field_class, field_type = 'volSymmTensorField', 'symmTensor'
        elif data.shape[1] == 9:
            field_class, field_type = 'volTensorField', 'tensor'
        else:
            print(f"Skipping {QoI}: unsupported shape {data.shape}")
            continue

        with open(os.path.join(time_dir, QoI), 'w') as f:
            f.write(OpenFOAM_header(field_class, QoI))
            f.write('dimensions      [0 0 0 0 0 0 0];\n\n')

            if field_type == 'scalar':
                f.write(f'internalField   nonuniform List<scalar>\n{n_cells}\n(\n')
                for v in data:
                    f.write(f'{v}\n')
            elif field_type == 'vector':
                f.write(f'internalField   nonuniform List<vector>\n{n_cells}\n(\n')
                for row in data:
                    f.write(f'({row[0]} {row[1]} {row[2]})\n')
            elif field_type == 'symmTensor':
                f.write(f'internalField   nonuniform List<symmTensor>\n{n_cells}\n(\n')
                for row in data:
                    f.write(f'({row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]})\n')
            elif field_type == 'tensor':
                f.write(f'internalField   nonuniform List<tensor>\n{n_cells}\n(\n')
                for row in data:
                    f.write(f'({row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]} {row[6]} {row[7]} {row[8]})\n')
            f.write(');\n\n')
            f.write('boundaryField\n{\n')

            if boundaries is not None:
                for patch_name in boundaries.keys():
                    f.write(f'\t{patch_name}\n\t{{\n\t\ttype\t\t\tzeroGradient;\n\t}}\n')

            f.write('}\n\n')
            f.write('// ************************************************************************* //')

    # create empty case.foam so PyVista/paraview can open it
    open(os.path.join(path_output, 'case.foam'), 'w').close()
    print(f"OpenFOAM case written to: {path_output}, time: {time_value}")