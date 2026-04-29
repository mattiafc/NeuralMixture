from paraview.simple import *
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="Path to the .foam file")
parser.add_argument("output", help="Base name of the output CSV files")
args = parser.parse_args()

file_path = args.file_path
case_dir = os.path.dirname(os.path.abspath(file_path))

# Trova l'ultimo timestep e la directory 0
timestep_dirs = [d for d in os.listdir(case_dir)
                 if os.path.isdir(os.path.join(case_dir, d))
                 and d.replace('.', '', 1).isdigit() and d != '0']
latest = max(timestep_dirs, key=float)
latest_dir = os.path.join(case_dir, latest)
zero_dir = os.path.join(case_dir, '0')

# Crea symlink temporanei per i campi presenti solo in 0/
symlinks_created = []
for field in ['Ma', 'T', 'rho']:
    src = os.path.join(zero_dir, field)
    dst = os.path.join(latest_dir, field)
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)
        symlinks_created.append(dst)

try:
    # =============================================
    # PARTE 1: Estrazione dati airfoil (patch)
    # =============================================
    case_airfoil = OpenFOAMReader(FileName=file_path)
    case_airfoil.MeshRegions = ['patch/airfoil']
    case_airfoil.UpdatePipeline()
    c2p_airfoil = CellDatatoPointData(Input=case_airfoil)

    pass_arrays = PassArrays(Input=c2p_airfoil)
    pass_arrays.PointDataArrays = ['p', 'rho', 'T']

    temp_csv_airfoil = "temp_raw_airfoil.csv"
    writer_a = CreateWriter(temp_csv_airfoil, pass_arrays)
    writer_a.FieldAssociation = "Point Data"
    writer_a.UpdatePipeline()

    airfoil_output = f"{args.output}_Mis.csv"
    airfoil_fieldnames = ['Block Name', 'p', 'Points_0', 'Points_1', 'Points_2', 'rho', 'T']

    with open(temp_csv_airfoil, mode='r') as infile:
        reader = csv.DictReader(infile)
        raw_fields = reader.fieldnames
        has_rho = 'rho' in raw_fields
        has_T = 'T' in raw_fields

        with open(airfoil_output, mode='w', newline='') as outfile:
            writer_csv = csv.DictWriter(outfile, fieldnames=airfoil_fieldnames)
            writer_csv.writeheader()
            for row in reader:
                writer_csv.writerow({
                    'Block Name': 'airfoil',
                    'p': row['p'],
                    'Points_0': row['Points:0'],
                    'Points_1': row['Points:1'],
                    'Points_2': row['Points:2'],
                    'rho': row.get('rho', ''),
                    'T': row.get('T', ''),
                })

    if os.path.exists(temp_csv_airfoil):
        os.remove(temp_csv_airfoil)

    Delete(pass_arrays)
    Delete(c2p_airfoil)
    Delete(case_airfoil)
    print(f"Completato! File generato: {airfoil_output}")

    # =============================================
    # PARTE 2: Estrazione dati lungo linee verticali
    # =============================================
    case_mesh = OpenFOAMReader(FileName=file_path)
    case_mesh.MeshRegions = ['internalMesh']
    case_mesh.CellArrays = ['Ma', 'T', 'U', 'p', 'rho']
    case_mesh.UpdatePipeline()

    c2p_mesh = CellDatatoPointData(Input=case_mesh)
    c2p_mesh.UpdatePipeline()
    bounds = c2p_mesh.GetDataInformation().GetBounds()
    y_min = bounds[2]
    y_max = bounds[3]

    lines_config = {
        "MP2": {"x": 0.08725350350, "z": 0.0},
        "MP1": {"x": -0.0201098565, "z": 0.0},
    }

    line_fieldnames = ['M', 'x', 'y', 'z', 'T', 'u', 'v', 'w', 'p', 'rho']

    for label, coord in lines_config.items():
        line = PlotOverLine(Input=c2p_mesh)
        line.Point1 = [coord["x"], y_min, coord["z"]]
        line.Point2 = [coord["x"], y_max, coord["z"]]
        line.Resolution = 1000
        line.UpdatePipeline()

        temp_csv_line = f"temp_raw_{label}.csv"
        writer_l = CreateWriter(temp_csv_line, line)
        writer_l.FieldAssociation = "Point Data"
        writer_l.UpdatePipeline()

        final_output = f"{args.output}_{label}.csv"

        with open(temp_csv_line, mode='r') as infile:
            reader = csv.DictReader(infile)
            with open(final_output, mode='w', newline='') as outfile:
                writer_csv = csv.DictWriter(outfile, fieldnames=line_fieldnames)
                writer_csv.writeheader()
                for row in reader:
                    if any(v.strip().lower() == 'nan' for v in row.values()):
                        continue
                    writer_csv.writerow({
                        'M': row['Ma'],
                        'x': row['Points:0'],
                        'y': row['Points:1'],
                        'z': row['Points:2'],
                        'T': row['T'],
                        'u': row['U:0'],
                        'v': row['U:1'],
                        'w': row['U:2'],
                        'p': row['p'],
                        'rho': row['rho'],
                    })

        if os.path.exists(temp_csv_line):
            os.remove(temp_csv_line)

        Delete(line)
        print(f"Completato! File generato: {final_output}")

finally:
    for link in symlinks_created:
        if os.path.islink(link):
            os.unlink(link)
