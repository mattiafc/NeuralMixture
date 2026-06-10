from matplotlib.pylab import var
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import pandas as pd

from scipy.interpolate import splprep, splev
from scipy.interpolate import Rbf
from scipy.interpolate import RBFInterpolator

from musicaa_utils import get_block_info, line_interp, mixed_out, plot_grid, read_grid, read_info, read_stats

def read_openfoam_centers(path, time_folder="0"):
    """Read OpenFOAM cell center coordinates from Cx, Cy files."""
    x = parse_scalar_field(os.path.join(path, time_folder, 'Cx'))
    y = parse_scalar_field(os.path.join(path, time_folder, 'Cy'))
    return x, y


def interpolate_rbf(x_src, y_src, values, x_tgt, y_tgt,
                    kernel='thin_plate_spline', neighbors=64):
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

def rearrange(x,d):
    len_ = d['nx_bl3']+d['nx_bl4']
    # save bl6 coords
    x_ = np.zeros(d['nx_bl6'])
    x_ = x[len_:len_+d['nx_bl6']].copy()
    # rotate array
    x[len_:len_+d['nx_bl7']] = x[-d['nx_bl7']:].copy()
    x[-d['nx_bl6']:] = x_.copy()

    return x

def read_musicaa_case(input_dir, dict_input, plot_mesh=False):

    print('Reading case from: ', input_dir)

    pitch      = dict_input['pitch']
    x1_samp    = dict_input['x1']
    x2_samp    = dict_input['x2']
    in_blocks  = dict_input['in_blocks']
    out_blocks = dict_input['out_blocks']
    wall_blocks= dict_input['wall_blocks']

    dict_info = read_info(input_dir)
    block_info = get_block_info(input_dir)
    ngh = int(dict_info["ngh"])
    n_block = dict_info["nbloc"]
    Re_in  = dict_info['Reref']
    u_in   = dict_info['Uref']
    p_in   = dict_info['Pref']
    rho_in = dict_info['Roref']
    mu_in  = dict_info['Muref']
    c_in   = dict_info['cref']
    nz     = dict_info['nz_bl1']

    print('Input case info: Re = ', Re_in, ', U = ', u_in, ', p = ', p_in, ', rho = ', rho_in, ', mu = ', mu_in, ', c = ', c_in)

    data = {}
    stats1 = {}
    stats2 = {}
    x_flat   = []
    y_flat   = []

    for bl in range(1, n_block + 1):
        data[bl] = {}
        # read grid
        bl_file = os.path.join(input_dir, f'grid_bl{bl}_ngh{ngh}.bin')
        nx, ny, nz, x, y, z = read_grid(input_dir, bl_file)
        # scale grid
        data[bl]["x"], data[bl]["y"], data[bl]["z"] = x/1000, y/1000, z/1000

        if bl == 6:
            data[bl]["y"] -= pitch

        if bl in wall_blocks:
            block_info[f'block_{bl}']['wall'] = True
        else:
            block_info[f'block_{bl}']['wall'] = False

        # Wall location
        block_info[f'block_{bl}']['jmin'] = True
        block_info[f'block_{bl}']['imax'] = False

    # Plot mesh
    if plot_mesh is True:
        plot_grid(input_dir, True, n_bl=n_block, every=5, figsize=(5.2, 3.64))
    
    nw = sum([dict_info[f'nx_bl{bl}'] for bl in wall_blocks])
    
    x1,xw,x1_     = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    y1,yw,y1_     = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    duxw,duyw,dxw = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    dvxw,dvyw,dyw = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    rhow,muw,pw   = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    tau = np.zeros((nw,2,2))

    # Statistics dictionary are created and filled with each block's data
    
    nw_loc = 0
    for bl in range(1, n_block + 1):

        nx, ny = block_info[f"block_{bl}"]["nx"], block_info[f"block_{bl}"]["ny"]
        stats1[bl] = read_stats(os.path.join(input_dir, f"stats1_bl{bl}.bin"), nx, ny)
        stats2[bl] = read_stats(os.path.join(input_dir, f"stats2_bl{bl}.bin"), nx, ny)
        
        cp = stats2[bl]['cp']
        cv = stats2[bl]['cv']
        gamma = cp / cv

        # # isentropic Mach number
        # data[f'block_{bl}']['M_is'] = np.sqrt(2/(gam_in-1)*\
        #       ((p0_in/stats1[bl] ['p'])**((gam_in-1)/gam_in)-1))
        # M_is_flat.append(data[f'block_{bl}']['M_is'].flatten())
        
        data[bl]['rho'] = stats1[bl]['rho']
        data[bl]['x_flat'] = data[bl]['x'].flatten()
        data[bl]['y_flat'] = data[bl]['y'].flatten()
        data[bl]['Mach']   = stats2[bl]['M']
        data[bl]['P0'] = stats1[bl]['p'] * (1 + (gamma - 1) / 2 * data[bl]['Mach']**2)**(gamma / (gamma - 1))
        data[bl]['u'] = stats1[bl]['u']
        data[bl]['p'] = stats1[bl]['p']
        data[bl]['v'] = stats1[bl]['v']
        data[bl]['TKE'] = 0.5*(stats1[bl]['uu'] + stats1[bl]['vv'] + stats1[bl]['ww'])

        x_flat.append(data[bl]['x'].flatten())
        y_flat.append(data[bl]['y'].flatten())

        if block_info[f'block_{bl}']['wall']:
            if block_info[f'block_{bl}']['jmin']:
                # Wall coordinates
                xw[nw_loc:nw_loc+nx]  = data[bl]['x'][:,0]
                yw[nw_loc:nw_loc+nx]  = data[bl]['y'][:,0]
                pw[nw_loc:nw_loc+nx] = stats1[bl]['p'][:,0]

    xw = rearrange(xw,dict_info)
    yw = rearrange(yw,dict_info)
    pw = rearrange(pw,dict_info)

    results = {'data': data, 'stats1':stats1, 'stats2':stats2}

    return results

def parse_scalar_field(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    start = content.index('(\n', content.index('nonuniform')) + 2
    end   = content.index('\n)', start)
    return np.array([float(v) for v in content[start:end].split()])
    

def read_foam_coords(path, time_folder="5000"):
    return {
        'x_flat': parse_scalar_field(os.path.join(path, time_folder, 'Cx')),
        'y_flat': parse_scalar_field(os.path.join(path, time_folder, 'Cy')),
    }


def write_openfoam_field(filepath, object_name, dimensions_str, location, values):
    """
    Write an OpenFOAM volScalarField or volVectorField (ASCII format).

    Boundary faces are set to zeroGradient (or the topological type where
    required: empty for right/left, cyclicAMI for bottom/top).

    Parameters
    ----------
    filepath       : full output path (e.g. '.../10000/U')
    object_name    : field name string, e.g. 'U' or 'k'
    dimensions_str : OpenFOAM dimension set, e.g. '[0 1 -1 0 0 0 0]'
    location       : time folder string, e.g. '10000'
    values         : (N,) array for scalar fields, (N, 3) for vector fields
    """
    values = np.asarray(values)
    is_vector = values.ndim == 2 and values.shape[1] == 3
    class_name = "volVectorField" if is_vector else "volScalarField"
    n = len(values)

    header = (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| =========                 |                                                 |\n"
        "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
        "|  \\\\    /   O peration     |                                                 |\n"
        "|   \\\\  /    A nd           |                                                 |\n"
        "|    \\\\/     M anipulation  |                                                 |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"    location    \"{location}\";\n"
        f"    object      {object_name};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )

    if is_vector:
        rows = "\n".join(f"({v[0]:.10g} {v[1]:.10g} {v[2]:.10g})" for v in values)
        type_str = "List<vector>"
        zero_bc  = "uniform (0 0 0)"
    else:
        rows = "\n".join(f"{v:.10g}" for v in values)
        type_str = "List<scalar>"
        zero_bc  = "uniform 0"

    internal_field = (
        f"internalField   nonuniform {type_str}\n"
        f"{n}\n"
        "(\n"
        f"{rows}\n"
        ");\n"
    )


    if object_name == 'k':
        bodyBC = "    airfoil\n    {\n        type            fixedValue;\n        value          uniform 0;\n    }\n"
    if object_name == 'U':
        bodyBC = "    airfoil\n    {\n        type            noSlip;\n    }\n"
    if object_name == 'p':
        bodyBC = "    airfoil\n    {\n        type            zeroGradient;\n    }\n"

    boundary_field = (
        "\nboundaryField\n"
        "{\n"
        "    back\n    {\n        type            empty;\n    }\n"
        "    inlet\n    {\n        type            zeroGradient;\n    }\n"
        f"    bottom\n    {{\n        type            zeroGradient;\n        value           {zero_bc};\n    }}\n"
        f"    top\n    {{\n        type            zeroGradient;\n        value           {zero_bc};\n    }}\n"
        "    front\n    {\n        type            empty;\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        f"{bodyBC}"
        "}\n\n"
        "// ************************************************************************* //\n"
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(header)
        f.write(f"dimensions      {dimensions_str};\n\n")
        f.write(internal_field)
        f.write(boundary_field)

    
fs = 9
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": fs,
    'legend.fontsize': fs, 
    "axes.titlesize": fs,
    "axes.labelsize": fs
})


figsize = (5.2, 3.64)
lw = 0.8

pitch = 40.39/1000

dict_input_case = {'pitch': pitch, 'cax': 1, 'x1': -20.108296, 'x2': 87.25188, 'in_blocks': [1, 2], 'out_blocks': [8, 9], 'wall_blocks': [3, 4, 7, 6]}

# input_dir_ADP_pok = "/home/mciarlatani/Irene/validation_cases/deltaP_good/MUSICAA/musicaa_g0_c0/ADP" # "ADP"

# input_dir_OP1_pok = "/home/mciarlatani/Irene/validation_cases/deltaP_good/MUSICAA/musicaa_g0_c0/OP1" # "OP1"

input_dir_MUSICAA = "Ling/MUSICAA/musicaa_g0_c0/OP2" # "OP2"
openFOAM_mesh_directory = 'Ling/cascade_2geo/'

# Read the cell centers of the mesh you want to interpolate to
# Need to have C, Cx, Cy, and Cz from OpenFOAM
x_OF,y_OF = read_openfoam_centers(openFOAM_mesh_directory,'0')

OP2 = read_musicaa_case(input_dir_MUSICAA, dict_input_case)

match_bl = [[10,2,pitch],[11,5,pitch],[12,9,pitch],[13,1,pitch],[14,2,2*pitch]]


for bl, bl_org, pitch in match_bl:
    OP2['data'][bl] = {}
    print(OP2['data'][bl_org].keys())
    for key in OP2['data'][bl_org].keys():
        OP2['data'][bl][key] = OP2['data'][bl_org][key]*1.0
    OP2['data'][bl]['y_flat'] -= pitch
    OP2['data'][bl]['y'] -= pitch

x_flat_LES, y_flat_LES = [], []
u_flat_LES, v_flat_LES, p_flat_LES, TKE_flat_LES = [], [], [], []
y_shape = 0

for bl in range(1,15):
    x_flat_LES.append(OP2['data'][bl]['x_flat'])
    y_flat_LES.append(OP2['data'][bl]['y_flat'])
    u_flat_LES.append(OP2['data'][bl]['u'].flatten())
    v_flat_LES.append(OP2['data'][bl]['v'].flatten())
    p_flat_LES.append(OP2['data'][bl]['p'].flatten())
    TKE_flat_LES.append(OP2['data'][bl]['TKE'].flatten())


x_flat_LES = np.concatenate(x_flat_LES)
y_flat_LES = np.concatenate(y_flat_LES)
u_flat_LES = np.concatenate(u_flat_LES)
v_flat_LES = np.concatenate(v_flat_LES)
p_flat_LES = np.concatenate(p_flat_LES)
TKE_flat_LES = np.concatenate(TKE_flat_LES)

print(f"x and y shapes are: {len(x_flat_LES)}, {len(y_flat_LES)}")

print(np.min(TKE_flat_LES))
input()

# fig, ax = plt.subplots()
# for bl in [3,4,6,7]:
#     ax.scatter(OP2['data'][bl]['x_flat'], OP2['data'][bl]['y_flat'], s=0.5)
# # ax.scatter(x_flat_LES, y_flat_LES, s=0.5, color='tab:grey', label='LES')
# ax.scatter(x_OF, y_OF, s=0.5, color='tab:blue', label='OF')
# ax.set_aspect('equal', adjustable='box')
# plt.show()

# plt.figure()
# plt.scatter(x_flat_LES, y_flat_LES, c=np.sqrt(u_flat_LES**2 + v_flat_LES**2), cmap='viridis', vmin=0, vmax=300, s=0.01)
# plt.show()

krnl = 'thin_plate_spline'
n_near = 80

u_interp   = interpolate_rbf(x_flat_LES, y_flat_LES, u_flat_LES,   x_OF, y_OF, kernel=krnl, neighbors=n_near)
v_interp   = interpolate_rbf(x_flat_LES, y_flat_LES, v_flat_LES,   x_OF, y_OF, kernel=krnl, neighbors=n_near)
p_interp   = interpolate_rbf(x_flat_LES, y_flat_LES, p_flat_LES,   x_OF, y_OF, kernel=krnl, neighbors=n_near)
TKE_interp = np.exp(interpolate_rbf(x_flat_LES, y_flat_LES, np.log(TKE_flat_LES+1e-6), x_OF, y_OF, kernel=krnl, neighbors=n_near))

# Write interpolated fields to OpenFOAM time folder 10000
foam_out = 'Ling/cascade_2geo/0'

U_vec = np.column_stack([u_interp, v_interp, np.zeros(len(u_interp))])
write_openfoam_field(os.path.join(foam_out, 'U'), 'U',
                     '[0 1 -1 0 0 0 0]', '0', U_vec)
write_openfoam_field(os.path.join(foam_out, 'k'), 'k',
                     '[0 2 -2 0 0 0 0]', '0', TKE_interp)
write_openfoam_field(os.path.join(foam_out, 'p'), 'p',
                     '[1 -1 -2 0 0 0 0]', '0', p_interp)
print(f"OpenFOAM fields written to {foam_out}/")

# plt.figure()
# plt.scatter(x_OF, y_OF, c=u_interp, cmap='viridis', vmin=0, vmax=250, s=0.05)
# plt.figure()
# plt.scatter(x_flat_LES, y_flat_LES, c=u_flat_LES, cmap='viridis', vmin=0, vmax=250, s=0.05)
# plt.show()


# var = "TKE"

# vmins, vmaxs = [], []
# for bl in range(1,15):
#     vmins.append(OP2['data'][bl][var].min())
#     vmaxs.append(OP2['data'][bl][var].max())
# vmin, vmax = min(vmins), max(vmaxs)

# fig, ax = plt.subplots(figsize=(3.25, 2.3))
# ax.set_xlabel('$x$ [mm]')
# ax.set_ylabel('$y$ [mm]')

# # for bl in range(1,15):
# #     plt.pcolormesh(OP2['data'][bl]['x'], OP2['data'][bl]['y'], OP2['data'][bl][var], vmin=vmin, vmax=vmax)
# # plt.show()

# comp_data = OP2['data']

# var_list = ['u','v','rhou','rhov','p','rho','omz','x','y']

# # Suction side
# # ============
# for var in var_list:
#     comp_data[f'{var}_flat'] = np.hstack((comp_data['block_1'][f'{var}_flat'],\
#                                           comp_data['block_2'][f'{var}_flat'],\
#                                           comp_data['block_3'][f'{var}_flat'],\
#                                           comp_data['block_4'][f'{var}_flat'],\
#                                           comp_data['block_5'][f'{var}_flat'],\
#                                           comp_data['block_6'][f'{var}_flat'],\
#                                           comp_data['block_7'][f'{var}_flat'],\
#                                           comp_data['block_8'][f'{var}_flat'],\
#                                           comp_data['block_9'][f'{var}_flat']))
    
# LE_idx = np.argmin(comp_data['xw'])

# TE_idx = np.argmax(x_new[:,0])

# plt.figure(figsize=(7,5))
# plt.rcParams['text.usetex'] = True
# plt.plot(comp_data['x_flat'],comp_data['y_flat'],'.r',markersize=0.5)
# plt.plot(x_new,y_new,'.c',markersize=0.5)
# for i in range(20):
#     plt.plot(x_new[int(i*1),:],y_new[int(i*1),:],'k',linewidth=3)

# plt.plot(x_new[TE_idx,:],y_new[TE_idx,:],'g',linewidth=3)
# plt.axis('equal')
# plt.show()