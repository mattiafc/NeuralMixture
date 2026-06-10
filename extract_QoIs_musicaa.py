import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import pandas as pd

from scipy.interpolate import splprep, splev
from scipy.interpolate import Rbf

from musicaa_utils import get_block_info, line_interp, mixed_out, plot_grid, read_grid, read_info, read_stats

def rearrange(x,d):
    len_ = d['nx_bl3']+d['nx_bl4']
    # save bl6 coords
    x_ = np.zeros(d['nx_bl6'])
    x_ = x[len_:len_+d['nx_bl6']].copy()
    # rotate array
    x[len_:len_+d['nx_bl7']] = x[-d['nx_bl7']:].copy()
    x[-d['nx_bl6']:] = x_.copy()

    return x

def read_case(input_dir, dict_input, plot_mesh=False):

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
        data[bl]["x"], data[bl]["y"], data[bl]["z"] = x, y, z

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

    # The sensors coordinates are also extracted for each block

    sensor = {}
    for bl in range(1, n_block + 1):

        sensor[bl] = []
        nb_pt = block_info[f"block_{bl}"]["nb_points"]
        nb_li = block_info[f"block_{bl}"]["nb_lines"]

        if nb_pt > 0:
            for pt in range(1, nb_pt + 1):
                xs = block_info[f"block_{bl}"][f"point_{pt}"]["nx1"]
                ys = block_info[f"block_{bl}"][f"point_{pt}"]["ny1"]
                sensor[bl].append([xs, ys])
                # print(f"point in block {bl} at indexes {xs, ys}")
        if nb_li > 0:
            for li in range(1, nb_li + 1):
                xs = block_info[f"block_{bl}"][f"line_{li}"]["nx1"]
                ys = block_info[f"block_{bl}"][f"line_{li}"]["ny1"]
                sensor[bl].append([xs, ys])
                # print(f"line in block {bl} at indexes {xs, ys}")

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
        data[bl]['v'] = stats1[bl]['v']

        x_flat.append(data[bl]['x'].flatten())
        y_flat.append(data[bl]['y'].flatten())

        if block_info[f'block_{bl}']['wall']:
            if block_info[f'block_{bl}']['jmin']:
                # Wall coordinates
                xw[nw_loc:nw_loc+nx]  = data[bl]['x'][:,0]
                yw[nw_loc:nw_loc+nx]  = data[bl]['y'][:,0]
                # Compute first cell height
                x1_[nw_loc:nw_loc+nx] = data[bl]['x'][:,1]
                y1_[nw_loc:nw_loc+nx] = data[bl]['y'][:,1]
                # Compute wall tangential distance from cell to cell
                dxw[nw_loc:nw_loc+nx] = np.hstack((data[bl]['x'][1:,0]-data[bl]['x'][:-1,0],data[bl]['x'][-2,0]-data[bl]['x'][-1,0]))
                dyw[nw_loc:nw_loc+nx] = np.hstack((data[bl]['y'][1:,0]-data[bl]['y'][:-1,0],data[bl]['y'][-2,0]-data[bl]['y'][-1,0]))
                if bl==6:
                    yw[nw_loc:nw_loc+nx] +=-pitch
                    y1_[nw_loc:nw_loc+nx]+=-pitch
                y1[nw_loc:nw_loc+nx]    = (np.sqrt((x1_-xw)**2+(y1_-yw)**2))[nw_loc:nw_loc+nx]

                # Cf
                rhow[nw_loc:nw_loc+nx] = stats1[bl]['rho'][:,0]
                muw[nw_loc:nw_loc+nx]  = stats1[bl]['mu'][:,0]
                duxw[nw_loc:nw_loc+nx] = stats2[bl]['rho*dux'][:,0]
                duyw[nw_loc:nw_loc+nx] = stats2[bl]['rho*duy'][:,0]
                dvxw[nw_loc:nw_loc+nx] = stats2[bl]['rho*dvx'][:,0]
                dvyw[nw_loc:nw_loc+nx] = stats2[bl]['rho*dvy'][:,0]

                # In cartesian coordinates
                tau[nw_loc:nw_loc+nx,0,0] = (muw*2*duxw/rhow)[nw_loc:nw_loc+nx]
                tau[nw_loc:nw_loc+nx,0,1] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
                tau[nw_loc:nw_loc+nx,1,0] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
                tau[nw_loc:nw_loc+nx,1,1] = (muw*2*dvyw/rhow)[nw_loc:nw_loc+nx]

                # # Cf
                # muw[nw_loc:nw_loc+nx]  = stats1[bl]['mu'][:,0]
                # duxw[nw_loc:nw_loc+nx] = np.sqrt(stats2[bl]['dux**2'][:,0])*np.sign(stats2[bl]['p*dux'][:,0])
                # duyw[nw_loc:nw_loc+nx] = np.sqrt(stats2[bl]['duy**2'][:,0])*np.sign(stats2[bl]['p*duy'][:,0])
                # dvxw[nw_loc:nw_loc+nx] = np.sqrt(stats2[bl]['dvx**2'][:,0])*np.sign(stats2[bl]['p*dvx'][:,0])
                # dvyw[nw_loc:nw_loc+nx] = np.sqrt(stats2[bl]['dvy**2'][:,0])*np.sign(stats2[bl]['p*dvy'][:,0])

                # # Cf
                # muw[nw_loc:nw_loc+nx]  = stats1[bl]['mu'][:,0]
                # duxw[nw_loc:nw_loc+nx] = stats2[bl]['p*dux'][:,0]/stats1[bl]['p'][:,0]
                # duyw[nw_loc:nw_loc+nx] = stats2[bl]['p*duy'][:,0]/stats1[bl]['p'][:,0]
                # dvxw[nw_loc:nw_loc+nx] = stats2[bl]['p*dvx'][:,0]/stats1[bl]['p'][:,0]
                # dvyw[nw_loc:nw_loc+nx] = stats2[bl]['p*dvy'][:,0]/stats1[bl]['p'][:,0]

                # # In cartesian coordinates
                # tau[nw_loc:nw_loc+nx,0,0] = (muw*2*duxw)[nw_loc:nw_loc+nx]
                # tau[nw_loc:nw_loc+nx,0,1] = (muw*(duyw+dvxw))[nw_loc:nw_loc+nx]
                # tau[nw_loc:nw_loc+nx,1,0] = (muw*(duyw+dvxw))[nw_loc:nw_loc+nx]
                # tau[nw_loc:nw_loc+nx,1,1] = (muw*2*dvyw)[nw_loc:nw_loc+nx]

                pw[nw_loc:nw_loc+nx] = stats1[bl]['p'][:,0]

                nw_loc+=nx

    xw = rearrange(xw,dict_info)
    yw = rearrange(yw,dict_info)
    pw = rearrange(pw,dict_info)
    tau = rearrange(tau,dict_info)

    xPlot = copy.deepcopy(xw)
    yPlot = copy.deepcopy(yw)

    # Valutazione spline
    tck, u = splprep([xPlot, yPlot], s=0, k=2, per = True)
    
    # Calcolo normali (rispetto alla curva originale xPlot, yPlot)
    dx, dy = splev(u, tck, der=1)
    norm = np.sqrt(dx**2 + dy**2)
    tx, ty = dx/norm, dy/norm
    nx, ny = -dy/norm, dx/norm

    # u_fine = np.linspace(0, 1, 10000)
    # x_s, y_s = splev(u_fine, tck)
    # # Plot
    # plt.figure(figsize=(8, 6))
    # plt.plot(xPlot[10:20], yPlot[10:20], '.', label='Punti')
    # plt.plot(x_s, y_s, '-', label='Spline')
    # plt.quiver(xPlot[10:20], yPlot[10:20], nx[10:20], ny[10:20], color='red', label='Normali')
    # plt.quiver(xPlot[10:20], yPlot[10:20], ny[10:20], -nx[10:20], color='green', label='Normali')
    # plt.axis('equal')
    # plt.show()

    # print(tau.shape)
    # input()
    tauw = []
    for idx in range(len(tau)):
        val = (tau[idx,0,0]*nx[idx]+tau[idx,0,1]*ny[idx])*tx[idx] + (tau[idx,1,0]*nx[idx]+tau[idx,1,1]*ny[idx])*ty[idx]
        # tauw.append(np.sqrt(val*val))
        tauw.append(val)

    data['cf'] = np.array(tauw)/(0.5*rho_in*(u_in**2))

    # union of data
    in_data = {}
    for bl in in_blocks:
        in_data[bl] = data[bl] | stats1[bl] | stats2[bl]

    out_data = {}
    for bl in out_blocks:
        out_data[bl] = data[bl] | stats1[bl] | stats2[bl]

    # The limits along the y axis of both measurement planes are computed 

    # find y corresponding to x1
    x0 = in_data[in_blocks[0]]["x"]
    closest_index = np.argmin(abs(x0[:, 0] - x1_samp))
    y1_samp = in_data[in_blocks[0]]["y"][closest_index, :].min()
    y2_samp = y1_samp + pitch
    # compute interpolation axis
    y_in = np.linspace(y1_samp, y2_samp, 1000)
    # build inlet_lims
    inlet_lims = [x1_samp, y1_samp, x1_samp, y2_samp]
    # print(f"inlet_lims: {inlet_lims}")

    # find y corresponding to x1
    x0 = out_data[out_blocks[0]]["x"]
    closest_index = np.argmin(abs(x0[:, 0] - x2_samp))
    y1_samp = out_data[out_blocks[0]]["y"][closest_index, :].min()
    y2_samp = y1_samp + pitch
    # compute interpolation axis
    y_out = np.linspace(y1_samp, y2_samp, 1000)
    outlet_lims = [x2_samp, y1_samp, x2_samp, y2_samp]
    # print(f"outlet_lims: {outlet_lims}")

    data_inlet = {}
    data_outlet = {}
    for var in ["uu", "vv", "ww", "rhou", "rhov", "rho*uu", "rho*uv", "rho*uw", "p", "T", "M", "cp", "cv","rho","u","v","w"]:
        data_inlet[f"{var}_interp"], data_inlet['y_interp'] = line_interp(in_data, var, inlet_lims, in_blocks)
        data_outlet[f"{var}_interp"], data_outlet['y_interp'] = line_interp(out_data, var, outlet_lims, out_blocks)

    # print('Length of y and var is', len(data_inlet['y_interp']), len(data_inlet['u_interp']))

    data['gamma']  = np.nanmean(data_inlet["cp_interp"] / data_inlet["cv_interp"])
    # gam1 = gam - 1.0

    data['xw'] = copy.deepcopy(xw)
    data['yw'] = copy.deepcopy(yw)
    data['pw'] = copy.deepcopy(pw)

    in_idx =  np.argwhere(~np.isnan(data_inlet["rhou_interp"]))
    q1 = np.sum(data_inlet["rhou_interp"][in_idx])
    # P1 = np.sum(data_inlet["rhou_interp"][in_idx] * data_inlet["p_interp"][in_idx]) / q1
    P1 = np.mean(data_inlet["p_interp"][in_idx])
    gamma = np.mean(data_inlet["cp_interp"][in_idx] / data_inlet["cv_interp"][in_idx])
    # P01 = np.sum(data_inlet["p_interp"][in_idx] * (1 + (gamma - 1 ) / 2 * data_inlet["M_interp"][in_idx]**2)**(gamma / (gamma - 1)) * data_inlet["rhou_interp"][in_idx]) / q1
    P01 = np.mean(data_inlet["p_interp"][in_idx] * (1 + (gamma - 1 ) / 2 * data_inlet["M_interp"][in_idx]**2)**(gamma / (gamma - 1)) )

    out_idx =  np.argwhere(~np.isnan(data_outlet["rhou_interp"]))
    q2 = np.sum(data_outlet["rhou_interp"][out_idx])
    gamma = np.mean(data_outlet["cp_interp"][out_idx] / data_outlet["cv_interp"][out_idx])
    P02 = np.mean(data_outlet["p_interp"][out_idx] * (1 + (gamma - 1 ) / 2 * data_outlet["M_interp"][out_idx]**2)**(gamma / (gamma - 1)))

    print(f"Loss coefficient w: {(P01 - P02) / (P01 - P1)}; Gamma: {gamma}")

    data['Loss'] = (P01 - data_outlet["p_interp"][out_idx] * (1 + (gamma - 1 ) / 2 * data_outlet["M_interp"][out_idx]**2)**(gamma / (gamma - 1))) / (P01 - P1)
    data['M_is'] = np.sqrt(2/(data['gamma']-1)*((P01/pw)**((data['gamma']-1)/data['gamma'])-1))
    data['P01']  = P01
    data['yLoss'] = y_out[out_idx]

    results = {'data_inlet': data_inlet,
               'data_outlet': data_outlet,
               'stats1': stats1,
               'stats2': stats2,
               'data': data,
               'inlet_lims': inlet_lims,
               'outlet_lims': outlet_lims,
               'y_out': y_out,
               'n_block': n_block,
               'sensor': sensor}

    return results

# plot params
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "Times"
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
figsize = (5.2, 3.64)

path_to_mis = "cascade_mis.dat"

x1 = -20.108296
x2 = 87.25188
pitch = 40.39
in_blocks = [1, 2]
out_blocks = [8, 9]

wall_blocks = [3, 4, 7, 6]
dict_input_case = {'pitch': 40.39, 'cax': 1, 'x1': -20.108296, 'x2': 87.25188, 'in_blocks': [1, 2], 'out_blocks': [8, 9], 'wall_blocks': [3, 4, 7, 6]}

input_dir_ADPb = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/MUSICAA_baseline_Fine/musicaa_g0_c0/ADP" # "ADP"
ADP_b = read_case(input_dir_ADPb, dict_input_case)
print(ADP_b['data'].keys())
xw, yw = ADP_b['data']['xw']/1000, ADP_b['data']['yw']/1000
M_is    = ADP_b['data']['M_is']
pd.DataFrame(ADP_b['data_inlet']).rename(columns=lambda x: x.replace('_interp', '')).to_csv('ADP_Fine_LES_MP1.csv',index=False, sep=',')
pd.DataFrame(ADP_b['data_outlet']).rename(columns=lambda x: x.replace('_interp', '')).to_csv('ADP_Fine_LES_MP2.csv',index=False, sep=',')
pd.DataFrame({'xw': xw.flatten(), 'yw': yw.flatten(), 'M_is': M_is.flatten()}).to_csv('ADP_Fine_LES_Mis.csv',index=False, sep=',')
print('===============================================================')
print(f'Mean Mach at inlet for ADP: {np.nanmean(ADP_b["data_inlet"]["M_interp"])}')
print(f'Mean rho at inlet for ADP : {np.nanmean(ADP_b["data_inlet"]["rho_interp"])}')
print(f'Mean T at inlet for ADP   : {np.nanmean(ADP_b["data_inlet"]["T_interp"])}')
print(f'Mean p at inlet for ADP   : {np.nanmean(ADP_b["data_inlet"]["p_interp"])}')
print('===============================================================')

input_dir_OP1b = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/MUSICAA_baseline_Fine/musicaa_g0_c0/OP1" # "OP1"
OP1_b = read_case(input_dir_OP1b, dict_input_case)
print(OP1_b['data'].keys())
xw, yw = OP1_b['data']['xw']/1000, OP1_b['data']['yw']/1000
M_is   = OP1_b['data']['M_is']
pd.DataFrame(OP1_b['data_inlet']).rename(columns=lambda x: x.replace('_interp', '')).to_csv('OP1_Fine_LES_MP1.csv',index=False, sep=',')
pd.DataFrame(OP1_b['data_outlet']).rename(columns=lambda x: x.replace('_interp', '')).to_csv('OP1_Fine_LES_MP2.csv',index=False, sep=',')
pd.DataFrame({'xw': xw.flatten(), 'yw': yw.flatten(), 'M_is': M_is.flatten()}).to_csv('OP1_Fine_LES_Mis.csv',index=False, sep=',')
print('===============================================================')
print(f'Mean Mach at inlet for OP1: {np.nanmean(OP1_b["data_inlet"]["M_interp"])}')
print(f'Mean rho at inlet for OP1 : {np.nanmean(OP1_b["data_inlet"]["rho_interp"])}')
print(f'Mean T at inlet for OP1   : {np.nanmean(OP1_b["data_inlet"]["T_interp"])}')
print(f'Mean p at inlet for OP1   : {np.nanmean(OP1_b["data_inlet"]["p_interp"])}')
print('===============================================================')

input_dir_OP2b = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/MUSICAA_baseline_Fine/musicaa_g0_c0/OP2" # "OP2"
OP2_b = read_case(input_dir_OP2b, dict_input_case)
xw, yw = OP2_b['data']['xw']/1000, OP2_b['data']['yw']/1000
M_is   = OP2_b['data']['M_is']
pd.DataFrame(OP2_b['data_inlet']).rename(columns=lambda x: x.replace('_interp', '')).to_csv('OP2_Fine_LES_MP1.csv',index=False, sep=',')
pd.DataFrame(OP2_b['data_outlet']).rename(columns=lambda x: x.replace('_interp', '')).to_csv('OP2_Fine_LES_MP2.csv',index=False, sep=',')
pd.DataFrame({'xw': xw.flatten(), 'yw': yw.flatten(), 'M_is': M_is.flatten()}).to_csv('OP2_Fine_LES_Mis.csv',index=False, sep=',')
print('===============================================================')
print(f'Mean Mach at inlet for OP2: {np.nanmean(OP2_b["data_inlet"]["M_interp"])}')
print(f'Mean rho at inlet for OP2 : {np.nanmean(OP2_b["data_inlet"]["rho_interp"])}')
print(f'Mean T at inlet for OP2   : {np.nanmean(OP2_b["data_inlet"]["T_interp"])}')
print(f'Mean p at inlet for OP2   : {np.nanmean(OP2_b["data_inlet"]["p_interp"])}')
print('===============================================================')

# print(ADP_b['data_inlet'].keys())

# for c in ADP_b:
#     print(f"{c}: {len(ADP_b[c])}")


# =============================================================================
# 1. Data processing
# =============================================================================

# # input_dir_ADP1 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_0/high_infill_0/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
# # input_dir_OP1 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_0/high_infill_0/MUSICAA/musicaa_g0_c0/OP1" # "OP1"
# # input_dir_OP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_0/high_infill_0/MUSICAA/musicaa_g0_c0/OP2" # "OP2"

# # input_dir_ADP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_6/high_infill_6/MUSICAA/musicaa_g0_c0/OP1" # "ADP"
# # input_dir_OP1 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_6/high_infill_6/MUSICAA/musicaa_g0_c0/OP1" # "OP1"
# # input_dir_OP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_6/high_infill_6/MUSICAA/musicaa_g0_c0/OP2" # "OP2"

# # input_dir_ADP3 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_3/high_infill_3/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
# # input_dir_OP1 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_3/high_infill_3/MUSICAA/musicaa_g0_c0/OP1" # "OP1"
# # input_dir_OP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_3/high_infill_3/MUSICAA/musicaa_g0_c0/OP2" # "OP2"

# # input_dir_ADP3 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_3/high_infill_3/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
# # input_dir_ADPb = "/home/mciarlatani/Hilbert/aero-optim/examples/LRN-CASCADE/cascade_musicaa_base/output_baseline/MUSICAA/musicaa_g0_c0/ADP" # "ADP"

# input_dir_ADP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_0/high_infill_0/MUSICAA/musicaa_g0_c0/OP1" # "ADP"

# # ADP_1 = read_case(input_dir_ADP1, in_blocks, out_blocks)
# ADP_2 = read_case(input_dir_ADP2, in_blocks, out_blocks)
# # ADP_3 = read_case(input_dir_ADP3, in_blocks, out_blocks)
# # ADP_b = read_case(input_dir_ADPb, in_blocks, out_blocks)

# # out_idx_ADP1, loss_ADP1, _, _ = compute_total_pressure(ADP_1['data'], ADP_1['stats1'], ADP_1['stats2'], ADP_1['inlet_lims'], ADP_1['outlet_lims'], ADP_1['n_block'])
# out_idx_ADP2, loss_ADP2, P01_ADP2, gamma = compute_total_pressure(ADP_2['data'], ADP_2['stats1'], ADP_2['stats2'], ADP_2['inlet_lims'], ADP_2['outlet_lims'], ADP_2['n_block'])
# # out_idx_ADP3, loss_ADP3, _, _ = compute_total_pressure(ADP_3['data'], ADP_3['stats1'], ADP_3['stats2'], ADP_3['inlet_lims'], ADP_3['outlet_lims'], ADP_3['n_block'])
# # out_idx_ADPb, loss_ADPb, _, _ = compute_total_pressure(ADP_b['data'], ADP_b['stats1'], ADP_b['stats2'], ADP_b['inlet_lims'], ADP_b['outlet_lims'], ADP_b['n_block'])

# fig, ax = plt.subplots(figsize=figsize)
# # ax.plot(ADP_1['y_out'][out_idx_ADP1] / 1000, loss_ADP1, label="LES 1")
# ax.plot(ADP_2['y_out'][out_idx_ADP2] / 1000, loss_ADP2, label="LES 2")
# # ax.plot(ADP_3['y_out'][out_idx_ADP3] / 1000, loss_ADP3, label="LES 3")
# # ax.plot(ADP_b['y_out'][out_idx_ADPb] / 1000, loss_ADPb, label="Baseline")
# ax.legend()
# ax.set_xlabel("$\\bar{y}$")
# ax.set_ylabel("$w$ [-]")
# plt.show()

# # # The loss data is saved

# # outfile = f"loss_{output_key}.csv"
# # loss_df = pd.DataFrame(np.column_stack([y_out[out_idx] / 1000, loss]), columns=["y", "loss"])
# # # loss_df.to_csv(outfile, index=False)

# # The data is extracted along the wall

# pres_wall_list = []
# x_list = []
# y_list = []

# for bl in wall_blocks:
#     new_pres_value = ADP_2['stats1'][bl]['p'][:, 0]
#     pres_wall_list.append(new_pres_value)
    
#     new_x_value = ADP_2['data'][bl]['x'][:, 0]
#     x_list.append(new_x_value)

#     new_y_value = ADP_2['data'][bl]['y'][:, 0]
#     y_list.append(new_y_value)

    
# pres_wall = np.concatenate(pres_wall_list)
# x_wall = np.concatenate(x_list) / 1000.
# y_wall = np.concatenate(y_list) / 1000.
# # x_tilde = np.cos(np.arctan(y_wall / x_wall) - (106.04 - 90) / 180 * np.pi) * np.sqrt(x_wall**2 + y_wall**2)

# # The isentropic Mach is computed

# Mach_is = np.sqrt(((P01_ADP2 / pres_wall)**((gamma - 1) / gamma) - 1) * 5)

# exp_mis = np.loadtxt(path_to_mis, skiprows=1)

# fig, ax = plt.subplots(figsize=figsize)
# ax.plot(x_wall / 0.067, Mach_is, color="blue", label="LES")
# ax.scatter(exp_mis[:, 0] / 0.067, exp_mis[:, 1], color="k", label="DLR exp.")
# ax.legend()
# ax.set_ylim(0.3, 1.)
# ax.set_xlabel('$\\bar{x}$ [-]')
# ax.set_ylabel('$Mis$ [-]')

# plt.show()

# # The isentropic Mach data is saved

# # outfile = f"mis_{output_key}.csv"
# # mis_df = pd.DataFrame(np.column_stack([x_wall, y_wall, x_wall / 0.067, Mach_is]), columns=["x", "y", "x/cax", "mis"])
# # # mis_df.to_csv(outfile, index=False)

# # =============================================================================
# # 4. Outflow/inflow angles
# # =============================================================================

# u_mean = np.nanmean(data_outlet["rhou_interp"])
# v_mean = np.nanmean(data_outlet["rhov_interp"])
# print(f"Outflow angle: {np.atan(v_mean / u_mean) / np.pi * 180} deg.")

# u_mean = np.nanmean(data_inlet["rhou_interp"])
# v_mean = np.nanmean(data_inlet["rhov_interp"])
# print(f"Inflow angle: {np.atan(v_mean / u_mean) / np.pi * 180} deg.")

# angle = np.nanmean(np.arctan(data_outlet["rhov_interp"] / data_outlet["rhou_interp"]))
# print(f"Outflow angle: {angle / np.pi * 180} deg.")

# # =============================================================================
# # Paraview stats (extra step for paper)
# # =============================================================================

# from musicaa_utils import write_para
# para_dir = "ADP_stats_para"
# os.makedirs(os.path.join(input_dir, para_dir, "par_planes"))
# write_para(input_dir, para_dir, plane_nb=-1, var_names="", stats=True)

# plt.show()
