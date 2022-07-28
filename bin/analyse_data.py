
# %%
import pickle
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy import io

# %%
input_imsize = 500  # 227 , 500
IregMinReg = False
networks = ['alexnet', 'vgg16', 'alexnet_caffe']
untrained_nets = [False, True]
layers = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'Layer7']
reg_sels = [True, False]
std_vals = [1, 2, 3]

mat_path_trained = f'../results/data/net_resp/alexnet_trained_resp_from_matlab_{input_imsize}.mat'
mat_path_untrained = f'../results/data/net_resp/alexnet_untrained_resp_from_matlab_{input_imsize}.mat'

# -- loaind the dataframe containg stimuli information
df = pd.read_csv('../data/df_stim.csv')
df.reset_index(inplace=True)

# ---- finding Monkey-selective units ----


def extract_activeIdx(resp, df, std_vals, reg_sel):
    mon_idx = df[df.stim == 'Mon'].index
    mon_resp = resp[mon_idx]

    unit_id = np.arange(resp.shape[1])
    zero_idx = ~np.all(mon_resp == 0, axis=0)
    unit_id_id = unit_id[zero_idx]
    resp_r = mon_resp[:, zero_idx]

    # mon_reg_idx = df[(df.stim == 'Mon') & (df.type == 'R')].index
    # mon_ireg_idx = df[(df.stim == 'Mon') & (df.type == 'Irr')].index

    mon_reg_resp = resp_r[20:]
    mon_ireg_resp = resp_r[:20]
    if reg_sel:
        mon_diff = mon_reg_resp - mon_ireg_resp
    else:
        mon_diff = mon_ireg_resp - mon_reg_resp

    mon_diff_mean = mon_diff.mean(axis=0)
    active_idx = np.where(mon_diff_mean >= mon_diff_mean.std()*std_vals)[0]
    return unit_id_id[active_idx]


# %%

def get_activities_dataframes(resp, net, layer, std_val, untrained_net, reg_sel, df):
    active_idx = extract_activeIdx(resp, df, std_val, reg_sel)
    active_resp = resp[:, active_idx]

    # craing respons tupple for each stimulus
    r_tuple = [(active_resp[i]) for i in range(120)]
    df['resp'] = r_tuple  # adding tupple array to the dataframe
    df['network'] = net
    df['std_val'] = std_val
    df['layer'] = layer
    df['nUnits'] = resp.shape[1]
    df['trained'] = not(untrained_net)
    df['nSelUnits'] = active_resp.shape[1]
    df['nUnits'] = resp.shape[1]
    df['reg_sel'] = reg_sel

    # --- creating datframe for Mon ans Sum -------
    df_MBody = df[df['stim'] == 'MBody']
    df_Mon = df[df['stim'] == 'Mon']
    df_MFace = df[df['stim'] == 'MFace']

    df_sum_ = df_MBody.copy()
    df_sum_['resp'] = df_MBody.resp.values + \
        df_MFace.resp.values  # summing face and body response
    df_sum_['stim'].replace({'MBody': 'sum'}, inplace=True)
    df_sum = pd.concat([df_sum_, df_Mon], ignore_index=True)

    # -- exploded dataframes with rows as unit's response
    df_sum_exp = df_sum.copy()
    df_sum_exp = df_sum_exp.explode('resp')
    df_sum_exp['units'] = df_sum_exp.groupby(
        'index')['resp'].cumcount()  # Enumerate Groups

    df_exp = df.copy()
    df_exp = df_exp.explode('resp')
    df_exp['units'] = df_exp.groupby(
        'index')['resp'].cumcount()  # Enumerate Groups

    return df_exp, df_sum_exp


def activities_from_mat(mat_path, layers):
    resp_mlab = io.loadmat(mat_path)
    resp_mlab = resp_mlab['face_resp']
    resp_dict = {}
    idx = 0
    # layers = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'Layer7']
    for key in layers:
        resp_dict[key] = resp_mlab[idx][0]
        idx += 1
    return resp_dict


# %%
df_all_all = []
df_sum_all = []
result_dir = Path('../results/')
# std_vals_tq = tqdm(std_vals)
# networks_tq = tqdm(networks)
# layers_tq = tqdm(layers)
for reg_sel in tqdm(reg_sels, desc="reg/ireg", position=0, leave=False):
    for std_val in tqdm(std_vals, desc="std_val", position=1, leave=False):
        for network in tqdm(networks, desc=" network", position=2, leave=False):
            for untrained_net in tqdm(untrained_nets, desc="trained/untrained", position=3, leave=False):
                if network == 'alexnet_caffe':
                    if untrained_net:
                        out = activities_from_mat(mat_path_untrained, layers)
                    else:
                        out = activities_from_mat(mat_path_trained, layers)

                else:
                    resp_name = f'{network}_trained_{not(untrained_net)}_inputsize_{input_imsize}.pkl'
                    resp_dir = result_dir / 'data' / 'net_resp' / resp_name
                    with open(resp_dir, 'rb') as f:
                        out = pickle.load(f)

                for layer in tqdm(layers, desc=" layer", position=4, leave=False):
                    # layers_tq.set_description('Processing %s' % layer)

                    resp_out = out[layer]
                    # subtracting backgroud response
                    resp_out_sub = resp_out - resp_out[-1]

                    if network != 'alexnet_caffe':
                        # getting activation as rows
                        rout = resp_out_sub.detach().numpy()
                        rout_reshape = rout.reshape(121, -1)
                    else:
                        rout_reshape = resp_out_sub

                    # leaving out the background response
                    rout_reshaped = rout_reshape[:-1]
                    # print(layer)

                    df_all, df_sum = get_activities_dataframes(
                        rout_reshaped, network, layer, std_val, untrained_net, reg_sel, df)
                    df_all_all.append(df_all)
                    # df_sum_all.append(df_sum) # not neccesary anymore 

print("calculatin done!")
# %%
df_all_all = pd.concat(df_all_all)
# df_sum_all = pd.concat(df_sum_all)

# %%
df_all_all.to_csv('../results/data/dataframes/grand_df_all.csv')
# df_sum_all.to_csv('../results/data/dataframes/grand_df_sum.csv')

print("saving done!")
