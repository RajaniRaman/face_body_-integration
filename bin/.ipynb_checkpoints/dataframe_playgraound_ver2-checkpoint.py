
# %%
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy import io

# %%
input_imsize = 500  # 227 , 500
IregMinReg = False
networks = ['alexnet', 'vgg16', 'alexnet_caffe']
untrained_nets = [False, True]
layers = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'Layer7']
# std_vals = [1.5, 2, 2.5, 3, 3.5, 4]
std_vals = [1, 2, 3]
# networks = ["alexnet"]

mat_path_trained = f'../results/data/net_resp/alexnet_trained_resp_from_matlab_{input_imsize}.mat'
mat_path_untrained = f'../results/data/net_resp/alexnet_untrained_resp_from_matlab_{input_imsize}.mat'

# -- loaind the dataframe containg stimuli information
df = pd.read_csv('df_stim.csv')
df.reset_index(inplace=True)

# ---- finding Monkey-selective units ----


def getActiveIndex(df_exp, std_val):
    # df_exp = df_Mon.explode('resp')
    mon_resp = df_exp[(df_exp['stim'] == 'Mon')]
    mon_resp_new = mon_resp.pivot_table('resp', ['units'], 'type')

    mon_resp_cl = mon_resp_new.copy()
    mon_resp_cl = mon_resp_cl[mon_resp_cl.any(axis=1)]  # removeing zeros
    mon_resp_cl['reg-ireg'] = mon_resp_cl['R'].values - \
        mon_resp_cl['Irr'].values
    mon_resp_cl['ireg-reg'] = mon_resp_cl['Irr'].values - \
        mon_resp_cl['R'].values
    mon_resp_cl

    idx_selected = mon_resp_cl[mon_resp_cl['reg-ireg']
                               >= mon_resp_cl['reg-ireg'].std()*std_val].index
    return idx_selected.values

# %%


def get_activities_dataframes(resp, net, layer, std_val, untrained_net, df):

    # craing respons tupple for each stimulus
    r_tuple = [(resp[i]) for i in range(120)]
    df['resp'] = r_tuple  # adding tupple array to the dataframe
    df['network'] = net
    df['std_val'] = std_val
    df['layer'] = layer
    df['nUnits'] = resp.shape[1]
    df['trained'] = not(untrained_net)

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
    # df_exp = df_exp.reset_index(inplace=False)
    # df_exp = df_exp.drop('level_0', 1)

    # ---- dataframes with selected units's response
    unitIdx_selected = getActiveIndex(df_exp, std_val)

    df_exp_sel = df_exp[df_exp['units'].isin(unitIdx_selected)].copy()
    df_sum_exp_sel = df_sum_exp[df_sum_exp['units'].isin(
        unitIdx_selected)].copy()

    df_exp_sel['nSelUnits'] = unitIdx_selected.shape[0]
    df_sum_exp_sel['nSelUnits'] = unitIdx_selected.shape[0]
    # df_exp_sel = df_exp_sel.drop('level_0', 1)
    # df_sum_exp_sel = df_sum_exp_sel('level_0', 1)
    return df_exp_sel, df_sum_exp_sel


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


def annotate(data, **kws):
    n = len(data)
    ax = plt.gca()
    ax.text(.1, .6, f"N = {n}", transform=ax.transAxes)

# %%


def annotate_nUnits(g, data, **kws):
    # N = tuple(data.nUnits.unique().astype(int))
    N = data[data.trained == True].nSelUnits.unique()
    id = 0
    # frac_N = np.round(
    #     (np.divide(data.loc[0].nUnits.values,  data.loc[0].nAllUnits.values))*100, 2)
    for ax in g.axes.flat:
        ax.text(0.05, 0.90, f'n={N[id]}', fontsize=9,
                transform=ax.transAxes)  # add text
        id += 1


# %%
df_all_all = []
df_sum_all = []
result_dir = Path('../results/')
# std_vals_tq = tqdm(std_vals)
# networks_tq = tqdm(networks)
# layers_tq = tqdm(layers)
for std_val in tqdm(std_vals, desc="std_val", position=0, leave=False):
    # std_vals_tq.set_description('Processing for std value %s' % std_val)

    for network in tqdm(networks, desc=" network", position=1, leave=False):
        # networks_tq.set_description('Processing %s' % network)

        for untrained_net in untrained_nets:
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

            for layer in tqdm(layers, desc=" layer", position=2, leave=False):
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
                    rout_reshaped, network, layer, std_val, untrained_net, df)
                df_all_all.append(df_all)
                df_sum_all.append(df_sum)

print("done!")
# %%
df_all_all = pd.concat(df_all_all)
df_sum_all = pd.concat(df_sum_all)

# %%
df_all_all.to_csv('../results/data/dataframes/grand_df_all.csv')
df_sum_all.to_csv('../results/data/dataframes/grand_df_sum.csv')

# %%
df_exp_sel_mean = df_sum_all.groupby(
    ['stim', 'type', 'layer', 'network', 'std_val', 'trained', 'units'])['resp'].mean()

# %%
df_exp_sel_mean_df = pd.DataFrame({'val': df_exp_sel_mean})
df_exp_sel_mean_df = df_exp_sel_mean_df.reset_index()

# %%

palette = sns.color_palette("muted", n_colors=2)
palette.reverse()

g = sns.FacetGrid(data=df_exp_sel_mean_df.reset_index(), col='trained',
                  row='std_val', height=2, aspect=3, margin_titles=True, sharey=False)
g.map_dataframe(sns.lineplot, x='layer', y='val', hue='type', marker="o",
                style='stim', ci=None, err_style='bars', palette=palette)
g.add_legend()
# g.fig.suptitle(f'Untrained (Im_size = {input_imsize}); {unit_type}')
g.fig.subplots_adjust(top=.85)
# g.map_dataframe(annotate)
# g.set_titles(col_template='{col_name}')
# annotate_nUnits(g, df_sum_all)
g.fig.savefig(f'../results/plots/mean_resp_alexnet_stds.png')
# %%
