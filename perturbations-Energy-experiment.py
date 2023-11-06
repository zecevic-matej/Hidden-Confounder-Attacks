# Paper published at ACML 2023 Journal Track
# Link to Conference: https://www.acml-conf.org/2023/papers.html
# corresponding author: Matej Zecevic, matej.zecevic@tu-darmstadt.de
#
# Reproducing the Energy example from the paper
import torch
import perturbations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid", {'axes.grid' : False})
import os
from Model import getKPI, HouseModel, getSettings
from Model_with_Gas import getKPI_Gas, HouseModel_Gas, getSettings_Gas
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data = {'CostPV': [], 'CostBat': [], 'CostBuy': [], 'Demand': [], 'CapPV': [], 'CapBat': [], 'OwnGen': [], 'TOTEX': [], 'CAPEX': [], "GasBought": []}
hm = HouseModel(getSettings())
hm_g = HouseModel_Gas(getSettings_Gas())

def compute_energy_model_solution(w, gas=False):
    if gas:
        mm = hm_g
        kpi = getKPI_Gas
        input_scales = {"CostPV": 100000, "CostBat": 1, "CostBuy": 1, "Demand": 1, "CostGas": 1}
    else:
        mm = hm
        kpi = getKPI
        input_scales = {"CostPV": 100000, "CostBat": 1, "CostBuy": 1, "Demand": 1}
    if isinstance(w, torch.Tensor) or isinstance(w, np.ndarray):
        d = {"cost_PV": w[0].item(), "cost_Battery": w[1].item(), "cost_buy": w[2].item(), "dem_tot": w[3].item()}
        if gas:
            d.update({"cost_gas": w[4].item()})
        result, status = mm.sample_model(input_changes=d, input_scales=input_scales)
    else:
        d = {"cost_PV": w[0], "cost_Battery": w[1], "cost_buy": w[2], "dem_tot": w[3]}
        if gas:
            d.update({"cost_gas": w[4]})
        result, status = mm.sample_model(input_changes=d, input_scales=input_scales)
    kpi = kpi([result, status])
    if gas:
        r = torch.tensor([kpi['Cap_PV'], kpi['Cap_Bat'], kpi['Own_Gen'], kpi['TOTEX'], kpi['CAPEX'], kpi['Gas_Bought'], kpi['Total_Cost'], kpi['Ele_Bought']])
    else:
        r = torch.tensor([kpi['Cap_PV'], kpi['Cap_Bat'], kpi['Own_Gen'], kpi['TOTEX'], kpi['CAPEX']])
    return r
def solve_EM_loop(w):
    results = []
    assert (len(w.shape) == 3)
    for g in w:
        results.append(compute_energy_model_solution(g[0,:], gas=False))
    r = torch.stack(results)
    print(f'\nThe Perturbed w:\n{w}\nThe Results:\n{r}')
    return r


def euclidean(a, b):
    return torch.cdist(a, b)

def compute_gradient(x, px, pw, grad_method, norm_function):
    if grad_method == 'norm':
        norm = norm_function(x[None,:], px[None,:])
        if len(norm.shape) > 0:
            norm = norm[0]
        norm.backward(retain_graph=True)
        norm_grad = pw.grad
        grad = norm_grad
        title = f'Gradient of\nNorm {norm_function.__name__}(x_opt(w), x_opt(z))\nw.r.t. Cost Matrix z'
    elif grad_method == 'cost':
        cost = pw.flatten() @ px
        cost.backward(retain_graph=True)
        cost_grad = pw.grad
        title = f'Gradient of\nCost w @ x_opt(w)\nw.r.t. Cost Matrix w'
        grad = cost_grad
    elif grad_method == 'targeted':
        target = 2
        px[target].backward(retain_graph=True)
        targeted_grad = pw.grad
        title = f'Gradient of\nx_opt(w)_[{target}]\nw.r.t. Cost Matrix w'
        grad = targeted_grad
    elif grad_method == 'counting':
        count = px.sum()
        count.backward(retain_graph=True)
        count_grad = pw.grad
        title = f'Gradient of\nCount sum(x_opt(w))\nw.r.t. Cost Matrix w'
        grad = count_grad
    else:
        raise Exception(f'Method {grad_method} not defined.')
    pw.grad = None
    return grad, title


# def plot_gradient(grad, title, save=None, no_text=False):
#     fig = plt.figure(figsize=(9,9))
#     a = plt.gca()
#     a.imshow(grad, vmin=-1, vmax=1)
#     if not no_text:
#         for (j, k), label in np.ndenumerate(grad):
#             val = np.round(grad[k][j].item(), decimals=2)
#             a.text(j, k, val, ha='center', va='center', color='black', fontsize=9)
#     a.get_yaxis().set_visible(False)
#     a.get_xaxis().set_visible(False)
#     plt.title(title)
#     if save is not None:
#         if not os.path.exists(save):
#             os.makedirs(save)
#         p = os.path.join(save, f'_Gradient.png')
#         plt.savefig(p)
#         print(f'Saved to {p}')
#     else:
#         plt.show()
#         plt.clf()
#         plt.close()

# def normalize(x, a, b):
#     ma = np.max(x)
#     mi = np.min(x)
#     return (b-a)*((x-mi)/(ma-mi))+a

w1 = [0.001, 300., 0.25, 3000., 0.25]#[1000., 300., 0.25, 3000., 0.06]
x1 = compute_energy_model_solution(w1, gas=True)
w2 = [0.0008, 300., 0.25, 3000., 0.25]
x2 = compute_energy_model_solution(w2, gas=True)
w3 = [0.0012, 300., 0.25, 3000., 0.25]
x3 = compute_energy_model_solution(w3, gas=True)
w4 = [0., 300., 0.25, 3000., 0.25]
x4 = compute_energy_model_solution(w4, gas=True)
w5 = [0.1, 300., 0.25, 3000., 0.25]
x5 = compute_energy_model_solution(w5, gas=True)
w6 = [0.01, 300., 0.25, 3000., 0.25]
x6 = compute_energy_model_solution(w6, gas=True)
w7 = [0.005, 300., 0.25, 3000., 0.25]
x7 = compute_energy_model_solution(w7, gas=True)
w8 = [0.00002, 300., 0.25, 3000., 0.25]
x8 = compute_energy_model_solution(w8, gas=True)
w9 = [0.0001, 300., 0.25, 3000., 0.25]
x9 = compute_energy_model_solution(w9, gas=True)
for x,w in zip([x5, x6, x7, x3, x1, x2, x4, x9, x8],[w5, w6, w7, w3, w1, w2, w4, w9, w8]):
    # print(f"CostPV {w[0]:.5f} - Cap_PV {x[0]:.3f} Cap_Bat {x[1]:.3f} Own_Gen {x[2]:.3f} TOTEX {x[3]:.3f}"
    #       f" CAPEX {x[4]:.3f} Gas_Bought {x[5]:.3f} Ele_Bought (h) {x[7]:.3f} Total_Cost {x[6]:.3f}")
    print(f"{x[0]:.2f} & {x[1]:.2f} & {x[2]:.2f} & {x[3]:.2f}"
          f" & {x[4]:.2f} & {x[5]:.2f} & {x[7]:.2f} & {w[0]:.5f}")

# w = [1000., 300., 0.25, 3000., 0.06]
# x = compute_energy_model_solution(w, gas=True)
#
# xs = []
# for i in range(3):
#     w = [1000, 300., 0.25, 3000., 0.05+i*.1]
#     x = compute_energy_model_solution(w, gas=True)
#     t = (w, x)
#     xs.append(t)
#     print(t)

# example perturbations
# note how the third columns values have been changed drastically, important!
# this has not happened before because in LA/SP the dimensions *have the same meaning!*
# W = np.array([[[  1000.2217,    299.8275,      0.5610,   2999.3872]],
#         [[   999.6596,    299.8254,      1.0448,   3000.7312]],
#         [[   999.8769,    300.1574,      1.0579,   3002.9150]],
#         [[  1000.5123,    300.2842,      1.1073,   2999.7708]],
#         [[  1000.3248,    299.6089,     -0.0641,   2999.8247]]])
# xs = []
# for w in W:
#     x = compute_energy_model_solution(w[0,:])
#     xs.append(x)
# print(f'\nFed w:\n{W}\nThe Results:\n{xs}')

# with perturbations
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

pert_ilp = perturbations.perturbed(solve_EM_loop,
                                      num_samples=5,
                                      sigma=0.05,
                                      noise='gumbel',
                                      batched=False,
                                      device=device)
pw = torch.tensor(np.array(w)[None,:], requires_grad=True, dtype=torch.float)
px = pert_ilp(pw)


grad_method = 'norm'
norm_function = euclidean
grad, title = compute_gradient(x, px, pw, grad_method, norm_function)
print(grad)

#
# w_adv = w.copy()
# indices = w_adv > 0
# w_adv[indices] += 0.01 * np.sign(grad.detach().numpy()[indices])
#
# x_adv = la.solve_LA_ILP(w_adv)
# edges_adv, weights_adv, mat_adv = la.convert_x_to_edges(x_adv)
#
# save = 'Adv-LA-example-Vaccine/'
# la.plot_bipartite_matching(w, mat, 'Regular', color='blue', scale=1, save=save)
# plot_gradient(grad, title, save=save, no_text=True)
# la.plot_bipartite_matching(w_adv, mat_adv, 'Adversarial', color='red', scale=1, save=save)
#
#
# lin2mat = {}
# mat2lin = {}
# for i, r in enumerate(w):
#     for j, c in enumerate(r):
#         lin2mat.update({(i * len(r)) + j: (i, j)})
#         mat2lin.update({(i, j): (i * len(r)) + j})
# mask_lin = np.where(w.flatten() > 0)[0].tolist()
# mask_mat = [lin2mat[x] for x in mask_lin]
# indices = np.where(mat.flatten() > 0)[0]
# indices_adv = np.where(mat_adv.flatten() > 0)[0]
# indices = dict([lin2mat[g] for g in indices])
# indices_adv= dict([lin2mat[g] for g in indices_adv])
# assert(len(hs) == len(indices))
# assert(len(hs) == len(indices_adv))
# def plot_hidden(xs, ys, sort_indices, mat_indices, vac_spots, color, title=None, save=None):
#     plt.figure(figsize=(8,5))
#     plt.bar(xs, ys[sort_indices], color='gray', alpha=0.5)
#
#     pos2id = []
#     id2pos = []
#     for i in range(len(xs)):
#          pos2id.append((i, xs[sort_indices][i]))
#          id2pos.append((xs[sort_indices][i], i))
#     pos2id = dict(pos2id)
#     id2pos = dict(id2pos)
#
#     do_vac = []
#     no_vac = []
#     for i in xs[sort_indices]:
#         try:
#             vac_spots.index(mat_indices[i])
#             do_vac.append((id2pos[i], ys[i]))
#         except:
#             no_vac.append((id2pos[i], ys[i]))
#     ys_do_vac = np.zeros(xs.shape)
#     for t in do_vac:
#         ys_do_vac[t[0]] = t[1]
#     ys_no_vac = np.zeros(xs.shape)
#     for t in no_vac:
#         ys_no_vac[t[0]] = t[1]
#
#     #plt.bar(xs, ys_no_vac, color=color, alpha=0.5)
#     plt.bar(xs, ys_do_vac, color=color, alpha=0.75)
#
#     plt.xticks(xs, xs[sort_indices])
#     plt.tight_layout()
#     if save is not None and title:
#         if not os.path.exists(save):
#             os.makedirs(save)
#         p = os.path.join(save, f'_HM_{title}.png')
#         plt.savefig(p)
#         print(f'Saved to {p}')
#     else:
#         plt.show()
#         plt.clf()
#         plt.close()
#     return do_vac, ys_do_vac
# vaccination_spots = list(np.arange(10,20))
# hs_sort_indices = np.argsort(hs)
# hs_b_1, hs_b_2 = plot_hidden(np.arange(len(hs)), hs, hs_sort_indices, indices, vaccination_spots, 'blue', 'Regular', save)
# hs_a_1, hs_a_2 = plot_hidden(np.arange(len(hs)), hs, hs_sort_indices, indices_adv, vaccination_spots, 'red', 'Adversarial', save)