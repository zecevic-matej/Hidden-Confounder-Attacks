# Paper published at ACML 2023 Journal Track
# Link to Conference: https://www.acml-conf.org/2023/papers.html
# corresponding author: Matej Zecevic, matej.zecevic@tu-darmstadt.de
#
# Reproducing the Shortest Path example from the paper
import torch
import torch.nn.functional as F
import perturbations
import numpy as np
from mip import Model, xsum, maximize, minimize, BINARY, OptimizationStatus
import string
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid", {'axes.grid' : False})
import networkx as nx
import pickle
import os
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.set_cmap('RdBu')


class ILP_SP:
    def __init__(self, s, t):
        self.m = Model("LA")
        self.names = {'alphas': string.ascii_uppercase}
        self.s = s
        self.t = t
        print(f'Shortest Path Problem (SP problem) with Source {self.s} and Target {self.t}')

    def solve_SP_ILP_loop(self, w):
        results = []
        assert (len(w.shape) == 3)
        for g in w:
            results.append(self.solve_SP_ILP(g))
        return torch.stack(results)

    def solve_SP_ILP(self, w):
        self.A = self.B = range(len(w))
        self.x = [[self.m.add_var(var_type=BINARY) for j in self.B] for i in self.A]

        self.m.objective = minimize(xsum(w[i][j].item() * self.x[i][j] for i in self.A for j in self.B))

        for i in self.A:
            if i == self.s:
                constr_val = 1
            elif i == self.t:
                constr_val = -1
            else:
                constr_val = 0
            self.m += (xsum(self.x[i][j] for j in self.B) - xsum(self.x[j][i] for j in self.B)) == constr_val
            self.m += xsum(self.x[i][j] for j in self.B) <= 1

        status = self.m.optimize()
        if status == OptimizationStatus.OPTIMAL:
            print(f"Optimal Solution found with Gain {self.m.objective_value:.2f}\n"
                  f"Optimal Assignment {[item.x for sublist in self.x for item in sublist]}:")
            self.edges = []
            self.selected_cost_matrix = np.inf * np.ones(np.array(self.x).shape)
            self.decision_matrix = np.zeros(np.array(self.x).shape)
            for i in self.A:
                for j in self.B:
                    if self.x[i][j].x:
                        print(f"{self.names['alphas'][i]} - {j}")
                        self.edges.append((self.names['alphas'][i], self.names['alphas'][j]))
                        self.selected_cost_matrix[i][j] = w[i][j].item()
                        self.decision_matrix[i][j] = 1
        else:
            self.edges = False
        return torch.from_numpy(self.decision_matrix.flatten())

    def plot_SP_graph(self, w, edges, weights, lin2mat, mat, seed, names=None, pos=None, ax=None, color='red'):
        G = nx.from_numpy_matrix(w, create_using=nx.DiGraph)
        if pos is None:
            pos = nx.spring_layout(G, seed=seed) #nx.random_layout(G, seed=seed)
        nx.draw(G, pos, node_color='k', ax=ax)
        if names:
            labels = {n: names[n] for n in G.nodes()}
            nx.draw_networkx_labels(G,pos, labels,font_color="white", ax=ax)
        else:
            nx.draw_networkx_labels(G, pos, font_color="white", ax=ax)
        weights = np.round(weights, decimals=2)
        #path = nx.shortest_path(G, source=self.s, target=self.t)
        path_edges = [lin2mat[x] for x in np.where(mat.flatten() > 0)[0].tolist()]
        path = np.unique(path_edges)
        path_weights = [weights[edges.index(e)] for e in path_edges]
        #path_edges = set(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=color, ax=ax)
        sty = 'simple'
        scale = 0.85
        edge_label_dict = dict(zip(edges, weights))
        path_edge_label_dict = dict(zip(path_edges, path_weights))
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=np.array(weights)*scale, arrowstyle=sty, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=np.array(path_weights)*scale, edge_color=color, arrowstyle=sty, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_dict, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=path_edge_label_dict, font_color=color, ax=ax)
        if ax is None:
            plt.axis('equal')
            plt.show()
        return pos

    def plot_shortest_path(self, ws, mats, edges=None, weights=None, titles=None, scale=4, ratios=[2.5,0.35,1.5], seed=None, mask_non_existing_edges=None):
        plt.set_cmap('RdBu')
        fig, axs = plt.subplots(len(ratios), len(ws), figsize=(6+len(ws), 5), gridspec_kw={'height_ratios':ratios})
        for i, a in enumerate(axs.flatten()):
            A = self.names['alphas'][:len(ws[i % len(ws)])]
            B = range(len(ws[i % len(ws)]))
            if i < len(ws):
                N = np.max([np.max(w) for w in ws])
                a.imshow(ws[i], vmin=-N, vmax=N)
                for (j, k), label in np.ndenumerate(ws[i]):
                    val = np.round(ws[i][k][j], decimals=1)
                    if titles and titles[i] == "Perturbed":
                        txt = f'{val} + G'
                    else:
                        txt = val
                    a.text(j, k, txt, ha='center', va='center', color='white', fontsize=7.5)
                for (j, k), c in np.ndenumerate(mats[i]):
                    if c:
                        width = height = 1
                        a.add_patch(Rectangle((k-.5, j-.5), width, height, fill=False, edgecolor='blue', lw=c*scale))
                a.set_yticks(np.arange(len(A)))
                a.set_yticklabels(A)
                a.set_xticks(np.arange(len(B)))
                a.set_xticklabels(A)
                ext = 0.25
                a.set_xlim(np.array(a.get_xlim()) + np.array([-ext,ext]))
                a.set_ylim(np.array(a.get_ylim()) + np.array([ext,-ext]))
                if titles:
                    a.set_title(titles[i])
            elif i < len(ws) * 2 and i >= len(ws):
                if mask_non_existing_edges:
                    indices = mask_non_existing_edges[0]
                else:
                    indices = len(mats[i % len(ws)])
                mat_s = np.array(mats[i % len(ws)].flatten())
                a.imshow(mat_s[:,np.newaxis].T[:,indices], vmin=-1, vmax=1)
                for ind, val in enumerate(mat_s[indices]):
                    a.text(ind, 0, np.round(val, decimals=1), ha='center', va='center', color='white', fontsize=7)
                a.set_xticks(np.arange(len(mat_s[indices])))
                if indices:
                    edge_possibilites = [(A[t[0]],A[t[1]]) for t in mask_non_existing_edges[1]]
                else:
                    edge_possibilites = [(x, y) for x in self.names['alphas'][:mats[0].shape[0]] for y in self.names['alphas'][:mats[0].shape[0]]]
                a.set_xticklabels(edge_possibilites, rotation=90, fontsize=8)
                a.get_yaxis().set_visible(False)
            else:
                if len(ratios) == 2:
                    continue
                self.plot_SP_graph(ws[i % len(ws)], edges[i % len(ws)], weights[i % len(ws)], ax=a, seed=seed)
                val = 10
                a.set_xlim(np.array([-0.2, 0.2]) * val + np.array([0., 1., ]))
                a.set_ylim(np.array([-0.15, 0.15]) * val + np.array([-1., 0., ]))
        plt.suptitle(f'Weight Matrix and Optimal Shortest Path from {A[self.s]} --> {A[self.t]}')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

    def convert_x_to_edges(self, x, cost_matrix=None, w=None):
        matrix =  x.reshape((len(self.A), len(self.B)))
        edges = []
        weights = []
        all_edges = []
        all_weights = []
        for i in self.A:
            for j in self.B:
                if cost_matrix is not None:
                    val = cost_matrix[i][j]
                else:
                    val = matrix[i][j]
                if val and not np.isinf(val):
                    edges.append((i, j))
                    weights.append(val.item())
                if w is not None and w[i][j]:
                    all_edges.append((i,j))
                    all_weights.append(w[i][j])
        return edges, weights, matrix, all_edges, all_weights


# SHD we will not use as we actually don't have a 'code' with the perturbed vector
def shd(a, b):
    return torch.cdist(a, b, p=0)  # norm 0 in torch equal to hamming distance
def euclidean(a, b):
    return torch.cdist(a, b)
# function to transform a perturbed to an official solution
def differentiable_projection_onto_polytope(b):
    # using approximate rounding function
    return b - ((torch.sin(2 * np.pi * b)) / (2 * np.pi))
# SHD still does not make sense because the approximate projection although differentiable is - well - approximate
def shd_projected(a, b):
    return shd(a, differentiable_projection_onto_polytope(b)[None, :])  # TODO: batches are not handled properly
def nll(a, b):
    c = a * b
    idx = c != 0  # to avoid 0-case
    return -torch.log(c[idx]).sum()
def nll_projected(a, b):
    return nll(a, differentiable_projection_onto_polytope(b)[None, :])
#def hidden()


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


def plot_cost_and_selection(C, S, a, names, N, title, scale=4):
    a.imshow(C, vmin=-N, vmax=N)
    for (j, k), label in np.ndenumerate(C):
        val = np.round(C[k][j], decimals=1)
        #a.text(j, k, val, ha='center', va='center', color='white', fontsize=4.5)
    for (j, k), c in np.ndenumerate(S):
        if c:
            width = height = 1
            #a.add_patch(Rectangle((k - .5, j - .5), width, height, fill=False, edgecolor='red', lw=c * scale))
    a.set_yticks(np.arange(len(names)))
    a.set_yticklabels(names)
    a.set_xticks(np.arange(len(names)))
    a.set_xticklabels(names)
    ext = 0.25
    a.set_xlim(np.array(a.get_xlim()) + np.array([-ext, ext]))
    a.set_ylim(np.array(a.get_ylim()) + np.array([ext, -ext]))
    a.set_title(title)


def plot_gradient(G, a, N, title, mask_mat):
    a.imshow(G, vmin=-N, vmax=N)
    for (j, k), label in np.ndenumerate(G):
        try:
            m = mask_mat.index((j, k)) >= 0
        except Exception as e:
            m = False
        if not m:
            width = height = 1
            a.add_patch(Rectangle((k - .5, j - .5), width, height, fill=True, edgecolor='black', facecolor='black'))
        else:
            val = np.round(G[j][k].item(), decimals=1)
            a.text(k, j, val, ha='center', va='center', color='white', fontsize=6.5)
    a.get_yaxis().set_visible(False)
    a.get_xaxis().set_visible(False)
    a.set_title(title)


def plot_perturbed_selection(mat, a, mask_lin, mask_mat, names, title):
    mat_s = np.array(mat.flatten())
    a.imshow(mat_s[:, np.newaxis].T[:, mask_lin].T, vmin=-1, vmax=1)
    for ind, val in enumerate(mat_s[mask_lin]):
        a.text(0, ind, np.round(val, decimals=2), ha='center', va='center', color='white', fontsize=7)
    a.set_yticks(np.arange(len(mat_s[mask_lin])))
    edge_possibilites = [f'{names[t[0]]}->{names[t[1]]}' for t in mask_mat]
    a.set_yticklabels(edge_possibilites, rotation=0, fontsize=8)
    a.get_xaxis().set_visible(False)
    a.set_title(title)


def compute_perturbed_solution(w, s, t, seed=0, num_samples=200, sigma=0.5):
    # with perturbations
    torch.manual_seed(seed)
    np.random.seed(seed)

    # regular solving
    w_mod = w.copy()
    w_mod[w_mod == 0] = 333  # because the standard formulationa actually considers a list of edges and not a matrix with possible zero entries
    f = ILP_SP(s=s, t=t)
    x = f.solve_SP_ILP(w_mod)
    _, _, mat, all_edges, all_weights = f.convert_x_to_edges(x, cost_matrix=f.selected_cost_matrix, w=w)

    pert_ilp = perturbations.perturbed(f.solve_SP_ILP_loop,
                                       num_samples=num_samples,
                                       sigma=sigma,
                                       noise='gumbel',
                                       batched=False,
                                       device=device)
    pw = torch.tensor(w_mod, requires_grad=True)
    px = pert_ilp(pw)
    _, _, pmat, pall_edges, pall_weights = f.convert_x_to_edges(px, cost_matrix=f.selected_cost_matrix, w=w)

    grad_n, title_n = compute_gradient(x, px, pw, 'norm', euclidean)
    grad_c, title_c = compute_gradient(x, px, pw, 'cost', None)

    del f
    gc.collect()

    return w, mat, all_edges, all_weights, pmat, grad_n, title_n, grad_c, title_c


def compute_indices_and_masks(w):
    lin2mat = {}
    mat2lin = {}
    for i, r in enumerate(w):
        for j, c in enumerate(r):
            lin2mat.update({(i * len(r)) + j: (i, j)})
            mat2lin.update({(i, j): (i * len(r)) + j})
    mask_lin = np.where(w.flatten() > 0)[0].tolist()
    mask_mat = [lin2mat[x] for x in mask_lin]
    return lin2mat, mat2lin, mask_lin, mask_mat


def normalize(X, a, b, mask_mat):
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if np.isnan(X).any():
        raise Exception
    C = [X[t] for t in mask_mat]
    #C = [X[t] for t,_ in np.ndenumerate(X)]
    ma = np.max(C)
    mi = np.min(C)
    if (np.allclose(ma, 0) and np.allclose(mi, 0)) or np.allclose(ma, mi):
        return X
    return (b-a)*((X - mi)/(ma - mi))+a


def produce_figure(plot, sigma, num_samples, names, sp, w, mat, all_edges, all_weights, pmat, mask_lin, mask_mat,
                   lin2mat, grad_n, title_n, grad_c, title_c, s, t, duration, pos, seed, N=None, normalize_grads=False, wratios=np.array([0.5,1,1.,1.,0.5, 0.]), color='red'):
    fig, axs = plt.subplots(1,len(wratios)-sum(wratios <= 0), figsize=(16,5), gridspec_kw={'width_ratios': wratios[wratios>0]})
    if not plot:
        canvas = FigureCanvas(fig)

    if N is None:
        N_cands = N_grads = np.max(w)
    else:
        N_cands = N[0]
        N_grads = N[1]
    pos = sp.plot_SP_graph(w, edges=all_edges, weights=all_weights, lin2mat=lin2mat, mat=mat, names=names, seed=seed, pos=pos, ax=axs[0], color=color)
    if wratios[1] > 0:
        plot_cost_and_selection(w, mat, axs[1-sum(wratios[:2] <= 0)], names, N_cands, 'Edge Costs and sel. Shortest Path (red)')
        print(f'Cost\n{w}\nselected Shortest Path edges\n{mat}\n')
    if normalize_grads:
        # CAUTION!
        # can be dangerous e.g. if matrix >= 0 and you want to have [-1,1]
        b = 1
        a = -1
        N_grads = b
        grad_n = normalize(grad_n, a, b, mask_mat)
        grad_c = normalize(grad_c, a, b, mask_mat)
        title_n += f'\nnormalized to [{a},{b}]'
        title_c += f'\nnormalized to [{a},{b}]'
    if wratios[2] > 0:
        print(f'{title_n}\n{grad_n}\n')
        plot_gradient(grad_n, axs[2-sum(wratios[:3] <= 0)], N_grads, title_n, mask_mat)
    if wratios[3] > 0:
        print(f'{title_c}\n{grad_c}\n')
        plot_gradient(grad_c, axs[3-sum(wratios[:4] <= 0)], N_grads, title_c, mask_mat)
    if wratios[4] > 0:
        plot_perturbed_selection(pmat.detach().numpy(), axs[4-sum(wratios[:5] <= 0)], mask_lin, mask_mat, names, f'Perturbed sel. SP\n{np.round(pmat.detach().numpy().sum(), decimals=2)}')
    if wratios[5] > 0:
        plot_perturbed_selection(mat.detach().numpy(), axs[5-sum(wratios[:6] <= 0)], mask_lin, mask_mat, names, 'Sel. SP')

    plt.suptitle(f'Shortest Path Problem {names[s]}->{names[t]}\nPerturbations: Sigma {sigma}, # Samples {num_samples}')

    plt.tight_layout()
    if plot:
        plt.show()
        return pos
    else:
        plt.pause(duration)
        canvas.draw()  # draw the canvas, cache the renderer
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return img, pos


def single_iteration(w, s, t, seed, num_samples, sigma, pos, names, duration, vmax, wratios=np.array([0.5,1,1.,1.,0.5, 0.]), color='red'):
    w, mat, all_edges, all_weights, pmat, grad_n, title_n, grad_c, title_c = compute_perturbed_solution(w, s, t, seed=seed, num_samples=num_samples, sigma=sigma)
    lin2mat, mat2lin, mask_lin, mask_mat = compute_indices_and_masks(w)
    if np.isnan(grad_n).any().item() or np.isnan(grad_c).any().item():
        raise Exception
    img, pos = produce_figure(False, sigma, num_samples, names, ILP_SP(s,t), w, mat, all_edges, all_weights, pmat, mask_lin, mask_mat, lin2mat, grad_n, title_n, grad_c, title_c, s, t, duration,pos, seed, N=vmax, wratios=wratios, color=color)
    return w, mat, all_edges, all_weights, pmat, grad_n, title_n, grad_c, title_c, lin2mat, mat2lin, mask_lin, mask_mat, img, pos


def create_matrix(d, use_f=False):
    num_cities = len(d)
    w = np.zeros((num_cities, num_cities))
    for k in d:
        id, out, f, h, _ = d[k]
        for ind, j in enumerate(out):
            if use_f:
                if isinstance(f, list):
                    val = f[ind]
                else:
                    val = f
            else:
                val = 1
            w[id,j] = val
    return w


def get_pos(d):
    dd = {}
    for k in d:
        id, _, _, _, pos = d[k]
        dd.update({id: np.array(pos)})
    return dd


def get_h(d):
    dd = {}
    for k in d:
        id, _, _, h, _ = d[k]
        dd.update({id: h})
    return dd


# general parameters
cities_id_out_f_h_pos = {
    'ny': (0, [1, 2], 1, [1, 1], [0., 0.]),
    'cl': (1, [3], 1, 1.5, [-0.1, 0.]),
    'wa': (2, [1], 1, 1, [-0.05, -0.1]),
    'ch': (3, [4,5,6], [0.5, 2.7, 1.], [1.5, 1., 0.5], [-0.2, 0.]),
    'mi': (4, [7], 1, 2., [-0.3, 0.1]),
    'sl': (5, [14,15], [1, 1.5], [1., 1.5], [-0.575, 0.]),
    'ka': (6, [8], 1, 1.5, [-0.3, -0.1]),
    'wi': (7, [9], 0.25, 2.5, [-0.4, 0.3]),
    'de': (8, [5], 1, 1.5, [-0.4, -0.1]),
    're': (9, [10], 0.25, 3., [-0.5, 0.3]),
    'ca': (10, [11], 0.25, 3, [-0.7, 0.4]),
    'se': (11, [12], 0.5, 2., [-0.8, 0.2]),
    'po': (12, [14], 1, 1.5, [-0.8, 0.1]),
    'lv': (13, [14,15], 2, 1., [-0.6, -0.15]),
    'sa': (14, [16], 1, 0.5, [-0.7, 0.]),
    'la': (15, [16], 1, 2, [-0.75, -0.25]),
    'sf': (16, [], 1, 1, [-0.85, -0.1])
}
id2city = {}
city2id = {}
for k in cities_id_out_f_h_pos:
    id2city.update({cities_id_out_f_h_pos[k][0]: k})
    city2id.update({k: cities_id_out_f_h_pos[k][0]})

seed = 0
s = 0
t = 16
num_samples = 20#100  # for expectation, perturbations approach
sigma = 0.25
names = list(cities_id_out_f_h_pos.keys())
save = False
exp_name = 'NY-trip-to-SF'

# make animation
duration = 0.0325
imgs = []
vmax = (3, 5)  # Cost and Selection, Gradients, max value across all changes such that plot colors have better meaning
step_size = 0.1
max_chg = 1.
chg_range = np.round(np.arange(0, max_chg+step_size,step_size), decimals=2) #np.round(np.arange(-max_chg, max_chg+step_size,step_size), decimals=2)
pos = get_pos(cities_id_out_f_h_pos)

w = create_matrix(cities_id_out_f_h_pos, use_f=True)
#w[(city2id['ch'], city2id['sl'])] += 0.1
wratios = np.array([2,0.75,0.,0.,0.35, 0.35])
w, mat_before, all_edges, all_weights, pmat, grad_n, title_n, grad_c, title_c, lin2mat, mat2lin, mask_lin, mask_mat, img, pos = single_iteration(
    w, s, t, seed, num_samples, sigma, pos, names, duration, vmax, wratios=wratios, color='blue')
indices = w > 0
w[indices] += 0.01 * np.sign(grad_n.detach().numpy()[indices])
fig = plt.figure()
a = fig.gca()
plot_gradient(grad_n.detach().numpy(), a, vmax[1], title_n, mask_mat)
w, mat_after, all_edges, all_weights, pmat, grad_n, title_n, grad_c, title_c, lin2mat, mat2lin, mask_lin, mask_mat, img, pos = single_iteration(
    w, s, t, seed, num_samples, sigma, pos, names, duration, vmax, wratios=wratios, color='red')

# hidden = get_h(cities_id_out_f_h_pos)
# ys = []
# for k in hidden:
#     if isinstance(hidden[k], list):
#         for v in hidden[k]:
#             ys.append(v)
#     else:
#         ys.append(hidden[k])
# xs = np.arange(len(ys))
# transitions = [f'{names[t[0]]}->{names[t[1]]}' for t in mask_mat]
# assert(len(ys) == len(transitions))
# def plot_hidden(xs, ys, mat, color):
#     plt.figure(figsize=(8,5))
#     plt.bar(xs, ys, color='gray', alpha=0.5)
#     ys_after_edges = [lin2mat[g] for g in np.where(mat.flatten() > 0)[0]]
#     ys_after = np.zeros(xs.shape)
#     for _, y in enumerate(ys_after_edges):
#         try:
#             ind = all_edges.index(y)
#             ys_after[ind] = ys[ind]
#         except:
#             pass
#     plt.bar(xs, ys_after, color=color, alpha=0.75)
#     plt.xticks(xs, transitions, rotation=90)
#     plt.tight_layout()
#     plt.show()
#     return ys_after
# ys_before = plot_hidden(xs, ys, mat_before, 'blue')
# ys_after = plot_hidden(xs, ys, mat_after, 'red')
# def hidden_formula(x):
#     return sum([y**2 for y in x])
# plt.bar([0], [hidden_formula(ys_before)], color='blue')
# plt.bar([1], [hidden_formula(ys_after)], color='red')
# plt.xticks([], [])
# plt.tight_layout()
# plt.show()

# attack_cities = ['ch']#, 'de']
# for n, ct in enumerate(attack_cities):
#     for ind, h in enumerate(chg_range):
#
#         w = create_matrix(cities_id_out_f_h_pos, use_f=True)
#         if n > 0:
#             for nn in range(n):
#                 w[(city2id[attack_cities[nn]], city2id['sl'])] += max_chg
#         chg_loc = (city2id[ct], city2id['sl'])
#         w[chg_loc] += h
#
#         w, mat, all_edges, all_weights, pmat, grad_n, title_n, grad_c, title_c, lin2mat, mat2lin, mask_lin, mask_mat, img, pos = single_iteration(
#             w, s, t, seed, num_samples, sigma, pos, names, duration, vmax, wratios=np.array([2,0.,1.,0.,0.5, 0.]))
#         imgs.append(img)

if save:
    name = f'{exp_name}-{names[s]}-to-{names[t]}-{int(num_samples)}-samples-{sigma}-sig-{int(max_chg / step_size)}-ssq'
    if not os.path.exists(name):
        os.makedirs(name)
    for ind, i in enumerate(imgs):
        imageio.imsave(os.path.join(name, name + f'_{ind+1}.png'), i)
        print(f'Saved IMG: {ind + 1}/{len(imgs)}          ', end='\r', flush=True)
    speeds = [1, 2, 4, 8]
    for ind, k in enumerate(speeds):
        imageio.mimsave(os.path.join(name, name + f'_x{k}.gif'), imgs, duration=(duration * 25)/k)
        print(f'Saved GIF: {ind + 1}/{len(speeds)}          ', end='\r', flush=True)