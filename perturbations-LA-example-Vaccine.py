# Paper published at ACML 2023 Journal Track
# Link to Conference: https://www.acml-conf.org/2023/papers.html
# corresponding author: Matej Zecevic, matej.zecevic@tu-darmstadt.de
#
# Reproducing the Linear Assignment example from the paper
import torch
import torch.nn.functional as F
import perturbations
import numpy as np
from mip import Model, xsum, maximize, BINARY, OptimizationStatus
import string
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid", {'axes.grid' : False})
import networkx as nx
import pickle
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ILP_LA:
    def __init__(self):
        self.m = Model("LA")
        self.names = {'alphas': string.ascii_uppercase}

    def solve_LA_ILP_loop(self, w):
        results = []
        assert (len(w.shape) == 3)
        for g in w:
            results.append(self.solve_LA_ILP(g))
        return torch.stack(results)

    def solve_LA_ILP(self, w):
        self.A = self.B = range(len(w))
        self.x = [[self.m.add_var(var_type=BINARY) for j in self.B] for i in self.A]

        self.m.objective = maximize(xsum(w[i][j].item() * self.x[i][j] for i in self.A for j in self.B))

        for i in self.A:
            self.m += xsum(self.x[i][j] for j in self.B) == 1
        for i in self.B:
            self.m += xsum(self.x[j][i] for j in self.A) == 1

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
                        self.edges.append((self.names['alphas'][i], j))
                        self.selected_cost_matrix[i][j] = w[i][j].item()
                        self.decision_matrix[i][j] = 1
        else:
            self.edges = False
        return torch.from_numpy(self.decision_matrix.flatten())

    def plot_bipartite_matching(self, w, mat, title=None, scale=4, color='green', save=None):
        plt.set_cmap('RdBu')
        fig, axs = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [2,1]})
        for i, a in enumerate(axs.flatten()):
            A = self.names['alphas'][:len(w)]
            B = range(len(w))
            if i == 0:
                N = np.max(w)
                a.imshow(w, vmin=-N, vmax=N)
                # for (j, k), label in np.ndenumerate(w):
                #     val = np.round(w[k][j], decimals=1)
                #     a.text(j, k, val, ha='center', va='center', color='white', fontsize=4)
                # for (j, k), c in np.ndenumerate(mat):
                #     if c:
                #         width = height = 1
                #         a.add_patch(Rectangle((k-.5, j-.5), width, height, fill=False, edgecolor=color, lw=c*scale))
                a.set_yticks(np.arange(len(A)))
                a.set_yticklabels(A)
                a.set_xticks(np.arange(len(B)))
                ext = 0.25
                a.set_xlim(np.array(a.get_xlim()) + np.array([-ext,ext]))
                a.set_ylim(np.array(a.get_ylim()) + np.array([ext,-ext]))
                if title:
                    a.set_title(title)
            else:
                a.imshow(np.zeros(mat.shape), vmin=-1, vmax=1)
                for (j, k), val in np.ndenumerate(mat):
                    if val:
                        a.add_patch(Rectangle((k-.5, j-.5), 1, 1, fill=True, facecolor=color))
        plt.suptitle(f'Weight Matrix and Optimal Matching (Solution to ILP)')
        plt.tight_layout()
        if save is not None and title:
            if not os.path.exists(save):
                os.makedirs(save)
            p = os.path.join(save, f'{title}.png')
            plt.savefig(p, dpi=300)
            print(f'Saved to {p}')
        else:
            plt.show()
            plt.clf()
            plt.close()

    def plot_LA_graph(self, nodes, nodes2, edges, weights, ax=None, style=(300, 10, 12), scale=2, draw_edge_text=False):
        G = nx.Graph()
        def create_pos(column, node_list):
            pos = {}
            y_val = 0
            for key in node_list:
                pos[key] = (column, y_val)
                y_val = y_val + 1
            return pos
        G.add_nodes_from(nodes)
        G.add_nodes_from(nodes2)
        color_map = []
        for node in G:
            if isinstance(node, str):
                color_map.append('cyan')
            else:
                color_map.append('orange')
        G.add_edges_from(edges)
        pos1 = create_pos(0, nodes)
        pos2 = create_pos(1, nodes2)
        pos = {**pos1, **pos2}
        pos = {node: (x, -y) for (node, (x, y)) in pos.items()}  # flip y-axis pos
        edge_label_dict = dict(zip(edges, weights))
        if ax:
            nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=style[0], arrowsize=style[1],
                    font_size=style[2], ax=ax, width=np.array(weights)*scale)
            if draw_edge_text:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_dict, ax=ax, font_size=5)
        else:
            nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=style[0], arrowsize=style[1],
                    font_size=style[2], width=np.array(weights)*scale)
            if draw_edge_text:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_dict, font_size=5)
            plt.show()
            plt.clf()
            plt.close()

    def convert_x_to_edges(self, x):
        matrix =  x.reshape((len(self.A), len(self.B)))
        edges = []
        weights = []
        for i in self.A:
            for j in self.B:
                val = matrix[i][j]
                if val:
                    edges.append((self.names['alphas'][i], j))
                    weights.append(val.item())
        return edges, weights, matrix

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


def plot_gradient(grad, title, save=None, no_text=False):
    fig = plt.figure(figsize=(9,9))
    a = plt.gca()
    a.imshow(grad, vmin=-1, vmax=1)
    if not no_text:
        for (j, k), label in np.ndenumerate(grad):
            val = np.round(grad[k][j].item(), decimals=2)
            a.text(j, k, val, ha='center', va='center', color='black', fontsize=9)
    a.get_yaxis().set_visible(False)
    a.get_xaxis().set_visible(False)
    plt.title(title)
    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)
        p = os.path.join(save, f'_Gradient.png')
        plt.savefig(p)
        print(f'Saved to {p}')
    else:
        plt.show()
        plt.clf()
        plt.close()


# regular solving
pid_f_h = [
    (0,  np.ones(20), 3),
    (1,  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 2., 1., 1., 2., 3., 3., 1.]), 7),
    (2,  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 2., 0., 0., 2., 3., 2., 2.]), 3),
    (3,  np.ones(20), 2),
    (4,  np.ones(20), 5),
    (5,  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 0., 2., 1., 1., 2., 0., 3., 1.]), 3),
    (6,  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 2., 1., 1., 2., 0., 0., 1.]), 7),
    (7,  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 2., 1., 2., 1., 3., 2., 1., 1., 1.]), 3),
    (8,  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 2., 1., 2., 1., 1., 1., 1., 3., 3.]), 1),
    (9,  np.ones(20), 3),
    (10, np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 1., 0., 2., 2., 2., 0., 1., 2.]), 2),
    (11, np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 1., 0., 1., 2., 2., 0., 1., 0.]), 4),
    (12, np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 2., 3., 0., 2., 3., 2., 0., 1., 2.]), 8),
    (13, np.ones(20), 3),
    (14, np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 2., 1., 1., 2., 3., 0., 1.]), 3),
    (15, np.ones(20), 2),
    (16, np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 2., 0., 0., 2., 3., 2., 2.]), 8),
    (17, np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0.]), 5),
    (18, np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., 0., 2., 0., 0., 0.]), 6),
    (19, np.ones(20), 2),
]
n_persons = len(pid_f_h)
w = np.zeros((n_persons, n_persons))
hs = []
for t in pid_f_h:
    id, f, h = t
    w[id,:] = f
    hs.append(h)
hs = np.array(hs)

la = ILP_LA()
x = la.solve_LA_ILP(w)
edges, weights, mat = la.convert_x_to_edges(x)

# with perturbations
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

pert_ilp = perturbations.perturbed(la.solve_LA_ILP_loop,
                                      num_samples=15,
                                      sigma=0.5,
                                      noise='gumbel',
                                      batched=False,
                                      device=device)
pw = torch.tensor(w, requires_grad=True)
px = pert_ilp(pw)

pedges, pweights, pmat = la.convert_x_to_edges(px)

grad_method = 'norm'
norm_function = euclidean
grad, title = compute_gradient(x, px, pw, grad_method, norm_function)

w_adv = w.copy()
indices = w_adv > 0
w_adv[indices] += 0.01 * np.sign(grad.detach().numpy()[indices])

x_adv = la.solve_LA_ILP(w_adv)
edges_adv, weights_adv, mat_adv = la.convert_x_to_edges(x_adv)

save = 'Adv-LA-example-Vaccine/'
la.plot_bipartite_matching(w, mat, 'Regular', color='blue', scale=1, save=save)
plot_gradient(grad, title, save=save, no_text=True)
la.plot_bipartite_matching(w_adv, mat_adv, 'Adversarial', color='red', scale=1, save=save)


lin2mat = {}
mat2lin = {}
for i, r in enumerate(w):
    for j, c in enumerate(r):
        lin2mat.update({(i * len(r)) + j: (i, j)})
        mat2lin.update({(i, j): (i * len(r)) + j})
mask_lin = np.where(w.flatten() > 0)[0].tolist()
mask_mat = [lin2mat[x] for x in mask_lin]
indices = np.where(mat.flatten() > 0)[0]
indices_adv = np.where(mat_adv.flatten() > 0)[0]
indices = dict([lin2mat[g] for g in indices])
indices_adv= dict([lin2mat[g] for g in indices_adv])
assert(len(hs) == len(indices))
assert(len(hs) == len(indices_adv))
def plot_hidden(xs, ys, sort_indices, mat_indices, vac_spots, color, title=None, save=None):
    plt.figure(figsize=(8,5))
    plt.bar(xs, ys[sort_indices], color='gray', alpha=0.5)

    pos2id = []
    id2pos = []
    for i in range(len(xs)):
         pos2id.append((i, xs[sort_indices][i]))
         id2pos.append((xs[sort_indices][i], i))
    pos2id = dict(pos2id)
    id2pos = dict(id2pos)

    do_vac = []
    no_vac = []
    for i in xs[sort_indices]:
        try:
            vac_spots.index(mat_indices[i])
            do_vac.append((id2pos[i], ys[i]))
        except:
            no_vac.append((id2pos[i], ys[i]))
    ys_do_vac = np.zeros(xs.shape)
    for t in do_vac:
        ys_do_vac[t[0]] = t[1]
    ys_no_vac = np.zeros(xs.shape)
    for t in no_vac:
        ys_no_vac[t[0]] = t[1]

    #plt.bar(xs, ys_no_vac, color=color, alpha=0.5)
    plt.bar(xs, ys_do_vac, color=color, alpha=0.75)

    plt.xticks(xs, xs[sort_indices])
    plt.tight_layout()
    if save is not None and title:
        if not os.path.exists(save):
            os.makedirs(save)
        p = os.path.join(save, f'_HM_{title}.png')
        plt.savefig(p)
        print(f'Saved to {p}')
    else:
        plt.show()
        plt.clf()
        plt.close()
    return do_vac, ys_do_vac
vaccination_spots = list(np.arange(10,20))
hs_sort_indices = np.argsort(hs)
hs_b_1, hs_b_2 = plot_hidden(np.arange(len(hs)), hs, hs_sort_indices, indices, vaccination_spots, 'blue', 'Regular', save)
hs_a_1, hs_a_2 = plot_hidden(np.arange(len(hs)), hs, hs_sort_indices, indices_adv, vaccination_spots, 'red', 'Adversarial', save)