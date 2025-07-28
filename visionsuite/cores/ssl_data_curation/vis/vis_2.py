import sys
sys.path.insert(0, "..")
from pathlib import Path
import pickle
import torch
import os
import os.path as osp

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances as pwd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from visionsuite.cores.ssl_data_curation.src.hierarchical_kmeans_gpu import hierarchical_kmeans_with_resampling, hierarchical_kmeans


def draw_voronoi(
    clusterings,
    clustering_names,
    legends = None,
    fontsize = 20,
    ylim = (-3, 3),
    xlim = (-3, 3),
    line_width=0.4,
    point_size=1,
    show_title=True,
    basic_fig_size = (6, 5),
    wspace=0.1,
    hspace=0.1,
    filename=None,
):
    if legends is None:
        legends = clustering_names
    figh, figw = 1, len(clustering_names)
    fig, ax = plt.subplots(figh, figw, figsize=(basic_fig_size[0] * figw, basic_fig_size[1]))
    axs = ax.ravel()
    for i, _name in enumerate(clustering_names):
        if _name == "data":
            axs[0].scatter(clusterings[_name][:, 0], clusterings[_name][:, 1], alpha=0.2)
        else:
            vor = Voronoi(clusterings[_name][-1]['centroids'].cpu().numpy())
            voronoi_plot_2d(vor, ax=axs[i], show_vertices=False, point_size=point_size, line_width=line_width, line_alpha=0.8)
        if show_title:
            axs[i].set_title(legends[i], fontsize=int(fontsize * 0.8), pad=20)
        axs[i].set_ylim(ylim[0], ylim[1])
        axs[i].set_xlim(xlim[0], xlim[1])
        axs[i].axis('off')
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    return fig

def compute_KL_divergence(kde, L, s):
    X = np.arange(-L, L, s)
    Y = np.arange(-L, L, s)
    X, Y = np.meshgrid(X, Y)
    XY = np.vstack([X.ravel(), Y.ravel()]).T
    d_u = 1/(4*L**2)
    
    d = np.exp(kde.score_samples(XY))
    d_norm = d * 1/(s**2 * d.sum())
    KL = -(d_norm * np.log(d_u/d_norm)).sum() * s**2
    return KL
    
def visualize_kde(
    clusterings,
    clustering_names,
    legends = None,
    fontsize=20,
    bandwidth=0.5,
    L=4,
    s=0.02,
    z_high=None,
    show_title=True,
    cmap_max=None,
    basic_fig_size = (6, 5),
    compute_kl = True,
    wspace=0.1,
    hspace=0.1,
    filename=None,
):
    if z_high is None:
        z_high = [0.05] * len(clustering_names)
    if cmap_max is None:
        cmap_max = z_high
    if legends is None:
        legends = clustering_names
    kdes = {}
    for _name in clustering_names:
        kdes[_name] = KernelDensity(bandwidth=bandwidth).fit(
            clusterings[_name][-1]["centroids"].cpu().numpy()
        )
    
    fig, ax = plt.subplots(
        1, len(clustering_names),
        figsize=(basic_fig_size[0] * len(clustering_names), basic_fig_size[1]),
        subplot_kw={"projection": "3d"},
    )
    axs = ax.ravel()
    for i, _name in enumerate(clustering_names):
        X = np.arange(-L, L, 0.05)
        Y = np.arange(-L, L, 0.05)
        X, Y = np.meshgrid(X, Y)
        XY = np.vstack([X.ravel(), Y.ravel()]).T
        Z = np.exp(kdes[_name].score_samples(XY)).reshape(X.shape)
        
        _ = axs[i].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=0, vmax=cmap_max[i],
        )
        
        axs[i].set_zlim(-0.01, z_high[i])
        axs[i].zaxis.set_major_formatter('{x:.02f}')
        if show_title:
            axs[i].set_title(legends[i], fontsize=fontsize * 8 // 10, pad=20)

        axs[i].tick_params(axis='x', which='major', labelsize=fontsize*5//10)
        axs[i].tick_params(axis='y', which='major', labelsize=fontsize*5//10)
        axs[i].tick_params(axis='z', which='major', labelsize=fontsize*5//10)
    
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
            
    kl_dist = None
    if compute_kl:
        kl_dist = {}
        for _name in clustering_names:
            kl = compute_KL_divergence(kdes[_name], L, s)
            kl_dist[_name] = kl
            print(f"Name: {_name}, KL divergence: {kl}")
    return fig, kl_dist


output_dir = '/HDD/etc/curation/tutorial'
os.makedirs(output_dir, exist_ok=True)

data = np.concatenate([
    np.random.randn(7000, 2)/2 + [-1, -1],
    np.random.randn(1000, 2)/2 + [1, -1],
    np.random.randn(500, 2)/2 + [0, 1],
    (np.random.rand(500,2) - 0.5) * 6,
])
print('data.shape: ', data.shape) # (9000, 2)

figh, figw = 1, 1
fig, ax = plt.subplots(figh, figw, figsize=(6 * figw, 5))
ax.scatter(data[:, 0], data[:, 1], alpha=0.2)
plt.savefig(osp.join(output_dir, 'vis_2_data.png'))

# eta: 4m30s
X = torch.tensor(data, device='cuda', dtype=torch.float32)
num_init=10
res_no_resampling = {"data": data}
print("Running k-means")
res_no_resampling["1level"] = hierarchical_kmeans(X, [300], 1, num_init=num_init, verbose=False)
print("Running 2-level hierarchical k-means")
res_no_resampling["2level"] = hierarchical_kmeans(X, [1500, 300], 2, num_init=num_init, verbose=False)
print("Running 3-level hierarchical k-means")
res_no_resampling["3level"] = hierarchical_kmeans(X, [3000, 1000, 300], 3, num_init=num_init, verbose=False)

clustering_names = ["data", "1level", "2level", "3level"]
fig = draw_voronoi(
    res_no_resampling,
    clustering_names,
    xlim=(-3, 3),
    ylim=(-3, 3),
    point_size=3,
    line_width=0.7,
    fontsize=30,
    basic_fig_size = (6,4),
    filename=osp.join(output_dir, 'vis_2_voronoi.png')
)

res_no_resampling["data"] = [{"centroids": torch.tensor(data),}]
fig, kl_dist_1 = visualize_kde(
    res_no_resampling,
    clustering_names,
    legends = clustering_names,
    z_high=[0.2, 0.2, 0.2, 0.2],
    fontsize=30,
    show_title=False,
    L=3,
    basic_fig_size = (6,4),
    filename=osp.join(output_dir, 'vis_2_kde.png')
)