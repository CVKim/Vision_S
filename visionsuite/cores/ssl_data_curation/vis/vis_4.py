import sys
sys.path.insert(0, "..")
from pathlib import Path
import pickle
import torch
import os.path as osp
from copy import deepcopy
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import shutil
from tqdm import tqdm
import random
import seaborn as sns
import pandas as pd

from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances as pwd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import os 
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
    fig_filename="/HDD/etc/voronoi.png",
):
    if legends is None:
        legends = clustering_names
    figh, figw = 1, len(clustering_names)
    fig, ax = plt.subplots(figh, figw, figsize=(basic_fig_size[0] * figw, basic_fig_size[1]))
    axs = ax.ravel()
    for i, _name in enumerate(clustering_names):
        vor = Voronoi(clusterings[_name][:, :2])
        voronoi_plot_2d(vor, ax=axs[i], show_vertices=False, point_size=point_size, line_width=line_width, line_alpha=0.8)
        if show_title:
            axs[i].set_title(f'{legends[i]} - {clusterings[_name][:, :2].shape[0]}', fontsize=int(fontsize * 0.8), pad=20)
        axs[i].set_ylim(ylim[0], ylim[1])
        axs[i].set_xlim(xlim[0], xlim[1])
        axs[i].axis('off')
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.savefig(fig_filename)
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
    s=0.2,
    z_high=None,
    show_title=True,
    cmap_max=None,
    basic_fig_size = (6, 5),
    compute_kl = True,
    wspace=0.1,
    hspace=0.1,
    fig_filename="/HDD/etc/kde.png",

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
            clusterings[_name][:, :2]
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
    plt.savefig(fig_filename)
    
    kl_dist = None
    if compute_kl:
        kl_dist = {}
        for _name in clustering_names:
            kl = compute_KL_divergence(kdes[_name], L, s)
            kl_dist[_name] = kl
            print(f"Name: {_name}, KL divergence: {kl}")
    return fig, kl_dist

make_clustered_dataset = True
level = '7'
ratio = 0.05
# project_name = 'tenneco'
# input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_dataset'
project_name = 'mr'
input_dir = '/HDD/datasets/projects/mr/split_dataset'
output_dir = f'/HDD/etc/curation/{project_name}'

ori_data = np.load(f'/HDD/etc/curation/{project_name}/embeddings/representations/dinov2/labelme_train.npy', mmap_mode="r")
ori_index_f = open(f'/HDD/etc/curation/{project_name}/embeddings/representations/dinov2/labelme_train_filenames.txt', 'r')
ori_index = []
while True:
    line = ori_index_f.readline()
    
    if not line: break
    
    ori_index.append(line.rstrip())
    
    
print("data: ", ori_data.shape)
fig, ax = plt.subplots(1, 1, figsize=(6 * 4, 5))
ax.scatter(ori_data[:, 0], ori_data[:, 1], alpha=0.2, color='r')

data1 = np.load(f'/HDD/etc/curation/{project_name}/outputs/level1/centroids.npy', mmap_mode="r")
data2 = np.load(f'/HDD/etc/curation/{project_name}/outputs/level2/centroids.npy', mmap_mode="r")
data3 = np.load(f'/HDD/etc/curation/{project_name}/outputs/level3/centroids.npy', mmap_mode="r")
data4 = np.load(f'/HDD/etc/curation/{project_name}/outputs/level4/centroids.npy', mmap_mode="r")
data5 = np.load(f'/HDD/etc/curation/{project_name}/outputs/level5/centroids.npy', mmap_mode="r")
data6 = np.load(f'/HDD/etc/curation/{project_name}/outputs/level6/centroids.npy', mmap_mode="r")
data7 = np.load(f'/HDD/etc/curation/{project_name}/outputs/level7/centroids.npy', mmap_mode="r")

clusters = {}
counts = []
for _level in range(1, 8):
    sorted_clusters = np.load(f'/HDD/etc/curation/{project_name}/outputs/level{_level}/sorted_clusters.npy', allow_pickle=True)
    clusters[f'cluster_{_level}'] = sorted_clusters
    count = []
    for cluster in sorted_clusters:
        count.append(len(cluster))
    counts.append(count)
    
labels = [f"Group {i+1}" for i in range(len(counts))]

# 데이터를 DataFrame으로 변환
all_values = []
group_labels = []
for i, group in enumerate(counts):
    all_values.extend(group)
    group_labels.extend([labels[i]] * len(group))

df = pd.DataFrame({
    "Value": all_values,
    "Group": group_labels
})

# seaborn 스타일 설정
sns.set(style="whitegrid", palette="pastel")

# boxplot 그리기
plt.figure(figsize=(8, 5))
sns.boxplot(x="Group", y="Value", data=df, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
            boxprops={"facecolor": "skyblue", "edgecolor": "gray"},
            medianprops={"color": "red"})

# 제목 및 레이아웃
plt.title("Distribution of Counts per Group", fontsize=14)
plt.ylabel("Value", fontsize=12)
plt.xlabel("Group", fontsize=12)
plt.tight_layout()

plt.savefig(osp.join(output_dir, 'clusters.png'))


res = {"ori_data": ori_data, "data1": data1, "data2": data2, "data3": data3, 
       "data4": data4, "data5": data5, "data6": data6, "data7": data7}

clustering_names = ['ori_data', 'data1', "data2", 'data3', "data4", 'data5', 'data6', 'data7']
fig = draw_voronoi(
    res,
    clustering_names,
    xlim=(-3, 3),
    ylim=(-3, 3),
    point_size=3,
    line_width=0.7,
    fontsize=30,
    basic_fig_size = (6,4),
    fig_filename=osp.join(output_dir, 'voronoi.png')
)

if make_clustered_dataset:
    clustered_dataset_dir = osp.join(output_dir, f'clustered_dataset_level{level}')
    sorted_clusters = np.load(f'/HDD/etc/curation/{project_name}/outputs/level{level}/sorted_clusters.npy', allow_pickle=True)
    sorted_clusters = np.concatenate(sorted_clusters)

    clustered_dataset_train_dir = osp.join(clustered_dataset_dir, 'train')
    os.makedirs(clustered_dataset_train_dir, exist_ok=True)

    for index in tqdm(sorted_clusters):
        img_file = osp.join(input_dir, 'train', ori_index[index] + '.bmp')
        json_file = osp.join(input_dir, 'train', ori_index[index] + '.json')
        assert osp.exists(img_file), RuntimeError(f'There is no such image file: {img_file}')
        assert osp.exists(json_file), RuntimeError(f'There is no such json file: {json_file}')

        shutil.copyfile(img_file, osp.join(clustered_dataset_train_dir, ori_index[index] + '.bmp'))
        shutil.copyfile(json_file, osp.join(clustered_dataset_train_dir, ori_index[index] + '.json'))

    clustered_dataset_val_dir = osp.join(clustered_dataset_dir, 'val')
    os.makedirs(clustered_dataset_val_dir, exist_ok=True)
    clustered_dataset_test_dir = osp.join(clustered_dataset_dir, 'test')
    os.makedirs(clustered_dataset_test_dir, exist_ok=True)

    for jdx, filename in tqdm(enumerate(ori_index)):
        if jdx not in sorted_clusters:
            img_file = osp.join(input_dir, 'train', ori_index[jdx] + '.bmp')
            json_file = osp.join(input_dir, 'train', ori_index[jdx] + '.json')
            assert osp.exists(img_file), RuntimeError(f'There is no such image file: {img_file}')
            assert osp.exists(json_file), RuntimeError(f'There is no such json file: {json_file}')

            if ratio > random.uniform(0, 1):
                shutil.copyfile(img_file, osp.join(clustered_dataset_val_dir, ori_index[jdx] + '.bmp'))
                shutil.copyfile(json_file, osp.join(clustered_dataset_val_dir, ori_index[jdx] + '.json'))
            else:
                shutil.copyfile(img_file, osp.join(clustered_dataset_test_dir, ori_index[jdx] + '.bmp'))
                shutil.copyfile(json_file, osp.join(clustered_dataset_test_dir, ori_index[jdx] + '.json'))


fig, kl_dist_1 = visualize_kde(
    res,
    clustering_names,
    legends = clustering_names,
    z_high=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    fontsize=30,
    show_title=False,
    L=3,
    basic_fig_size = (6,4),
    fig_filename=osp.join(output_dir, 'kde.png')
)


# figh, figw = 5, 5
# fig, axs = plt.subplots(figh, figw, figsize=(12 * figw, 5 * figh))
# axs[0][0].scatter(ori_data[:, 0], ori_data[:, 1], alpha=0.2)
# axs[0][0].set_title("original data", fontsize=20)
# axs[0][0].tick_params(labelsize=16)
# axs[0][1].scatter(data1[:, 0], data1[:, 1], alpha=0.2)
# axs[0][1].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[0][1].tick_params(labelsize=16)
# axs[0][2].scatter(data2[:, 0], data2[:, 1], alpha=0.2)
# axs[0][2].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[0][2].tick_params(labelsize=16)
# axs[0][3].scatter(data3[:, 0], data3[:, 1], alpha=0.2)
# axs[0][3].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[0][3].tick_params(labelsize=16)
# axs[0][4].scatter(data4[:, 0], data4[:, 1], alpha=0.2)
# axs[0][4].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[0][4].tick_params(labelsize=16)

# axs[1][0].scatter(ori_data[:, 2], ori_data[:, 3], alpha=0.2)
# axs[1][0].set_title("original data", fontsize=20)
# axs[1][0].tick_params(labelsize=16)
# axs[1][1].scatter(data1[:, 2], data1[:, 3], alpha=0.2)
# axs[1][1].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[1][1].tick_params(labelsize=16)
# axs[1][2].scatter(data2[:, 2], data2[:, 3], alpha=0.2)
# axs[1][2].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[1][2].tick_params(labelsize=16)
# axs[1][3].scatter(data3[:, 2], data3[:, 3], alpha=0.2)
# axs[1][3].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[1][3].tick_params(labelsize=16)
# axs[1][4].scatter(data4[:, 2], data4[:, 3], alpha=0.2)
# axs[1][4].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[1][4].tick_params(labelsize=16)

# axs[2][0].scatter(ori_data[:, 22], ori_data[:, 23], alpha=0.2)
# axs[2][0].set_title("original data", fontsize=20)
# axs[2][0].tick_params(labelsize=16)
# axs[2][1].scatter(data1[:, 22], data1[:, 23], alpha=0.2)
# axs[2][1].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[2][1].tick_params(labelsize=16)
# axs[2][2].scatter(data2[:, 22], data2[:, 23], alpha=0.2)
# axs[2][2].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[2][2].tick_params(labelsize=16)
# axs[2][3].scatter(data3[:, 22], data3[:, 23], alpha=0.2)
# axs[2][3].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[2][3].tick_params(labelsize=16)
# axs[2][4].scatter(data4[:, 22], data4[:, 23], alpha=0.2)
# axs[2][4].set_title("data sampled with hierarchical k-means", fontsize=20)
# axs[2][4].tick_params(labelsize=16)

# plt.savefig(osp.join(output_dir, 'vis_4.png'))