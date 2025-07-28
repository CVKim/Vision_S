
embs_memory_loc = "/HDD/etc/semdedup/outputs/clip_openai/clip-vit-large-patch14-336/embedding.npy"
save_loc = '/HDD/etc/semdedup/outputs/results'
sorted_clusters_path = '/HDD/etc/semdedup/outputs/sorted_clusters'
eps_list = [0.2, 0.5]
which_to_keep = 'hard'
device = 'cuda'
num_clusters = 2
dataset_size = 3835
emb_size = 768

import time 
import torch 
import numpy as np
from tqdm import tqdm 
import os
import pandas as pd
import pickle
import random

def init_memmap_embs(
    embs_memory_loc: str, dataset_size: int, emd_size: int = 512, dtype: str = "float32"
) -> np.memmap:
    """
    Initializes a memory-mapped NumPy array to read embeddings of examples.

    Args:
        embs_memory_loc (str): Path to the memory-mapped file.
        dataset_size (int): Size of the dataset.
        emd_size (int): Dimensionality of the embeddings.
        dtype (str): Data type of the embeddings.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
    embs = np.memmap(
        embs_memory_loc, dtype=dtype, mode="r", shape=(dataset_size, emd_size)
    )
    return embs


def semdedup(cluster, cluster_reps, device):

    def _contains_duplicates(arr):
        return len(np.unique(arr)) != len(arr)

    st = time.time()
    ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
    cluster_reps.to(device)
    pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
    del cluster_reps
    pair_w_sim_matrix.fill_diagonal_(0.0)
    assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

    ## -- get paths to cluster i images
    image_urls = cluster[:, 0]

    ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
    assert not _contains_duplicates(image_urls)

    ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

    ## -- if the max sim between one example and any other example is > 1-eps, remove this example
    M = torch.max(triu_sim_mat, dim=0)[0].cpu()
    print(f"Step time: {time.time()-st}(s)")

    return M

# print("SemDeDup params: ", self.args)
st = time.time()

embs = init_memmap_embs(
    embs_memory_loc, dataset_size, emb_size
)

step_time = []

for cluster_id in tqdm(range(0, num_clusters)):
    step_st = time.time()

    df_file_loc = os.path.join(
        save_loc, f"dataframes/cluster_{cluster_id}.pkl"
    )

    # if os.path.exists(df_file_loc):  # and os.path.exists(dict_file_loc):
    #     print(f"{df_file_loc} exists, moving on")
    #     continue

    ## -- load cluster i representations
    cluster_i = np.load(
        os.path.join(
            sorted_clusters_path, f"cluster_{cluster_id}.npy"
        ), allow_pickle=True
    )
    # 1) store cluster size
    cluster_size = cluster_i.shape[0]
    print("cluster_size: ", cluster_size)

    if cluster_size == 1:
        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = [0]
        for eps in eps_list:
            ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
            points_to_remove_df[f"eps={eps}"] = [False]
        if save_loc != "":
            ## --save df
            with open(df_file_loc, "wb") as file:
                pickle.dump(points_to_remove_df, file)
        print("DONE cluster_id ", cluster_id)
        continue

    ## -- By default, we keep hard examples from groups
    clutser_items_indices = list(range(cluster_size))
    ## -- OR: shuffle cluster to keep random example from each group
    if which_to_keep.lower() == "random":
        random.shuffle(clutser_items_indices)
        cluster_i = cluster_i[clutser_items_indices]
    ## -- OR: reverse cluster to keep easy examples
    if which_to_keep.lower() == "easy":
        clutser_items_indices = clutser_items_indices[::-1]
        cluster_i = cluster_i[clutser_items_indices]

    ## -- indices for cluster items in the dataset
    cluster_ids = cluster_i[:, 1].astype("int32")
    cluster_reps = embs[cluster_ids]
    cluster_reps = torch.tensor(cluster_reps)

    M = semdedup(cluster_i, cluster_reps, device)

    points_to_remove_df = pd.DataFrame()
    points_to_remove_df["indices"] = clutser_items_indices

    for eps in eps_list:
        ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
        eps_points_to_remove = M > 1 - eps
        points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

    if save_loc != "":
        os.makedirs(save_loc, exist_ok=True)
        import os.path as osp
        os.makedirs(osp.join(save_loc, 'dataframes'), exist_ok=True)
        
        ## --save df
        with open(df_file_loc, "wb") as file:
            pickle.dump(points_to_remove_df, file)

    # step_time.append_cluster(time.time() - step_st)
    print("DONE cluster: ", cluster_id)