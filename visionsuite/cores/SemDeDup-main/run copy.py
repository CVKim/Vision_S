from semdedup import SemDeDupJob

embs_memory_loc = "/HDD/etc/semdedup/outputs/clip_openai/clip-vit-large-patch14-336/embedding.npy"
save_loc = '/HDD/etc/semdedup/outputs/results'
sorted_clusters_path = '/HDD/etc/semdedup/outputs/sorted_clusters'
eps_list = [0.2, 0.5]
which_to_keep = 'hard'
device = 'cuda'
clusters_per_job = 1
num_clusters = 300
dataset_size = 3835
emb_size = 768

import math
total_jobs = math.ceil(num_clusters / clusters_per_job)

import time
from concurrent import futures
with futures.ThreadPoolExecutor(max_workers=8) as executor:
    # args: struct of arguments to pass
    from argparse import Namespace
    args = Namespace(
        embs_memory_loc=embs_memory_loc,
        save_loc=save_loc,
        sorted_clusters_path=sorted_clusters_path,
        eps_list=eps_list,
        which_to_keep=which_to_keep,
        device=device,
        dataset_size=dataset_size,
        emd_size=emb_size,
        seed=1234,
        clusters_per_job=clusters_per_job,
        num_clusters=num_clusters
    )

    jobs = []

    def execute_it(jobber):
        print('STARTING JOB: ', jobber)
        exp = SemDeDupJob(
            args=args,
            job_start_cluster=jobber * clusters_per_job)

        return exp()

    for job in range(total_jobs):
        jb = executor.submit(execute_it, jobber=job)
        jobs.append(jb)

    [job.result() for job in jobs]