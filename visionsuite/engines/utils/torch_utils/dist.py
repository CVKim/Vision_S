import os
import torch.distributed as dist
import torch



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args, mode):
    args['distributed']['gpu'] = args[mode]['device_ids'][0]
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print(">>> case 1")
        args['distributed']['rank'] = int(os.environ["RANK"])
        args['distributed']['world_size'] = int(os.environ["WORLD_SIZE"])
        args['distributed']['gpu'] = int(os.environ["LOCAL_RANK"])
        print(f"rank: ", args['distributed']['rank'])
        print(f"world_size: ", args['distributed']['world_size'])
        print(f"gpu: ", args['distributed']['gpu'])
        
        
    # elif "SLURM_PROCID" in os.environ:
    #     args['distributed']['rank = int(os.environ["SLURM_PROCID"])
    #     args['distributed']['gpu = args['distributed']['rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        print(">>> case 2")
        pass
    else:
        print("Not using distributed mode")
        args['distributed']['use'] = False
        return

    args['distributed']['use'] = True

    torch.cuda.set_device(args['distributed']['gpu'])
    args['distributed']['dist_backend'] = "nccl"
    print(f"| distributed init (rank {args['distributed']['rank']}): {args['distributed']['dist_url']}", flush=True)
    torch.distributed.init_process_group(
        backend=args['distributed']['dist_backend'], init_method=args['distributed']['dist_url'], world_size=args['distributed']['world_size'], rank=args['distributed']['rank']
    )
    torch.distributed.barrier()
    setup_for_distributed(args['distributed']['rank'] == 0)


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t
