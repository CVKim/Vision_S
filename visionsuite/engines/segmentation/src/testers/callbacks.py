import os.path as osp
import psutil
import time 

from visionsuite.engines.utils.torch_utils.utils import save_on_master

def on_tester_build_start(tester, *args, **kwargs):
    pass        
     
def on_tester_build_end(tester, *args, **kwargs):
    pass 
    
def on_tester_start(tester, *args, **kwargs):
    pass 

def on_tester_end(tester, *args, **kwargs):
    pass
    
def on_tester_batch_start(tester, *args, **kwargs):
    pass

def on_tester_batch_end(tester, *args, **kwargs):
    pass

def on_tester_step_start(tester, *args, **kwargs): # iteration for a batch
    pass

def on_tester_step_end(tester, *args, **kwargs): # iteration for a batch
    pass

callbacks = {
    "on_tester_build_start": [on_tester_build_start], 
    "on_tester_build_end": [on_tester_build_end], 
    "on_tester_start": [on_tester_start],
    "on_tester_end": [on_tester_end],
    "on_tester_batch_start": [on_tester_batch_start],
    "on_tester_batch_end": [on_tester_batch_end],
    "on_tester_step_start": [on_tester_step_start],
    "on_tester_step_end": [on_tester_step_end],
}

