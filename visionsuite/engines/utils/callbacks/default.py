from collections import defaultdict
from copy import deepcopy

### Runner -------------------------------------
def on_runner_set_configs(runner, *args, **kwargs):
    pass

def on_runner_set_variables(runner, *args, **kwargs):
    pass

def on_runner_run_start(runner, *args, **kwargs):
    pass

def on_runner_run_end(runner, *args, **kwargs):
    pass

### Loop ---------------------------------------------
def on_loop_build_start(loop, *args, **kwargs):
    pass

def on_loop_build_end(loop, *args, **kwargs):
    pass

def on_loop_run_start(loop, *args, **kwargs):
    pass

def on_loop_run_end(loop, *args, **kwargs):
    pass

### Trainer -------------------------------------
def on_trainer_build_start(trainer, *args, **kwargs):
    pass

def on_trainer_build_end(trainer, *args, **kwargs):
    pass

def on_trainer_epoch_start(trainer, *args, **kwargs):
    pass

def on_trainer_epoch_end(trainer, *args, **kwargs):
    pass

def on_trainer_batch_start(trainer, *args, **kwargs):
    pass

def on_trainer_batch_end(trainer, *args, **kwargs):
    pass

def on_trainer_step_start(trainer, *args, **kwargs): # iteration for a batch
    pass

def on_trainer_step_end(trainer, *args, **kwargs): # iteration for a batch
    pass

### Validator -------------------------------------
def on_validator_build_start(validator, *args, **kwargs):
    pass

def on_validator_build_end(validator, *args, **kwargs):
    pass

def on_validator_epoch_start(validator, *args, **kwargs):
    pass

def on_validator_epoch_end(validator, *args, **kwargs):
    pass

def on_validator_batch_start(validator, *args, **kwargs):
    pass

def on_validator_batch_end(validator, *args, **kwargs):
    pass

def on_validator_step_start(validator, *args, **kwargs): # iteration for a batch
    pass

def on_validator_step_end(validator, *args, **kwargs): # iteration for a batch
    pass

### Tester -------------------------------------
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

default_callbacks = {
    ### Runner ---------------------------------------
    "on_runner_set_configs": [on_runner_set_configs],
    "on_runner_set_variables": [on_runner_set_variables],
    "on_runner_run_start": [on_runner_run_start],
    "on_runner_run_end": [on_runner_run_end],
    
    ### Loop -------------------------------
    "on_loop_build_start": [on_loop_build_start],
    "on_loop_build_end": [on_loop_build_end],
    "on_loop_run_start": [on_loop_run_start],
    "on_loop_run_end": [on_loop_run_end],
    
    ### Trainer -------------------------------------
    "on_trainer_build_start": [on_trainer_build_start],
    "on_trainer_build_end": [on_trainer_build_end],
    "on_trainer_epoch_start": [on_trainer_epoch_start],
    "on_trainer_epoch_end": [on_trainer_epoch_end],
    "on_trainer_batch_start": [on_trainer_epoch_start],
    "on_trainer_batch_end": [on_trainer_batch_end],
    "on_trainer_step_start": [on_trainer_step_start],
    "on_trainer_step_end": [on_trainer_step_end],

    ### Validator -------------------------------------
    "on_validator_build_start": [on_validator_build_start],
    "on_validator_build_end": [on_validator_build_end],
    "on_validator_epoch_start": [on_validator_epoch_start],
    "on_validator_epoch_end": [on_validator_epoch_end],
    "on_validator_batch_start": [on_validator_batch_start],
    "on_validator_batch_end": [on_validator_batch_end],
    "on_validator_step_start": [on_validator_step_start],
    "on_validator_step_end": [on_validator_step_end],
    
    ### Tester -----------------------------------------
    "on_tester_build_start": [on_tester_build_start], 
    "on_tester_build_end": [on_tester_build_end], 
    "on_tester_start": [on_tester_start],
    "on_tester_end": [on_tester_end],
    "on_tester_batch_start": [on_tester_batch_start],
    "on_tester_batch_end": [on_tester_batch_end],
    "on_tester_step_start": [on_tester_step_start],
    "on_tester_step_end": [on_tester_step_end],
}


def get_default_callbacks():
    return defaultdict(list, deepcopy(default_callbacks))
