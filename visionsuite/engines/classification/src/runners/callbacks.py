

def on_train_runner_set_configs(runner, *args, **kwargs):
    pass

def on_train_runner_set_variables(runner, *args, **kwargs):
    pass

def on_train_runner_run_start(runner, *args, **kwargs):
    pass

def on_train_runner_run_end(runner, *args, **kwargs):
    pass

callbacks = {
    "on_runner_set_configs": [on_train_runner_set_configs],
    "on_runner_set_variables": [on_train_runner_set_variables],
    "on_runner_run_start": [on_train_runner_run_start],
    "on_runner_run_end": [on_train_runner_run_end],
}

