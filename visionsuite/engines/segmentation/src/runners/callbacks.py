
### TrainRunner
def on_train_runner_set_configs(runner, *args, **kwargs):
    pass

def on_train_runner_set_variables(runner, *args, **kwargs):
    if runner.args['augment']['train']['backend'].lower() != "pil" and not runner.args['augment']['train']['use_v2']:
        # TODO: Support tensor backend in V1?
        raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
    if runner.args['augment']['train']['use_v2'] and runner.args['dataset']['type'] != "coco":
        raise ValueError("v2 is only support supported for coco dataset for now.")

def on_train_runner_run_start(runner, *args, **kwargs):
    pass

def on_train_runner_run_end(runner, *args, **kwargs):
    pass

train_callbacks = {
    "on_runner_set_configs": [on_train_runner_set_configs],
    "on_runner_set_variables": [on_train_runner_set_variables],
    "on_runner_run_start": [on_train_runner_run_start],
    "on_runner_run_end": [on_train_runner_run_end],
}


### TestRunner
def on_test_runner_set_configs(runner, *args, **kwargs):
    pass

def on_test_runner_set_variables(runner, *args, **kwargs):
    if runner.args['augment']['test']['backend'].lower() != "pil" and not runner.args['augment']['test']['use_v2']:
        # TODO: Support tensor backend in V1?
        raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
    if runner.args['augment']['test']['use_v2'] and runner.args['dataset']['type'] != "coco":
        raise ValueError("v2 is only support supported for coco dataset for now.")

def on_test_runner_run_start(runner, *args, **kwargs):
    pass

def on_test_runner_run_end(runner, *args, **kwargs):
    pass

test_callbacks = {
    "on_runner_set_configs": [on_train_runner_set_configs],
    "on_runner_set_variables": [on_train_runner_set_variables],
    "on_runner_run_start": [on_train_runner_run_start],
    "on_runner_run_end": [on_train_runner_run_end],
}