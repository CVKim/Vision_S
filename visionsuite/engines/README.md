# Guide

## Directory Structure
```
<task>
   |----- cfgs
            |----- dataset
                     |----- xxx.yaml
            |----- default.yaml
   |----- src
           |----- dataloaders
                     |----- root.py
                     |----- xxx.py
           |----- datasets
                     |----- root.py
                     |----- xxx.py
           |----- loops
                     |----- root.py
                     |----- xxx.py
           |----- losses
           |----- models
           |----- optimizers         
           |----- runners
           |----- schedulers
           |----- trainers
           |----- validators
   |----- utils
           |----- augment
           |----- metrics
           |----- results
           |----- vis
```

## Naming Rules
- File names: ***snake_case***
- Class names: ***CamelCase***
- File-Class alignment: Each file name should correspond to the class name it contains, with the file name reflecting the class in snake_case
    - For example, if the class name is `CustomDataset`, then the file should be named `custom_dataset.py`.

## Add new module
1. OOP is preferable
2. Register the module with decorator
3. Importing in `root.py` and including it in the `__all__` list.

## Running process

- `engine`: select and build `runner` based on **task** such as **classification** and **segmentation**

- `runner`: determine one of **mode** including **train**, **test**, and **export** 
    - `BaseTrainRunner`: select and build `dataset`, `model`, and `loop` according to the **task**
        - `set_configs`: parse `yaml` file
        - `set_variables`: convert configs into `args` required for each module
        - `run`: execute defined modules

- `loop`: 
    - `Loop`: select and build `dataloader`, `loss`, `optimizer`, and `scheduler`
    - According the **mode**, it will select and build `trainer`, `validator`, or `tester`
    




