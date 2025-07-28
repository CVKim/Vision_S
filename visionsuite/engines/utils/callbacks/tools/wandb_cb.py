import subprocess

try:
    import wandb as wb
except (ImportError, AssertionError):
    subprocess.run(["pip", "install", "wandb"])
    import wandb as wb

# assert hasattr(wb, '__version__')  # verify package is not directory
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

_processed_plots = {}


def _custom_table(
    x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"
):
    """
    Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of wandb's default precision-recall curve
    while allowing for enhanced customization. The visual metric is useful for monitoring model performance across different classes.

    Args:
        x (List): Values for the x-axis; expected to have length N.
        y (List): Corresponding values for the y-axis; also expected to have length N.
        classes (List): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot; defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis; defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis; defaults to 'Precision'.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    df = pd.DataFrame({"class": classes, "y": y, "x": x}).round(3)
    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wb.plot_table(
        "wandb/area-under-curve/v0",
        wb.Table(dataframe=df),
        fields=fields,
        string_fields=string_fields,
    )


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """
    Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb.
    The curve can represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.

    Args:
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape CxN, where C represents the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C. Defaults to an empty list.
        id (str, optional): Unique identifier for the logged data in wandb. Defaults to 'precision-recall'.
        title (str, optional): Title for the visualization plot. Defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis. Defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis. Defaults to 'Precision'.
        num_x (int, optional): Number of interpolated data points for visualization. Defaults to 100.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted. Defaults to True.

    Note:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    # Create new x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # Create arrays for logging
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # add new x
            y_log.extend(np.interp(x_new, x, yi))  # interpolate y to new x
            classes.extend([names[i]] * len(x_new))  # add class names
        wb.log(
            {id: _custom_table(x_log, y_log, classes, title, x_title, y_title)},
            commit=False,
        )


def _log_plots(plots, step):
    """Logs plots from the input dictionary if they haven't been logged already at the specified step."""
    for name, params in plots.items():
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_runner_set_variables(engine):
    try:
        engine.flags.is_wandb_logined = wb.login(
            key="817e3db4693a77186b46ea420d9fcb5bdd74f4d8"
        )
        name = ""
        if hasattr(engine.configs, "db") and hasattr(
            engine.configs.db, "sub_project_name"
        ):
            name += f"{engine.configs.db.sub_project_name}"
        if hasattr(engine.configs, "logging") and hasattr(
            engine.configs.logging, "logs_dir"
        ):
            name += f"_{str(Path(engine.configs.logging.logs_dir).parent.parent).split('/')[-1]}"

        wb.run or wb.init(
            project=engine.configs.db.project_name,
            name=name,
            config=OmegaConf.to_container(engine.configs.__dict__["_configs"]),
        )

        engine.flags.is_wandb_initialized = True

        if engine._callbacks.logger is not None:
            engine._callbacks.logger.info(f"Logined WanDB: ", on_runner_set_variables.__name__)
            engine._callbacks.logger.info(
                f" - project name: {name}", on_runner_set_variables.__name__
            )

    except Exception as error:
        engine.flags.is_wandb_logined, engine.flags.is_wandb_initialized = False, False
        if engine._callbacks.logger is not None:
            engine._callbacks.logger.info(f"CANNOT connect to WanDB b/c {error}")


def on_fit_epoch_end(engine):
    """Logs training metrics and model information at the end of an epoch."""
    if engine.flags.is_wandb_logined and engine.flags.is_wandb_initialized:
        wb.run.log(engine.metrics, step=engine.epoch + 1)
        _log_plots(engine.plots, step=engine.epoch + 1)
        _log_plots(engine.validator.plots, step=engine.epoch + 1)
    # if engine.epoch == 0:
    #     wb.run.log(model_info_for_loggers(engine), step=engine.epoch + 1)


def on_trainer_epoch_end(engine):
    """Log metrics and save images at the end of each training epoch."""
    # wb.run.log(engine.label_loss_items(engine.tloss, prefix='train'), step=engine.epoch + 1)
    # wb.run.log(engine.lr, step=engine.epoch + 1)
    if engine.flags.is_wandb_logined and engine.flags.is_wandb_initialized:
        log = {}
        for key, val in engine.train_epoch_results.log.items():
            log["train_" + key] = val
        wb.log(log)
        if engine._callbacks.logger is not None:
            engine._callbacks.logger.info(
                f"Logged train results of one epoch into WanDB",
                on_trainer_epoch_end.__name__,
            )
        # if engine.epoch == 1:
        #     _log_plots(engine.plots, step=engine.epoch + 1)


def on_train_end(engine):
    """Save the best model as an artifact at end of training."""
    if engine.flags.is_wandb_logined and engine.flags.is_wandb_initialized:
        _log_plots(engine.validator.plots, step=engine.epoch + 1)
        _log_plots(engine.plots, step=engine.epoch + 1)
        art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
        if engine.best.exists():
            art.add_file(engine.best)
            wb.run.log_artifact(art, aliases=["best"])
        for curve_name, curve_values in zip(
            engine.validator.metrics.curves, engine.validator.metrics.curves_results
        ):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(engine.validator.metrics.names.values()),
                id=f"curves/{curve_name}",
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )
        wb.run.finish()  # required or run continues on dashboard
        if engine._callbacks.logger is not None:
            engine._callbacks.logger.info(
                f"Logged final train results of one epoch into WanDB",
                on_train_end.__name__,
            )


def on_val_end(engine):
    if engine.flags.is_wandb_logined and engine.flags.is_wandb_initialized:
        log = {}
        for key, val in engine.val_results.log.items():
            log["val_" + key] = val
        wb.log(log)

        if engine._callbacks.logger is not None:
            engine._callbacks.logger.info(
                f"Logged val. results of one epoch into WanDB", on_val_end.__name__
            )


callbacks = (
    {
        "on_runner_set_variables": on_runner_set_variables,
        "on_trainer_epoch_end": on_trainer_epoch_end,
        "on_val_end": on_val_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
