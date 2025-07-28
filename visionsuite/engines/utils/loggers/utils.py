import datetime
import json
import os
import os.path as osp

# import socket

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def get_config(config_json, log_stream_level, log_file_level, log_dir, name=None):
    with open(config_json) as f:
        config = json.load(f)

    set_log_level(config, log_stream_level, log_file_level)
    set_log_dir(config, log_dir, name)

    return config


def set_log_level(config, log_stream_level, log_file_level):
    log_stream_level = log_stream_level.upper()
    log_file_level = log_file_level.upper()
    assert log_stream_level.upper() in get_log_levels(), ValueError(
        f"Log-level for streaming should be one of {LOG_LEVELS}"
    )
    assert log_file_level.upper() in get_log_levels(), ValueError(
        f"Log-level for file should be one of {LOG_LEVELS}"
    )

    config["handlers"]["stream"]["level"] = log_stream_level
    config["root"]["level"] = log_file_level


def set_log_dir(config, log_dir, name=None):
    if log_dir is None:
        # hostname = socket.gethostname()

        # log_dir = f'/home/{hostname}/logs'
        log_dir = "/DeepLearning/_logs/logger"
        current_date = datetime.datetime.now()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        hour = current_date.hour

        log_dir = osp.join(log_dir, f"{year}_{month}_{day}", str(hour))

        if not osp.exists(log_dir):
            os.makedirs(log_dir)

    assert log_dir is not None, ValueError(
        f"Log directory should be assigned, not {log_dir}"
    )

    if name is None:
        config["handlers"]["file"]["filename"] = osp.join(
            log_dir, config["handlers"]["file"]["filename"]
        )
    else:
        config["handlers"]["file"]["filename"] = osp.join(log_dir, name + ".log")


def get_log_levels():
    return LOG_LEVELS