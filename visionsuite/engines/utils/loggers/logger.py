import datetime
import json
import logging
import logging.config
import os
import os.path as osp
import sys

# import socket
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Logger:
    """
    arguments:
        - log_dir: directory to save log file
        - log_stream_level: one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', and 'CRITICAL'
        - log_file_level: one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', and 'CRITICAL'
        - config_json: default is already set
    """

    def __init__(self, name=None):
        if name is None:
            self.__name = "no_name"
        else:
            self.__name = name
        self._logger = None
        self.is_logger_set = False

        self.__log_dir = None
        self.__log_stream_level = "DEBUG"
        self.__log_file_level = "DEBUG"
        self.__config_json = None
        self.__log_filename = None

        self.__config = None

    def __repr__(self):
        if self._logger is None:
            return f"The logger({self._logger}) is not yet defined. Need to execute 'set_logger' first"
        else:
            return f"This logger is for {self.name} and locates in {self.log_dir if self.log_filename is None else self.log_filename} with file-level({self.__log_file_level}) and stream-level({self.__log_stream_level})"

    @property
    def log_dir(self):
        return self.__log_dir

    @log_dir.setter
    def log_dir(self, value):
        self.__log_dir = value

    @property
    def log_filename(self):
        return self.__log_filename

    @log_filename.setter
    def log_filename(self, value):
        self.__log_filename = value

    @property
    def log_stream_level(self):
        return self.__log_stream_level

    @log_stream_level.setter
    def log_stream_level(self, value):
        assert value.upper() in self._get_log_levels(), ValueError(
            f"Log-level for streaming should be one of {LOG_LEVELS}, not {value}"
        )
        self.__log_stream_level = value.upper()

    @property
    def log_file_level(self):
        return self.__log_file_level

    @log_file_level.setter
    def log_file_level(self, value):
        assert value.upper() in self._get_log_levels(), ValueError(
            f"Log-level for file should be one of {LOG_LEVELS}, not {value}"
        )
        self.__log_file_level = value.upper()

    @property
    def config_json(self):
        return self.__config_json

    @config_json.setter
    def config_json(self, value):
        assert osp.exists(value), ValueError(f"There is no such config_json({value})")

        self.__config_json = value

    @property
    def config(self):
        return self.__config

    @config.setter
    def config(self, value):
        self.__config = value

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    def set_logger(
        self,
        log_stream_level="DEBUG",
        log_file_level="DEBUG",
        log_dir=None,
        config_json=osp.join(ROOT, "data/colored-logging.json"),
    ):
        if self.is_logger_set:
            self.log_info(f"Logger is already set for {self._name}")
            return 
        self.log_stream_level = log_stream_level.upper()
        self.log_file_level = log_file_level.upper()
        self.log_dir = log_dir
        self.config_json = config_json

        # FIXME: logging.config.dictlogging.config.dictConfigConfig가 오버랩되어서 사용되어 다중 logger가 불가능,,, json을 변경해야한다고 함
        # try:
        #     self._get_config()
        #     logging.config.dictConfig(self.config)
        #     self._logger = logging.getLogger(self.name)
        #     self.is_logger_set = True
        # except Exception as e:
        #     warnings.warn(f"Cannot define logger: {e}")
        #     try:
        #         self.configs_json = osp.join(ROOT, 'data/logging.json')
        #         self._get_config()
        #         logging.config.dictConfig(self.config)
        #         self._logger = logging.getLogger(self.name)
        #         self.is_logger_set = True
        #     except Exception as e:
        #         warnings.warn(f"Cannot define logger: {e}")
        #         self._logger = None
        #         self.is_logger_set = False

        # self.log_info(f"Logger has been set for {self.__name}", self.__init__.__name__, __class__.__name__)

        try:
            self._get_config()
            self._logger = logging.getLogger(self.name)
            self._logger.setLevel(self.log_stream_level)
            file_handler = logging.FileHandler(
                osp.join(self.__log_dir, self.__name + ".log")
            )
            file_handler.setLevel(self.log_file_level)
            file_handler.setFormatter(
                logging.Formatter(self.config["formatters"]["colored"]["format"])
            )
            self._logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_stream_level)
            console_handler.setFormatter(
                logging.Formatter(self.config["formatters"]["colored"]["format"])
            )
            self._logger.addHandler(console_handler)

            self.is_logger_set = True
        except Exception as e:
            warnings.warn(f"Cannot define logger: {e}")
            try:
                self.configs_json = osp.join(ROOT, "data/logging.json")
                self._get_config()
                self._logger = logging.getLogger(self.name)
                file_handler = logging.FileHandler(
                    osp.join(self.__log_dir, self.__name + ".log")
                )
                file_handler.setFormatter(
                    logging.Formatter(self.config["formatters"]["colored"]["format"])
                )
                self._logger.addHandler(file_handler)

                console_handler = logging.StreamHandler()
                console_handler.setFormatter(
                    logging.Formatter(self.config["formatters"]["colored"]["format"])
                )
                self._logger.addHandler(console_handler)
                self.is_logger_set = True
            except Exception as e:
                warnings.warn(f"Cannot define logger: {e}")
                self._logger = None
                self.is_logger_set = False

        self.log_info(
            f"Logger has been set for {self.__name}",
            self.__init__.__name__,
            __class__.__name__,
        )

    def log_debug(self, msg, func_name="", class_name=""):
        if self._logger is not None:
            self._logger.debug(f"[{class_name}] - [{func_name}] - {msg}")

    def log_info(self, msg, func_name="", class_name=""):
        if self._logger is not None:
            self._logger.info(f"[{class_name}] - [{func_name}] - {msg}")

    def log_warning(self, msg, func_name="", class_name=""):
        if self._logger is not None:
            self._logger.warning(f"[{class_name}] - [{func_name}] - {msg}")

    def log_error(self, msg, func_name="", class_name=""):
        if self._logger is not None:
            self._logger.error(f"[{class_name}] - [{func_name}] - {msg}")

    def log_critical(self, msg, func_name="", class_name=""):
        if self._logger is not None:
            self._logger.critical(f"[{class_name}] - [{func_name}] - {msg}")

    def _get_config(self):
        with open(self.__config_json) as f:
            self.__config = json.load(f)

        self._set_log_level()
        self._set_log_dir()

    def _set_log_level(self):
        self.__config["handlers"]["stream"]["level"] = self.__log_stream_level
        self.__config["root"]["level"] = self.__log_file_level

    def _set_log_dir(self):
        if self.__log_dir is None:
            # hostname = socket.gethostname()

            # log_dir = f'/home/{hostname}/logs'
            self.__log_dir = f"/DeepLearning/_logs/logger"
            current_date = datetime.datetime.now()
            year = current_date.year
            month = current_date.month
            day = current_date.day
            hour = current_date.hour

            self.__log_dir = osp.join(
                self.__log_dir, f"{year}_{month}_{day}", str(hour)
            )

        if not osp.exists(self.__log_dir):
            os.makedirs(self.__log_dir)

        assert self.__log_dir is not None, ValueError(
            f"Log directory should be assigned, not {self.__log_dir}"
        )

        if self.__name is None:
            self.log_filename = osp.join(
                self.__log_dir, self.__config["handlers"]["file"]["filename"]
            )
        else:
            self.log_filename = osp.join(self.__log_dir, self.__name + ".log")
        self.__config["handlers"]["file"]["filename"] = self.log_filename

    def _get_log_levels(self):
        return LOG_LEVELS

    # TODO: Need to extract the argument for exit value
    def try_except_log(
        self,
        func,
        msg="",
        post_action=None,
        exit=False,
        parent_class=None,
        parent_fn=None,
    ):
        try:
            if msg == "In the post-action, ":
                if self._logger:
                    self._logger.info("Post-action runs after raising error")
            func()
        except Exception as error_msg:
            error_type = type(error_msg).__name__
            if len(msg) != 0:
                error_msg = msg + ", and " + str(error_msg)
            else:
                error_msg = str(error_msg)
            if self._logger:
                self._logger.error(
                    f" [{self.try_except_log.__name__}] {error_type}: {error_msg} at [{parent_fn}] of [{parent_class}]"
                )

            if post_action is not None:
                self.try_except_log(post_action, msg="In the post-action, ")
            if error_type in __builtins__:
                if not exit:
                    raise __builtins__[error_type](error_msg)
                else:
                    sys.exit()

    def assertion_log(
        self,
        condition,
        error,
        msg="",
        post_action=None,
        exit=False,
        parent_class=None,
        parent_fn=None,
    ):
        try:
            if msg == "In the post-action, ":
                if self._logger:
                    self._logger.info("Post-action runs after raising error")
            assert condition
        except AssertionError:
            error_type = type(error).__name__
            if len(msg) != 0:
                error_msg = msg + ", and " + str(error)
            else:
                error_msg = str(error)
            if self._logger:
                self._logger.error(
                    f" [{self.assertion_log.__name__}] {error_type}: {error_msg} at [{parent_fn}] of [{parent_class}]"
                )

            if post_action is not None:
                self.try_except_log(post_action, msg="In the post-action, ")
            if not exit:
                raise AssertionError(error_msg)
            else:
                sys.exit()

    def raise_error_log(self, error_obj, error_msg):
        if self._logger:
            self._logger.error(f"[{error_obj.__name__}] {error_msg}")
        if error_obj.__name__ in __builtins__:
            raise __builtins__[error_obj.__name__](error_msg)
        else:
            sys.exit()
