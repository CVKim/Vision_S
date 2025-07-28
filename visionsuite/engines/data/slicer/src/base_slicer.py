import os.path as osp

from visionsuite.engines.utils.loggers import Logger
from omegaconf.listconfig import ListConfig


class BaseSlicer:

    def __init__(self, child=None):
        # input
        self._child = child
        self.logger = None

        self.__mode = None
        self.__input_dir = None
        self.__exts = []
        self.__classes = []
        self._roi_info = None
        self._patch_info = None
        self._roi_from_json = False

        # output
        self.__num_data = 0
        self.__img_files = []
        self.__imgs_info = []  # final output
        """
        - imgs_info: [
                {
                    'img_file' : image file name, # string
                    'patches': [
                                [x1, y1, x2, y2],
                                [x1, y1, x2, y2],
                                ...
                            ], # list of lists
                    'counts': [0, 0, ...], # list
                    'backgrounds': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...], # list of lists
                    'labels': [
                                [{'label': 'abc', 'points': [[x1, y1], [x2, y2], ...], 'shape_type': '', 'roi': [x1, y1, x2, y2]},
                                 {'label': '', 'points': [[x1, y1], [x2, y2], ...], 'shape_type': '', 'roi': [x1, y1, x2, y2]}, ...],
                                [{'label': '', 'points': [[x1, y1], [x2, y2], ...], 'shape_type': '', 'roi': [x1, y1, x2, y2]},
                                 {'label': '', 'points': [[x1, y1], [x2, y2], ...], 'shape_type': '', 'roi': [x1, y1, x2, y2]}, ...],
                                ...
                            ] # list of lists
                },
                {
                    'img_file' : image file name,
                    'patches': ...,
                    ...
                },
                ...
        ]
            - each patch coordinate is synched to each labels list and count.
        - backgrounds exist only when using patch-based training
    """

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, value):
        assert isinstance(value, str), ValueError(
            f"Mode variable must be string, not {type(value)}"
        )
        self.__mode = value

    @property
    def input_dir(self):
        return self.__input_dir

    @input_dir.setter
    def input_dir(self, value):
        assert isinstance(value, str), ValueError(
            f"Input directory variable must be string, not {type(value)}"
        )
        assert osp.exists(value), ValueError(f"There is no such directory: {value}")

        self.__input_dir = value

    @property
    def exts(self):
        return self.__exts

    @exts.setter
    def exts(self, value):
        assert isinstance(value, str) or (
            isinstance(value, (list, ListConfig))
            and len(value) != 0
            and isinstance(value[0], str)
        ), ValueError(
            f"Image extension variable must be string or list of string, not {type(value)}"
        )

        self.__exts = value

    @property
    def classes(self):
        return self.__classes

    @classes.setter
    def classes(self, value):
        assert (
            isinstance(value, (list, ListConfig))
            and len(value) != 0
            and isinstance(value[0], str)
        ), ValueError(f"classes variable must be list of string, not {type(value)}")
        self.__classes = value

    @property
    def imgs_info(self):
        return self.__imgs_info

    @imgs_info.setter
    def imgs_info(self, value):
        # assert value > 0, ValueError(f"Number of data must be higher than 0, not {value}")
        self.__imgs_info.append(value)

    @property
    def roi_info(self):
        return self._roi_info

    @roi_info.setter
    def roi_info(self, value):
        # assert value > 0, ValueError(f"Number of data must be higher than 0, not {value}")
        self._roi_info = value

    def set_log(self, log_dir=None, log_stream_level="DEBUG", log_file_level="DEBUG"):

        if self._child is not None:
            log_name = self._child
        else:
            log_name = None
        self._logger = Logger(name=log_name)
        self._logger.set(
            log_dir=log_dir,
            log_stream_level=log_stream_level,
            log_file_level=log_file_level,
        )
