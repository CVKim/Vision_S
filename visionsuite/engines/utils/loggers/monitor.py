import copy
import os
import os.path as osp

import pandas as pd
import plotly.graph_objects as go


class Monitor:
    """Since using pandas as default, setting columns should be list and adding data should be dict."""

    def __init__(self):
        super().__init__()
        self.__output_dir = None
        self.__fn = None
        # TODO: need another extension?
        self.__ext = 'csv'
        self.__info = {}
        self.__cnt_log = 0
        self.__save_csv = True
        self.__save_figs = True
        self.__save_freq = 1

        self.__is_on = False

    @property
    def output_dir(self):
        return self.__output_dir

    @output_dir.setter
    def output_dir(self, value):
        assert osp.exists(value), ValueError(f"output_dir ({value}) does not exist")

        self.__output_dir = value

    @property
    def fn(self):
        return self.__fn

    @fn.setter
    def fn(self, value):
        assert isinstance(value, str), ValueError(f"fn ({value}) must string")
        assert len(value) != 0, ValueError(f"fn ({value}) must not be empty")

        self.__fn = value

    @property
    def is_on(self):
        return self.__is_on

    @is_on.setter
    def is_on(self, value):
        assert osp.exists(value), ValueError(f"is_on ({value}) does not exist")

        self.__is_on = value

    @property
    def save_csv(self):
        return self.__save_csv

    @save_csv.setter
    def save_csv(self, value):
        assert osp.exists(value), ValueError(f"save_csv ({value}) does not exist")

        self.__save_csv = value

    @property
    def save_figs(self):
        return self.__save_figs

    @save_figs.setter
    def save_figs(self, value):
        assert osp.exists(value), ValueError(f"save_figs ({value}) does not exist")

        self.__save_figs = value

    @property
    def save_freq(self):
        return self.__save_freq

    @save_freq.setter
    def save_freq(self, value):
        assert osp.exists(value), ValueError(f"save_freq ({value}) does not exist")

        self.__save_freq = value

    @property
    def info(self):
        return self.__info

    def set(self, output_dir, fn, use=True, save_csv=True, save_figs=True, save_freq=1):
        self.is_on = use
        if use:
            self.__output_dir = output_dir
            self.__fn = fn

            self.save_csv = save_csv
            self.save_figs = save_figs
            self.save_freq = save_freq

    def log(self, data):
        if self.is_on:
            if isinstance(data, dict):
                for key, val in data.items():
                    if key not in self.info.keys():
                        self.info[key] = [val]
                    else:
                        self.info[key].append(val)

                self.__cnt_log += 1
            else:
                raise TypeError(f"Logging data should be dict, not {type(data)}")

    def save(self):
        if self.is_on:
            if self.__ext == 'csv':
                if self.__cnt_log % self.save_freq == 0:
                    try:
                        _df = pd.DataFrame.from_dict(self.info)
                        if not osp.exists(self.__output_dir):
                            os.mkdir(self.__output_dir)
                        _df.to_csv(osp.join(self.__output_dir, self.__fn + '.csv'), index=False)
                    except Exception as error:
                        print(f"There has been error: {error}")
                        __info = copy.deepcopy(self.info)
                        cnt1, cnt2 = 0, 1
                        while True:
                            _info = {}
                            _saved_list = []
                            for idx, (key, val) in enumerate(__info.items()):
                                if key not in _saved_list:
                                    if idx == 0:
                                        _length = len(val)
                                    if len(val) == _length:
                                        _info.update({key: val})
                                        _saved_list.append(key)
                                        cnt1 += 1

                            _df = pd.DataFrame.from_dict(_info)
                            if not osp.exists(self.__output_dir):
                                os.mkdir(self.__output_dir)
                            if self.save_csv:
                                _df.to_csv(osp.join(self.__output_dir, self.__fn + '_{}.csv'.format(cnt2)), index=False)
                            cnt2 += 1

                            for _saved_key in _saved_list:
                                if _saved_key in __info.keys():
                                    del __info[_saved_key]

                            if cnt1 == len(self.info):
                                break

                    if self.save_figs:
                        try:
                            for key, val in self.info.items():
                                x_values = list(range(len(self.info[key])))  # Convert range to a list
                                fig = go.Figure(data=go.Scatter(x=x_values, y=self.info[key]))
                                fig.update_layout(title=key,
                                                  xaxis_title='Index',
                                                  yaxis_title=key,
                                                  height=500,
                                                  width=1000)
                                fig.write_image(osp.join(self.__output_dir, '{}_{}.png'.format(self.__fn, key)))
                        except Exception as error:
                            print(f"Cannot make figure: {error}")

            else:
                raise TypeError(f"Cannot save that format: {self.__ext}")
