from abc import ABCMeta

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class BaseConfig(BaseModel, metaclass=ABCMeta):

    def add_field(self, key, value):
        self.__dict__[key] = value

    def to_dict(self):
        _dict = {}
        # for key, val in self.model_dump().items():
        #     if isinstance(val, DictConfig):
        #         _dict[key] = OmegaConf.to_container(val)
        #     else:
        #         _dict[key] = val
        for key, val in dict(self).items():
            if isinstance(val, DictConfig):
                _dict[key] = OmegaConf.to_container(val)
            elif isinstance(val, BaseModel):
                _dict[key] = val.dict()
            else:
                _dict[key] = val

        return _dict

    def json_schema(self):
        return self.model_json_schema(mode='serialization')
