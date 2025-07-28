from enum import Enum
from typing import Any, Literal, Union

from omegaconf import ListConfig
from pydantic import BaseModel, Field

from visionsuite.utils.helpers import string_to_list_of_type


class DeviceIdsConfig(str, Enum):
    gpu_0 = "0"
    gpu_1 = "1"
    gpu_2 = "2"
    gpu_3 = "3"


class DeviceConfig(BaseModel):

    device: Literal['cpu', 'gpu', 'cuda'] = Field('gpu', frozen=True)
    device_ids: DeviceIdsConfig = Field('0', frozen=True)

    tmp_device_ids: Union[str, None] = Field(None, exclude=True)

    def __init__(self, **data):
        if 'device_ids' in data.keys():
            if isinstance(data['device_ids'], str):
                self._set_tmp_device_ids(data)
            elif isinstance(data['device_ids'], (list, ListConfig)):
                if len(data['device_ids']) == 0:
                    ValueError(f"Device-id must be defined, not f{data['device_ids']}")
                else:
                    if isinstance(data['device_ids'][0], str):
                        data['device_ids'] = ','.join(data['device_ids'])
                        self._set_tmp_device_ids(data)
                    elif isinstance(data['device_ids'][0], int):
                        data['device_ids'] = ','.join(map(str, data['device_ids']))
                        self._set_tmp_device_ids(data)
                    else:
                        NotImplementedError(
                            f"This case is not considered: {type(data['device_ids']), data['device_ids']}")
            elif isinstance(data['device_ids'], int):
                data['device_ids'] = str(data['device_ids'])
            else:
                NotImplementedError(f"This case is not considered: {type(data['device_ids']), data['device_ids']}")

        super().__init__(**data)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._device_post_init()
        self._device_ids_post_init()

    def _set_tmp_device_ids(self, data):
        if len(data['device_ids']) != 0:
            data['tmp_device_ids'] = data['device_ids']
            data['device_ids'] = data['device_ids'][0]
        else:
            ValueError(f"Device-id must be defined, not f{data['device_ids']}")

    def _device_ids_post_init(self):
        if self.tmp_device_ids is not None:
            object.__setattr__(self, 'device_ids', string_to_list_of_type(self.tmp_device_ids, int))

        elif isinstance(self.device_ids, str):
            object.__setattr__(self, 'device_ids', string_to_list_of_type(self.device_ids, int))

    def _device_post_init(self):
        if self.device == 'gpu':
            object.__setattr__(self, 'device', 'cuda')

    @property
    def device_ids_list(self):
        if isinstance(self.device_ids, str):
            return string_to_list_of_type(self.device_ids, int)
        else:
            return self.device_ids
