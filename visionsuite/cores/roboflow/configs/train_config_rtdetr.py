from pydantic import Field
from enum import Enum
from typing import Any, Literal

from visionsuite.utils.helpers import string_to_list_of_type
from visionsuite.utils.configs.base_config import BaseConfig

class DeviceConfig(str, Enum):
    gpu_0 = "0"
    gpu_1 = "1"
    gpu_2 = "2"
    gpu_3 = "3"

class TrainConfig(BaseConfig):
    # required
    backbone: Literal['l', 'x'] = Field(frozen=True)
    epochs: int = Field(frozen=True)
    batch: int = Field(frozen=True)
    imgsz: int = Field(frozen=True)
    device: DeviceConfig = Field('0', frozen=True)
    
    #
    conf: float = Field(0.25, frozen=True)
    iou: float = Field(0.7, frozen=True)

    # 
    lrf: float = Field(0.001, frozen=True)
    label_smoothing: float = Field(0.1, frozen=True)
    degrees: float = Field(0.0, frozen=True)
    translate: float = Field(0.1, frozen=True)
    scale: float = Field(0.5, frozen=True)
    flipud: float = Field(0.0, frozen=True)
    fliplr: float = Field(0.5, frozen=True)
    mosaic: float = Field(1.0, frozen=True)
    
    def __init__(self, **data):
        super().__init__(**data)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        
        self._device_post_init()
        
    def _device_post_init(self):
        if isinstance(self.device, str):
            object.__setattr__(self, 'device', string_to_list_of_type(self.device, int))

