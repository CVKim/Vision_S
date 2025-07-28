import pynvml
import numpy as np
import time

class GPULogger:
    def __init__(self, device_ids=[0]):
        self._device_ids = device_ids
        pynvml.nvmlInit()

        self.data = {}

    def update(self):
        for device_id in self._device_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_info = pynvml.nvmlDeviceGetPowerUsage(handle)

            if f'GPU {device_id}' not in self.data:
                self.data[f'GPU {device_id}'] = {"Util (%)": [utilization.gpu],
                                                 "Mem. (GB)": [float(round(memory_info.used / 1024**3, 2))],
                                                 "Power (W)": [float(round(power_info / 1000, 2))]
                                            }
            else:
                self.data[f'GPU {device_id}']["Util (%)"].append(utilization.gpu),
                self.data[f'GPU {device_id}']["Mem. (GB)"].append(float(round(memory_info.used / 1024**3, 2)))
                self.data[f'GPU {device_id}']["Power (W)"].append(float(round(power_info / 1000, 2)))

    def mean(self):
        ret = {}
        for key1, val1 in self.data.items():
            ret[key1] = {}
            for key2, val2 in val1.items():
                ret[key1].update({key2: round(np.mean(val2), 3)})
            
        return ret
    
    def max(self):
        ret = {}
        for key1, val1 in self.data.items():
            ret[key1] = {}
            for key2, val2 in val1.items():
                ret[key1].update({key2: round(max(val2), 3)})
            
        return ret

    def min(self):
        ret = {}
        for key1, val1 in self.data.items():
            ret[key1] = {}
            for key2, val2 in val1.items():
                ret[key1].update({key2: round(min(val2), 3)})
            
        return ret
    
    def clear(self):
        self.data = {}

    def end(self):
        pynvml.nvmlShutdown()
        
if __name__ == '__main__':
    logger = GPULogger([0, 1])
    
    for _ in range(3600):
        logger.update()
        time.sleep(1)
        
        print(logger.max())

