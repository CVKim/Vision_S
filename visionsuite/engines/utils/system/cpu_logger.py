import psutil
import numpy as np

class CPULogger:
    def __init__(self):
        self.data = {'Power (%)': []}

    def update(self):
        cpu_usage = psutil.cpu_percent(interval=0.01)
        
        self.data['Power (%)'].append(cpu_usage)
        
    def mean(self):
        mean_data = {}
        for key1, val1 in self.data.items():
            mean_data[key1] = round(np.mean(val1), 3)
            
        return mean_data

    def clear(self):
        self.data = {}

if __name__ == '__main__':
    logger = CPULogger()
    for _ in range(10):
        logger.update()
        
    print(logger.mean())