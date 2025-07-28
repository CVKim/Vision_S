from visionsuite.engines.utils.loggers.monitor import Monitor

monitor = Monitor()
monitor.set(output_dir='/HDD/etc/logger', fn='test-monitor')

for idx in range(10):
    monitor.log({"a": idx})
    
monitor.save()