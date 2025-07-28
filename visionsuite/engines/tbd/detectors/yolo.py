
try:
    from ultralytics import YOLO 
except Exception as error:
    print(f"There has been error to import YOLO from ultralytics:{error}")
    import subprocess as sb 
    sb.run(['pip', 'install', 'ultralytics'])
    
    from ultralytics import YOLO 
    

class YoloDetector:
    def __init__(self, model_name):
        self._model_name = model_name 

    
    def _set_model(self):
        pass