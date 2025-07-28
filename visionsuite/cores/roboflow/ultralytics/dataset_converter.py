from typing import Union 

from visionsuite.cores.roboflow.utils import labelme2yolo_iseg, labelme2yolo_hbb, labelme2yolo_obb

class DatasetConverter:
    
    def __init__(self, task):
        self.__task = task
        self.input_dir = None
        self.output_dir = None 
        self.copy_image = True 
        self.image_ext = 'bmp'

        self._set_converter()
        
    @property 
    def task(self):
        return self.__task 
    
    @task.setter
    def task(self, value):
        self.__task = value
    
    def _set_converter(self):
        if self.__task == 'hbb_detection':
            self._converter = labelme2yolo_hbb
        elif self.__task == 'instance_segmentation':
            self._converter = labelme2yolo_iseg
        elif self.__task == 'obb_detection':
            self._converter = labelme2yolo_obb
        else:
            raise NotImplementedError(f"There is no such converter for {self.__task}")
        
        print(f"SELECTED converter is {self._converter.__name__}")
        
    def run(self, input_dir: str, copy_image: bool=True, image_ext: Union[list, str]='bmp', output_dir: str=None):
                      
        def _set_output_dir(input_dir, output_dir):
            if output_dir is None:
                if self.__task == 'hbb_detection':
                    output_dir = input_dir + '_yolo_hbb'
                elif self.__task == 'instance_segmentation':
                    output_dir = input_dir + '_yolo_iseg'
                elif self.__task == 'obb_detection':
                    output_dir = input_dir + '_yolo_obb'

            return output_dir
        
        output_dir = _set_output_dir(input_dir, output_dir)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.copy_image = copy_image
        self.image_ext = image_ext
        
        self._converter(input_dir=input_dir, output_dir=output_dir, copy_image=copy_image, image_ext=image_ext)
            
        
if __name__ == '__main__':
    # input_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/split_dataset'

    # copy_image = True
    # image_ext = 'bmp'
       
    # converter = DatasetConverter('obb_detection')
    # converter.run(input_dir, copy_image=copy_image, image_ext=image_ext)
    
    input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/warp/TOP_BOT_CHAMFER_WARP_ALL/241204/split_labelme_patch_dataset'

    copy_image = True
    image_ext = 'bmp'
       
    converter = DatasetConverter('hbb_detection')
    converter.run(input_dir, copy_image=copy_image, image_ext=image_ext)
    
            