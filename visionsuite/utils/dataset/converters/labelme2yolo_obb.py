from .labelme2dota import convert_labelme2dota
from .dota2yolo import convert_dota2yolo_obb

def convert_labelme2yolo_obb(input_dir: str, output_dir: str=None):

    output_dir = input_dir + '_dota'

    convert_labelme2dota(input_dir, output_dir, copy_image=True, image_ext='bmp')

    input_dir = output_dir
    output_dir = input_dir + '_yolo_obb'

    convert_dota2yolo_obb(input_dir, output_dir, copy_image=True, image_ext='bmp')
    
if __name__ == '__main__':
    input_dir = '/HDD/datasets/projects/kt_g/datasets/split_dataset'
    copy_image = True
    image_ext = 'bmp'
    
    convert_labelme2yolo_obb(input_dir, copy_image=copy_image, image_ext=image_ext)
