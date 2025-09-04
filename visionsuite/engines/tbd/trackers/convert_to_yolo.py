import xml.etree.ElementTree as ET
import glob
import os
from tqdm import tqdm

CLASSES = ["tilt_car"] 

XML_DIR = "E:/KIA_Vehicle_Dataset/labels"
YOLO_DIR = "E:/KIA_Vehicle_Dataset/labels_yolo"

def convert_voc_to_yolo(xml_file, output_dir, classes):
    """
    하나의 Pascal VOC(xml) 파일을 YOLO(.txt) 형식으로 변환합니다.
    """
    # XML 파일 파싱
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 이미지 크기 정보 가져오기
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    yolo_labels = []
    for obj in root.iter('object'):
        # 클래스 이름 확인
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        class_index = classes.index(class_name)

        # Bounding Box 좌표(xmin, ymin, xmax, ymax) 가져오기
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # YOLO 형식으로 변환 (정규화)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_labels.append(f"{class_index} {x_center} {y_center} {width} {height}")

    # YOLO 라벨 파일(.txt) 저장
    basename = os.path.splitext(os.path.basename(xml_file))[0]
    output_txt_path = os.path.join(output_dir, basename + '.txt')
    with open(output_txt_path, 'w') as f:
        f.write("\n".join(yolo_labels))


# --- 메인 코드 실행 ---
if __name__ == '__main__':
    os.makedirs(YOLO_DIR, exist_ok=True)
    xml_files = glob.glob(os.path.join(XML_DIR, '*.xml'))
    
    if not xml_files:
        print(f"'{XML_DIR}' 폴더에 XML 파일이 없습니다. 경로를 확인해주세요.")
    else:
        for xml_file in tqdm(xml_files, desc="Converting XML to YOLO"):
            convert_voc_to_yolo(xml_file, YOLO_DIR, CLASSES)
        print(f"✅ 변환 완료! '{len(xml_files)}'개의 파일이 '{YOLO_DIR}' 폴더에 저장되었습니다.")