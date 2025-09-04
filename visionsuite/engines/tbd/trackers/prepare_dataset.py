import os
import glob
import random
import shutil
from tqdm import tqdm

# --- 설정 (경로 수정) ---
# 🔴 데이터셋의 최상위 경로를 절대 경로로 직접 지정합니다.
# Windows 경로 사용 시, 역슬래시(\)를 두 번 쓰거나(r'E:\\...') 슬래시(/)로 바꿔주는 것이 좋습니다.
DATASET_DIR = 'E:/KIA_Vehicle_Dataset'

# 폴더 이름
IMAGE_DIR_NAME = 'images'
LABEL_DIR_NAME = 'labels'

# Train / Validation 분할 비율 (0.1 = 10%를 validation으로 사용)
VAL_SPLIT_RATIO = 0.1

# 클래스 이름
CLASS_NAME = 'tilt_car'
# -------------------------

# 전체 경로 설정
image_path = os.path.join(DATASET_DIR, IMAGE_DIR_NAME)
label_path = os.path.join(DATASET_DIR, LABEL_DIR_NAME)

print(f"--- 목표 폴더: {DATASET_DIR} ---")
print("--- 1. Labeled 데이터 기준으로 이미지 파일 정리 시작 ---")

# 1. 라벨 파일 목록을 기준으로 유효한 파일 이름(확장자 제외)을 set으로 만듭니다.
label_files = glob.glob(os.path.join(label_path, '*.txt'))
valid_basenames = {os.path.splitext(os.path.basename(f))[0] for f in label_files}
print(f"'{len(valid_basenames)}'개의 라벨링된 데이터를 찾았습니다.")

# 2. 이미지 폴더의 모든 파일을 확인하며, 라벨 파일이 없는 이미지는 삭제합니다.
all_image_files = glob.glob(os.path.join(image_path, '*.*'))
valid_image_files = []

for img_file in tqdm(all_image_files, desc="이미지 파일 정리 중"):
    basename = os.path.splitext(os.path.basename(img_file))[0]
    if basename in valid_basenames:
        valid_image_files.append(img_file)
    else:
        # 라벨이 없는 이미지 파일 삭제
        os.remove(img_file)

print(f"정리 완료. 총 '{len(valid_image_files)}'개의 이미지 파일이 남았습니다.")
print("-" * 50)

# --- train/val 분리 ---
print("--- 2. Train / Validation 데이터셋 분리 시작 ---")

# 3. train, val 폴더 생성
train_img_path = os.path.join(image_path, 'train')
val_img_path = os.path.join(image_path, 'val')
train_label_path = os.path.join(label_path, 'train')
val_label_path = os.path.join(label_path, 'val')

os.makedirs(train_img_path, exist_ok=True)
os.makedirs(val_img_path, exist_ok=True)
os.makedirs(train_label_path, exist_ok=True)
os.makedirs(val_label_path, exist_ok=True)

# 4. 파일 목록을 무작위로 섞고, 비율에 따라 분리
random.shuffle(valid_image_files)
split_index = int(len(valid_image_files) * VAL_SPLIT_RATIO)

val_images = valid_image_files[:split_index]
train_images = valid_image_files[split_index:]

# 5. 파일을 각각의 폴더로 이동
def move_files(file_list, img_dest_path, label_dest_path):
    for img_file in tqdm(file_list, desc=f"{os.path.basename(img_dest_path)} 세트로 이동 중"):
        basename = os.path.splitext(os.path.basename(img_file))[0]
        # 원본 라벨 폴더에서 해당 라벨 파일을 찾음
        source_label_file = os.path.join(label_path, basename + '.txt')

        # 이미지와 라벨 파일 이동
        shutil.move(img_file, os.path.join(img_dest_path, os.path.basename(img_file)))
        shutil.move(source_label_file, os.path.join(label_dest_path, os.path.basename(source_label_file)))

move_files(train_images, train_img_path, train_label_path)
move_files(val_images, val_img_path, val_label_path)

print(f"분리 완료: Train '{len(train_images)}'개, Validation '{len(val_images)}'개")
print("-" * 50)

# --- data.yaml 파일 생성 ---
print("--- 3. data.yaml 파일 생성 시작 ---")

# 경로 구분자를 OS에 맞게 자동으로 처리
safe_dataset_path = os.path.normpath(DATASET_DIR)

yaml_content = f"""
# 데이터셋 경로
path: {safe_dataset_path}

# 학습 및 검증 이미지 폴더 경로 (위 path 기준 상대 경로)
train: ./{IMAGE_DIR_NAME}/train
val: ./{IMAGE_DIR_NAME}/val

# 클래스 정보
nc: 1
names: ['{CLASS_NAME}']
"""

yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content.strip())

print(f"✅ 'data.yaml' 파일이 '{yaml_path}'에 생성되었습니다.")
print("모든 준비가 완료되었습니다. 이제 이 YAML 파일로 학습을 시작할 수 있습니다.")