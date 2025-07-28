import os 
import os.path as osp
import tensorflow as tf 
from tqdm import tqdm
import json 
import numpy as np 
import cv2 
from PIL import Image
import io

def create_mask_from_json(json_path, img_size, class_names):
    # LabelMe 어노테이션 파싱
    with open(json_path, 'r') as f:
        label_data = json.load(f)
    
    # 빈 마스크 생성
    mask = np.zeros(img_size, dtype=np.uint8)  # (height, width)
    
    # 각 객체별로 마스크 채우기
    for shape in label_data['shapes']:
        if shape['shape_type'] in ['line', 'point']:
            continue
        class_id = class_names.index(shape['label']) + 1 # add background
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=class_id)
    
    return mask

def labelme2tfrecord_auto_shard(data_root, output_dir, split='train', max_shard_size_mb=200, class_names=None):
    
    SHARD_MAX_BYTES = max_shard_size_mb * 1024 * 1024
    img_dir = os.path.join(data_root, split)
    img_extensions = ['*.bmp', '*.png', '*.jpeg', '*.jpg']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(tf.io.gfile.glob(os.path.join(img_dir, ext)))
    img_paths = sorted(img_paths)

    # 샤드 초기화
    shard_idx = 0
    current_shard_size = 0
    writer = None

    for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        # 이미지 로드
        image = tf.io.read_file(img_path)
        image_np = np.array(Image.open(io.BytesIO(image.numpy())))
        height, width = image_np.shape[:2]

        # JSON 어노테이션 로드
        json_path = os.path.splitext(img_path)[0] + '.json'
        if not tf.io.gfile.exists(json_path):
            print(f"Warning: JSON not found for {img_path}")
            continue

        # 마스크 생성
        mask = create_mask_from_json(json_path, 
            img_size=(height, width),
            class_names=class_names
        ).astype(np.uint8)

        # 마스크를 PNG로 인코딩 (메모리 상에서)
        mask_img = Image.fromarray(mask)
        mask_bytes_io = io.BytesIO()
        mask_img.save(mask_bytes_io, format='PNG')
        mask_bytes = mask_bytes_io.getvalue()

        shape_tensor = tf.convert_to_tensor([height, width], dtype=tf.int64)

        # TFRecord 예제 직렬화
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()])),
            'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_bytes])),
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[os.path.basename(img_path).encode()])),
            'original_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'original_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        }))
        
        example_bytes = example.SerializeToString()

        # 샤드 용량 체크
        if writer is None or current_shard_size + len(example_bytes) > SHARD_MAX_BYTES:
            if writer: writer.close()
            output_path = os.path.join(output_dir, f'labelme_{split}-{shard_idx:05d}.tfrecord')
            writer = tf.io.TFRecordWriter(output_path)
            shard_idx += 1
            current_shard_size = 0

        writer.write(example_bytes)
        current_shard_size += len(example_bytes)

    if writer: writer.close()


def build_optimized_dataset(tfrecord_dir, batch_size, 
                            cache=False, shuffle_buffer=300,
                            image_format='bmp',
                            shuffle=True,
                            one_hot_encoding=False,
                            split='train', roi=None, fp16=False):
    options = tf.data.Options()
    options.threading.private_threadpool_size = os.cpu_count()  # CPU 코어 수에 맞춤
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_and_batch_fusion = True
    
    files = tf.data.Dataset.list_files(f'{tfrecord_dir}/labelme_{split}-*.tfrecord', 
                                shuffle=shuffle).with_options(options)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False  # 성능 향상을 위해 비결정적 순서 허용
    )

    def parse_fn(example):
        feature_desc = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
            'filename': tf.io.FixedLenFeature([], tf.string),
            'original_height': tf.io.FixedLenFeature([], tf.int64),
            'original_width': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(example, feature_desc)
        height = tf.cast(parsed['original_height'], tf.int64)
        width = tf.cast(parsed['original_width'], tf.int64)
        tf.debugging.assert_positive(height, "Invalid height")
        tf.debugging.assert_positive(width, "Invalid width")

        image = tf.image.decode_image(parsed['image'], channels=3, expand_animations=False)
        image = tf.image.resize(image, [height, width])# / 255.0  # [0,1] 정규화
        if roi:
            image = tf.image.crop_to_bounding_box(
                image, roi[1], roi[0], roi[3] - roi[1], roi[2] - roi[0])
        
        if fp16: # AMP 사용시
            image = tf.image.convert_image_dtype(image, tf.float16)  # 대신 255 나누기 제거

        ### TODO: augmentation
        # def augment(image, mask):
        #     image = tf.image.random_flip_left_right(image)
        #     # 추가 증강 작업...
        #     return image, mask

        # dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

        
        ### TODO: 클래스 밸런싱 필요시
        # class_weights = ...  # 클래스별 가중치 계산
        # dataset = dataset.map(lambda x,y: (x, y, tf.gather(class_weights, y)),
        #                     num_parallel_calls=tf.data.AUTOTUNE)
        
        mask = tf.image.decode_png(parsed['mask'], channels=1)
        mask = tf.image.resize(mask, [height, width])
        if roi:
            mask = tf.image.crop_to_bounding_box(
                mask, roi[1], roi[0], roi[3] - roi[1], roi[2] - roi[0])
        mask = tf.squeeze(mask, axis=-1)  # (H,W,1) → (H,W)
        mask = tf.cast(mask, tf.int32)

        if one_hot_encoding:
            mask = tf.one_hot(mask, depth=4, axis=-1)
        
        return image, mask, parsed['filename']

    # 3. 파이프라인 최적화
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()  # 캐싱 위치 변경(셔플 전)
    # dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, 
                            num_parallel_calls=tf.data.AUTOTUNE, 
                            drop_remainder=True)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE).with_options(options)


