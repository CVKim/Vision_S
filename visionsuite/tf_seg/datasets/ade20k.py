import os 
import tensorflow as tf 
from tqdm import tqdm


def create_ade20k_pipeline(data_root, batch_size, strategy, split='training', img_size=(512,512)):
    img_dir = os.path.join(data_root, 'images', split)
    ann_dir = os.path.join(data_root, 'annotations', split)
    img_paths = tf.data.Dataset.list_files(os.path.join(img_dir, '*.jpg'), shuffle=True)
    # mask_paths = img_paths.map(
    # lambda x: tf.strings.regex_replace(
    # x, 'images', 'annotations').replace('.jpg', '.png'))
    mask_paths = img_paths.map(
    lambda x: tf.strings.regex_replace(
    tf.strings.regex_replace(x, 'images', 'annotations'),
    '.jpg', '.png'
    )
    )

    dataset = tf.data.Dataset.zip((img_paths, mask_paths))
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    def load_process(img_path, mask_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size) / 255.0

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, img_size, method='nearest')

        return image, mask

    dataset = dataset.map(load_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size * strategy.num_replicas_in_sync)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_ade20k_pipeline_interleave(data_root, batch_size, strategy, split='training', img_size=(512,512)):
    img_dir = os.path.join(data_root, 'images', split)
    
    # 1. 이미지 경로 데이터셋 생성
    img_paths = tf.data.Dataset.list_files(os.path.join(img_dir, '*.jpg'), shuffle=True)
    
    # 2. Interleave로 병렬 처리
    dataset = img_paths.interleave(
        lambda img_path: tf.data.Dataset.from_tensors({
            'image': img_path,
            'mask': tf.strings.regex_replace(
                tf.strings.regex_replace(img_path, 'images', 'annotations'),
                '.jpg', '.png'
            )
        }),
        cycle_length=64,  # 병렬 처리할 파일 수[1][3]
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # 3. 병렬 로드 및 전처리
    def load_process(files):
        image = tf.io.read_file(files['image'])
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size) / 255.0
        
        mask = tf.io.read_file(files['mask'])
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, img_size, method='nearest')
        return image, mask
    
    dataset = dataset.map(load_process, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. 성능 최적화
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size * strategy.num_replicas_in_sync)
    return dataset.prefetch(tf.data.AUTOTUNE)

def create_tfrecord(data_root, output_dir, split='training', img_size=(512,512)):
    img_dir = os.path.join(data_root, 'images', split)
    ann_dir = os.path.join(data_root, 'annotations', split)
    
    # 이미지-마스크 경로 매칭
    img_paths = tf.io.gfile.glob(os.path.join(img_dir, '*.jpg'))
    mask_paths = [p.replace('/images/', '/annotations/').replace('.jpg', '.png') for p in img_paths]

    # TFRecord 작성
    output_path = os.path.join(output_dir, f'ade20k_{split}.tfrecord')
    with tf.io.TFRecordWriter(output_path) as writer:
        for img_path, mask_path in zip(img_paths, mask_paths):
            image = tf.io.read_file(img_path)
            mask = tf.io.read_file(mask_path)
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()])),
                'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.numpy()]))
            }))
            writer.write(example.SerializeToString())

def create_tfrecord_auto_shard(data_root, output_dir, split='training', max_shard_size_mb=200):
    SHARD_MAX_BYTES = max_shard_size_mb * 1024 * 1024  # 200MB
    
    # 데이터 경로 수집
    img_dir = os.path.join(data_root, 'images', split)
    img_paths = tf.io.gfile.glob(os.path.join(img_dir, '*.jpg'))
    
    # 샤드 초기화
    shard_idx = 0
    current_shard_size = 0
    writer = None
    
    for idx, img_path in tqdm(enumerate(img_paths)):
        # 이미지/마스크 로드
        image = tf.io.read_file(img_path)
        mask_path = img_path.replace('/images/', '/annotations/').replace('.jpg', '.png')
        mask = tf.io.read_file(mask_path)
        
        # 예제 직렬화
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()])),
            'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.numpy()]))
        }))
        example_bytes = example.SerializeToString()
        
        # 샤드 용량 체크
        if writer is None or current_shard_size + len(example_bytes) > SHARD_MAX_BYTES:
            if writer: writer.close()
            output_path = os.path.join(output_dir, f'ade20k_{split}-{shard_idx:05d}.tfrecord')
            writer = tf.io.TFRecordWriter(output_path)
            shard_idx += 1
            current_shard_size = 0
            
        # 기록
        writer.write(example_bytes)
        current_shard_size += len(example_bytes)
    
    if writer: writer.close()

def create_tfrecord_parallel(data_root, output_dir, split='training', max_shard_size_mb=200):
    SHARD_MAX_BYTES = max_shard_size_mb * 1024 * 1024  # 200MB
    
    # 데이터 경로 수집
    img_dir = os.path.join(data_root, 'images', split)
    img_paths = tf.io.gfile.glob(os.path.join(img_dir, '*.jpg'))
    
    # 샤드 초기화
    shard_idx = 0
    current_shard_size = 0
    writer = None
    
    for idx, img_path in enumerate(img_paths):
        # 이미지/마스크 로드
        image = tf.io.read_file(img_path)
        mask_path = img_path.replace('/images/', '/annotations/').replace('.jpg', '.png')
        mask = tf.io.read_file(mask_path)
        
        # 예제 직렬화
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()])),
            'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.numpy()]))
        }))
        example_bytes = example.SerializeToString()
        
        # 샤드 용량 체크
        if writer is None or current_shard_size + len(example_bytes) > SHARD_MAX_BYTES:
            if writer: writer.close()
            output_path = os.path.join(output_dir, f'ade20k_{split}-{shard_idx:05d}.tfrecord')
            writer = tf.io.TFRecordWriter(output_path)
            shard_idx += 1
            current_shard_size = 0
            
        # 기록
        writer.write(example_bytes)
        current_shard_size += len(example_bytes)
    
    if writer: writer.close()



def build_optimized_dataset(tfrecord_dir, global_batch_size, strategy, split='training'):
    # 1. 병렬 파일 읽기 (num_parallel_reads=1024)
    files = tf.data.Dataset.list_files(f'{tfrecord_dir}/ade20k_{split}-*.tfrecord')
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=1024),
        cycle_length=256,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # 2. 중복 연산 제거를 위한 캐싱
    def parse_fn(example):
        feature_desc = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string)
        }
        parsed = tf.io.parse_single_example(example, feature_desc)
        
        image = tf.image.decode_jpeg(parsed['image'], channels=3)
        image = tf.image.resize(image, [512,512]) / 255.0
        
        mask = tf.image.decode_png(parsed['mask'], channels=1)
        mask = tf.image.resize(mask, [512,512], method='nearest')
        mask = tf.squeeze(mask, -1)
        
        # masks = [(mask == v) for v in range(150)]
        # mask = tf.stack(mask, axis=-1)
        # mask = tf.one_hot(mask, depth=150, axis=-1)  # 150개 클래스 원-핫 인코딩[3]

        return image, mask
    
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()  # 파싱 결과 캐싱[4][5]
    
    # 3. 성능 최적화
    dataset = dataset.shuffle(200, reshuffle_each_iteration=True)
    dataset = dataset.batch(global_batch_size//strategy.num_replicas_in_sync)
    return dataset.prefetch(tf.data.AUTOTUNE)



if __name__ == '__main__':
    from glob import glob 
    import os.path as osp 
    
    input_dir = '/HDD/datasets/public/ade20k_2016/ADEChallengeData2016'
    
    mask_files = glob(osp.join(input_dir, 'annotations/training/*.png'))
    print(f"There are {len(mask_files)} mask files")
    
    
    for mask_file in mask_files:
        import cv2 
        import numpy as np
        mask = cv2.imread(mask_files[0], 0)
        uniques = np.unique(mask)
        
        for unique in uniques:
            if unique > 150:
                print(">>>>>>>>>>>>>>>>")
                