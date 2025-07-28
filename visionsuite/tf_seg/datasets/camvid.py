import os 
import tensorflow as tf 
from tqdm import tqdm


def create_tfrecord_auto_shard(data_root, output_dir, split='train', max_shard_size_mb=200):
    SHARD_MAX_BYTES = max_shard_size_mb * 1024 * 1024  # 200MB
    
    # 데이터 경로 수집
    img_dir = os.path.join(data_root, split, 'images')
    img_paths = tf.io.gfile.glob(os.path.join(img_dir, '*.png'))
    
    # 샤드 초기화
    shard_idx = 0
    current_shard_size = 0
    writer = None
    
    for idx, img_path in tqdm(enumerate(img_paths)):
        # 이미지/마스크 로드
        image = tf.io.read_file(img_path)
        mask_path = img_path.replace('/images/', '/masks/')
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
            output_path = os.path.join(output_dir, f'camvid_{split}-{shard_idx:05d}.tfrecord')
            writer = tf.io.TFRecordWriter(output_path)
            shard_idx += 1
            current_shard_size = 0
            
        # 기록
        writer.write(example_bytes)
        current_shard_size += len(example_bytes)
    
    if writer: writer.close()


def build_optimized_dataset(tfrecord_dir, global_batch_size, strategy, split='train'):
    # 1. 병렬 파일 읽기 (num_parallel_reads=1024)
    files = tf.data.Dataset.list_files(f'{tfrecord_dir}/camvid_{split}-*.tfrecord')
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