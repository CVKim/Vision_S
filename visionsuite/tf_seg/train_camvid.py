import tensorflow as tf
import os.path as osp
import datetime
import os

if __name__ == "__main__":
    from loops.epoch_based_loop import train_loop
    from datasets.camvid import create_tfrecord_auto_shard, build_optimized_dataset
    # from datasets.ade20k import create_tfrecord_auto_shard, build_optimized_dataset
    from losses import categorical_crossentropy
    from models.deeplabv3plus import DeepLabV3plus
    from models.backbones import create_base_model


    # input_dir = '/HDD/datasets/public/ade20k_2016/ADEChallengeData2016'
    input_dir = '/HDD/datasets/public/camvid'
    output_dir = '/HDD/etc/tensorflow'
    now = datetime.datetime.now()

    tfrecords_dir = osp.join(output_dir, 'tfrecords')
    if not osp.exists(tfrecords_dir):
        os.mkdir(tfrecords_dir)
        
        create_tfrecord_auto_shard(
            data_root=input_dir,
            output_dir=tfrecords_dir,
            split='train'
        )
    
    output_dir = osp.join(output_dir, f'{now.year}{now.month}{now.day}-{now.hour}{now.minute}{now.second}')
    logs_dir = osp.join(output_dir, 'logs')
    
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    if not osp.exists(logs_dir):
        os.mkdir(logs_dir)

    epochs = 10
    global_batch_size = 4
    num_classes = 150

    strategy = tf.distribute.MirroredStrategy()
    print(f'활성화된 GPU 수: {strategy.num_replicas_in_sync}')


    with strategy.scope():
        base_model, layers, layer_names = create_base_model('efficientnetb0', 'imagenet', 512, 512)
        model = DeepLabV3plus(150, base_model, layers, 512, 512)
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-4, decay_steps=1000, decay_rate=0.96
        )
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                    reduction=tf.keras.losses.Reduction.NONE # 필수 변경[2][4]
                )
        # loss_fn = tf.keras.losses.CategoricalCrossentropy(
        #             reduction=tf.keras.losses.Reduction.NONE # 필수 변경[2][4]
        #         )
        # loss_fn = categorical_crossentropy
        # loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
        #                 alpha=0.25,  # 클래스 불균형 조정 파라미터[1]
        #                 gamma=2.0,   # 어려운 샘플 가중치 강조[1]
        #                 from_logits=False,
        #                 reduction=tf.keras.losses.Reduction.NONE
        #             )
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    train_dataset = build_optimized_dataset(tfrecords_dir, global_batch_size, strategy)
    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    
    # sample_batch = next(iter(dist_dataset))
    # per_replica_images = strategy.experimental_local_results(sample_batch[0])
    # print("replica 0 image shape:", per_replica_images[0].shape)  # (bs_per_replica, 512,512,3)
    # per_replica_masks = strategy.experimental_local_results(sample_batch[1])
    # print("replica 0 mask 범위:", per_replica_masks[0].shape)
    
    
    train_loop(model, dist_dataset, epochs, optimizer, loss_fn, strategy, train_acc_metric)
