import tensorflow as tf
import os.path as osp
import datetime
import os

if __name__ == "__main__":
    from loops.epoch_based_loop import train_loop
    from datasets.labelme import labelme2tfrecord_auto_shard, build_optimized_dataset
    from losses import categorical_crossentropy
    from models.deeplabv3plus import DeepLabV3plus
    from models.backbones import create_base_model

    input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_dataset_unit'
    output_dir = '/HDD/etc/tensorflow'
    now = datetime.datetime.now()

    tfrecord_dir = osp.join(output_dir, 'tfrecords_unit')

    if not osp.exists(tfrecord_dir):
        os.mkdir(tfrecord_dir)
            
        labelme2tfrecord_auto_shard(
            data_root=input_dir,
            output_dir=tfrecord_dir,
            split='train',
            class_names = ['MARK', 'CHAMFER_MARK', 'LINE']
        )

    output_dir = osp.join(output_dir, f'{now.year}{now.month}{now.day}-{now.hour}{now.minute}{now.second}')
    logs_dir = osp.join(output_dir, 'logs')
    
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    if not osp.exists(logs_dir):
        os.mkdir(logs_dir)

    epochs = 3
    batch_size = 16
    num_classes = 4
    roi = [220, 60, 1340, 828]
    strategy = tf.distribute.MirroredStrategy()
    print(f'활성화된 GPU 수: {strategy.num_replicas_in_sync}')
    '''
        >>>>> time for epoch:  70.28238725662231
        0/3 > 9: loss(1372844.5000), gpu(19.96GB)>>>>> time for epoch:  5.5843212604522705
        0/3 > 19: loss(1352080.5000), gpu(21.16GB)>>>>> time for epoch:  5.551506280899048
        0/3 > 28: loss(1330189.0000), gpu(14.28GB)
        Epoch 1, Loss: 1363394.6250, Accuracy: 0.2831
        EPOCH is done:  85.90839838981628
        >>>>> time for epoch:  1.0797467231750488
        1/3 > 9: loss(1311259.7500), gpu(21.16GB)>>>>> time for epoch:  5.594640493392944
        1/3 > 19: loss(1295287.0000), gpu(20.05GB)>>>>> time for epoch:  5.598327875137329
        1/3 > 28: loss(1272684.6250), gpu(12.86GB)
        Epoch 2, Loss: 1303733.2500, Accuracy: 0.2875
        EPOCH is done:  16.802703380584717
        >>>>> time for epoch:  1.270380973815918
        2/3 > 9: loss(1262342.5000), gpu(19.76GB)>>>>> time for epoch:  5.563065767288208
        2/3 > 19: loss(1256613.2500), gpu(22.04GB)>>>>> time for epoch:  5.536720275878906
        2/3 > 28: loss(1241454.2500), gpu(14.26GB)
        Epoch 3, Loss: 1260447.7500, Accuracy: 0.2827
        EPOCH is done:  16.78995704650879
    '''

    with strategy.scope():
        height, width = 768, 1120
        '''
        
            effb0
            Total params: 10,783,535
            Trainable params: 0
            Non-trainable params: 10,783,535
        
            tf - deeplabv3plus
            Total params: 2,329,209
            Trainable params: 2,139,820
            Non-trainable params: 189,389
            
            
            torch - deeplabv3plus
            Total params: 18,352,180
            Trainable params: 18,352,180
            Non-trainable params: 0
            ----------------------------------------------------------------
            Input size (MB): 9.84
            Forward/backward pass size (MB): 54425825554351.47
            Params size (MB): 70.01
            Estimated Total Size (MB): 54425825554431.32
            
            
            torch - efficientnetb0
            Total params: 5,858,704
            Trainable params: 5,858,704
            Non-trainable params: 0
            ----------------------------------------------------------------
            Input size (MB): 9.84
            Forward/backward pass size (MB): 2697.39
            Params size (MB): 22.35
            Estimated Total Size (MB): 2729.58

        '''
        base_model, layers, layer_names = create_base_model('efficientnetb3', 'imagenet', height, width)
        model = DeepLabV3plus(num_classes, base_model, layers, height=height, width=width)
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-4, decay_steps=1000, decay_rate=0.96
        )
        
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                    reduction=tf.keras.losses.Reduction.NONE # 필수 변경[2][4]
                )
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    train_dataset = build_optimized_dataset(tfrecord_dir=tfrecord_dir, 
                                            batch_size=batch_size, 
                                            cache=True, shuffle_buffer=batch_size*20,
                                            image_format='bmp',
                                            shuffle=True,
                                            one_hot_encoding=False,
                                            split='train', roi=roi, fp16=False
                                        )
    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    print("Start training >>>")
    train_loop(model, dist_dataset, epochs, optimizer, loss_fn, strategy, train_acc_metric)



    # import cv2 
    # import numpy as np
    # sample_batch = next(iter(dist_dataset))
    # per_replica_images = strategy.experimental_local_results(sample_batch[0])
    # print("replica 0 image shape:", per_replica_images[0].shape) 
    # cv2.imwrite("/HDD/etc/image0.png", per_replica_images[0][0].numpy())
    # per_replica_masks = strategy.experimental_local_results(sample_batch[1])
    # print("replica 0 mask shape:", per_replica_masks[0].shape)
    # print("replica 0 mask unique:", np.unique(per_replica_masks[0][0].numpy()))
    # cv2.imwrite("/HDD/etc/mask0.png", per_replica_masks[0][0].numpy()*60)
    # print("replica 0 mask unique:", np.unique(per_replica_masks[0][0].numpy()))
    # per_replica_filename = strategy.experimental_local_results(sample_batch[2])
    # print("replica 0 filename:", per_replica_filename[0][0])


    # per_replica_images = strategy.experimental_local_results(sample_batch[0])
    # print("replica 1 image shape:", per_replica_images[0].shape) 
    # cv2.imwrite("/HDD/etc/image1.png", per_replica_images[0][1].numpy())
    # per_replica_masks = strategy.experimental_local_results(sample_batch[1])
    # print("replica 1 mask shape:", per_replica_masks[0].shape)
    # print("replica 1 mask unique:", np.unique(per_replica_masks[0][1].numpy()))
    # cv2.imwrite("/HDD/etc/mask1.png", per_replica_masks[0][1].numpy()*60)
    # print("replica 1 mask unique:", np.unique(per_replica_masks[0][1].numpy()))
    # per_replica_filename = strategy.experimental_local_results(sample_batch[2])
    # print("replica 1 filename:", per_replica_filename[0][1])

    
