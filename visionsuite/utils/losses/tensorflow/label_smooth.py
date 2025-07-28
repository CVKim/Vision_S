import tensorflow as tf

class LabelSmoothSoftmaxCEV1(tf.keras.layers.Layer):
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    # def call(self, logits, label):
    #     logits = tf.cast(logits, tf.float32)
    #     num_classes = tf.shape(logits)[1]
    #     label = tf.cast(label, tf.int32)
        
    #     # Reshape logits and label
    #     logits = tf.transpose(logits, [0, 2, 3, 1])  # (8, 384, 384, 19)
    #     label = tf.expand_dims(label, axis=-1)  # (8, 384, 384, 1)
        
    #     ignore = tf.equal(label, self.lb_ignore)
    #     n_valid = tf.reduce_sum(tf.cast(tf.logical_not(ignore), tf.float32))
        
    #     label = tf.where(ignore, tf.zeros_like(label), label)
    #     lb_pos, lb_neg = 1.0 - self.lb_smooth, self.lb_smooth / tf.cast(num_classes, tf.float32)
        
    #     lb_one_hot = tf.one_hot(tf.squeeze(label, axis=-1), depth=num_classes, on_value=lb_pos, off_value=lb_neg)
        
    #     logs = tf.nn.log_softmax(logits, axis=-1)
    #     loss = -tf.reduce_sum(logs * lb_one_hot, axis=-1)
    #     loss = tf.where(tf.squeeze(ignore, axis=-1), tf.zeros_like(loss), loss)
        
    #     if self.reduction == 'mean':
    #         loss = tf.reduce_sum(loss) / n_valid
    #     elif self.reduction == 'sum':
    #         loss = tf.reduce_sum(loss)
        
    #     return loss

    def call(self, logits, label):
        logits = tf.cast(logits, tf.float32)  # use fp32 to avoid nan
        
        num_classes = tf.shape(logits)[1]
        label = tf.identity(label)
        ignore = tf.equal(label, self.lb_ignore)
        n_valid = tf.reduce_sum(tf.cast(tf.logical_not(ignore), tf.float32))
        label = tf.where(ignore, tf.zeros_like(label), label)
        lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / tf.cast(num_classes, tf.float32)

        lb_one_hot = tf.one_hot(label, depth=logits.shape[1], on_value=lb_pos, off_value=lb_neg)
        lb_one_hot = tf.reshape(lb_one_hot, tf.shape(logits))
        
        logs = tf.nn.log_softmax(logits, axis=1)
        loss = -tf.reduce_sum(logs * lb_one_hot, axis=1)
        loss = tf.where(ignore, tf.zeros_like(loss), loss)
        
        if self.reduction == 'mean':
            loss = tf.reduce_sum(loss) / n_valid
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(loss)
        
        return loss

    
if __name__ == '__main__':
    
    import numpy as np
    
    np.random.seed(0)
    input_data = np.random.randn(8, 19, 384, 384).astype(np.float32)
    labels = np.random.randint(0, 19, size=(8, 384, 384)).astype(np.int64)
    
    tf_loss = LabelSmoothSoftmaxCEV1()
    tf_inputs = tf.constant(input_data)
    tf_labels = tf.constant(labels)
    tf_result = tf_loss(tf_inputs, tf_labels).numpy()
    print("tf: ", tf_result)
    
    import torch
    from visionsuite.utils.losses.torch.label_smooth import LabelSmoothSoftmaxCEV1 as TorchLabelSmoothSoftmaxCEV1
    
    torch_loss = TorchLabelSmoothSoftmaxCEV1()
    torch_inputs = torch.from_numpy(input_data)
    torch_labels = torch.from_numpy(labels).to(torch.int64)
    torch_result = torch_loss(torch_inputs, torch_labels)
    print("torch: ", torch_result)
    
    