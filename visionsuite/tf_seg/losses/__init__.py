import tensorflow as tf 

def categorical_crossentropy(y_true, y_pred, from_logits=False, epsilon=1e-7):
    # 소프트맥스 확률값이 아닌 로짓이 들어오면 softmax 적용
    if from_logits:
        y_pred = tf.nn.softmax(y_pred)
    # 작은 값 더해서 log(0) 방지
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    # 크로스 엔트로피 계산
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return loss


if __name__ == '__main__':
    y_true = tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.05, 0.95, 0.0], [0.1, 0.8, 0.1]], dtype=tf.float32)
    loss = categorical_crossentropy(y_true, y_pred)
    print(loss.numpy())  # [0.05129329 2.3025851]
