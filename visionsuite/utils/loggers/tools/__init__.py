import tensorflow as tf
import datetime
import numpy as np

# 예시를 위한 간단한 모델 정의
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# 데이터 생성
x_train = np.random.randn(100, 32).astype(np.float32)
y_train = np.random.randint(0, 10, 100)

# 모델 및 옵티마이저 초기화
model = SimpleModel()
optimizer = tf.keras.optimizers.Adam()

# 손실 함수 정의
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# TensorBoard 로그 디렉토리 설정
log_dir = "/HDD/etc/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)

# 훈련 스텝 함수 정의
@tf.function
def train_step(x, y, step):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    with summary_writer.as_default():
        # 손실 기록
        tf.summary.scalar('loss', loss, step=step)
        
        # 가중치 및 그래디언트 기록
        for i, (weight, grad) in enumerate(zip(model.trainable_variables, gradients)):
            tf.summary.histogram(f'layer_{i+1}_weights', weight, step=step)
            tf.summary.histogram(f'layer_{i+1}_gradients', grad, step=step)
    
    return loss

def train_model(epochs, batch_size=32):
    step = 0
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(batch_size)
    
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}')
        
        for x_batch, y_batch in dataset:
            loss = train_step(x_batch, y_batch, step)
            step += 1
        
        # 활성화 값 기록
        with summary_writer.as_default():
            activations = model(tf.convert_to_tensor(x_train))
            tf.summary.histogram('activations', activations, step=epoch)
        
    summary_writer.close()

# 모델 훈련 실행
train_model(epochs=2, batch_size=32)
