import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras
from tensorflow.keras import Sequential, layers, losses

# 加载数据集
data_path = r'../data/caltech256/256_ObjectCategories'  # 图像数据路径
H, W = 299, 299  # 图像缩放后的高和宽
batch_size = 256

# 训练集
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=23,
    image_size=(H, W),
    batch_size=batch_size
)
class_names = train_ds.class_names  # 获取图像的名称
train_ds = train_ds.map(lambda x, y: (x/255, y))  # 将图像归一化处理

# 测试集
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=23,
    image_size=(H, W),
    batch_size=batch_size
)
test_ds = test_ds.map(lambda x, y: (x/255, y))  # 将图像归一化处理
print('测试集和训练集batch个数：', len(train_ds), len(test_ds))  # 766, 192
# 测速集
tot_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    seed=23,
    image_size=(H, W),
    batch_size=32
)
tot_ds = tot_ds.map(lambda x, y: (x/255, y))  # 将图像归一化处理

# 模型导入
model = Sequential([
    hub.KerasLayer("saved_model/inception_resnet_v2/", input_shape=(H, W, 3)),
    layers.Dense(1000, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(257, activation='softmax')
])
model.summary()  # 查看模型参数

# 保存模型权重信息及tensorboard
checkpoint_path = "checkpoint/inception_resnet_v2.ckpt"
# model.load_weights(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True
)
tb_callback = keras.callbacks.TensorBoard(log_dir="logs/inception_resnet_v2", histogram_freq=1)
# tb_callback = keras.callbacks.TensorBoard(log_dir="logs/tmp", histogram_freq=1)

model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

model.fit(train_ds, validation_data=test_ds, epochs=100, callbacks=[cp_callback, tb_callback])
# model.evaluate(tot_ds)
