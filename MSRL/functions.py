# 필요한 패키지
import tensorflow as tf             # 텐서플로
import tensorflow_addons as tfa     # 텐서플로 애드온

# 이미지 불러오기
@tf.function
def load_image(image, label):
    image = tf.io.read_file(image)                  # 파일 로드
    image = tf.image.decode_png(image, channels=1)  # png 파일로 변환
    image = tf.cast(image, tf.float32)              # uint8 -> float32
    image = image / 255.0                           # 0~1 로 정규화
    image = 1.0 - image                             # 흑백 이미지 반전
    return image, label

# 이미지 잡음 추가
@tf.function
def add_noise(image, label):
    #noise = tf.random.normal(shape=tf.shape(image), mean=0.5, stddev=0.2, dtype=tf.float32)     # 정규 분포
    noise = tf.random.uniform(shape=tf.shape(image), minval=-0.7, maxval=0.7, dtype=tf.float32)  # 균등 분포
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# 이미지 회전
@tf.function
def rotate_image(image, label):
    # 이미지 회전
    angle = tf.random.uniform([], minval=-15, maxval=15, dtype=tf.float32)
    angle_rad = angle * (3.141592653589793 / 180.0)
    image = tfa.image.rotate(image, angle_rad)
    return image, label

# 이미지 확대 및 축소
@tf.function
def scale_image(image, label):
    return image, label

# 이미지 이동
@tf.function
def shift_image(image, label):
    return image, label

# 이미지 진동
@tf.function
def shake_image(image, label):
    return image, label
