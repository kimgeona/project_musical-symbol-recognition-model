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

# 이미지 회전
@tf.function
def rotate_image(image, label):
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
    # 이미지 크기 계산
    width = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    height = tf.cast(tf.shape(image)[1], dtype=tf.float32)

    # 이미지 이동할 크기 계산
    shift_x = tf.random.uniform([], tf.cast(-0.07 * width, dtype=tf.int32), tf.cast(0.07 * width, dtype=tf.int32), dtype=tf.int32)
    shift_y = tf.random.uniform([], tf.cast(-0.07 * height, dtype=tf.int32), tf.cast(0.07 * height, dtype=tf.int32), dtype=tf.int32)

    # 이미지 이동
    mask_zero = tf.zeros_like(image)
    shifted_image = tf.roll(image, shift=[shift_y, shift_x], axis=[0, 1])

    # 이동하고 빈자리 0으로 채우기
    if shift_y > 0:
        shifted_image = tf.concat([mask_zero[:shift_y, :, :], shifted_image[shift_y:, :, :]], axis=0)
    if shift_y < 0:
        shifted_image = tf.concat([shifted_image[:shift_y, :, :], mask_zero[shift_y:, :, :]], axis=0)

    if shift_x > 0:
        shifted_image = tf.concat([mask_zero[:, :shift_x, :], shifted_image[:, shift_x:, :]], axis=1)
    if shift_x < 0:
        shifted_image = tf.concat([shifted_image[:, :shift_x, :], mask_zero[:, shift_x:, :]], axis=1)
    
    # 이미지를 이동
    image = shifted_image

    # 데이터 형태 변경
    label = tf.reshape(label, shape=(-1, 7))

    # 데이터 분리, 레이블 데이터 좌표 수정
    shift_x = tf.cast(shift_x, tf.int16)
    shift_y = tf.cast(shift_y, tf.int16)
    x1 = label[:, 0:1] + shift_x
    y1 = label[:, 1:2] + shift_y
    x2 = label[:, 2:3] + shift_x
    y2 = label[:, 3:4] + shift_y
    rx = label[:, 4:5] + shift_x
    ry = label[:, 5:6] + shift_y
    p  = label[:, 6:7]

    # 하나로 합치기
    label = tf.concat([x1, y1, x2, y2, rx, ry, p], axis=-1)

    # 원래 형태로 변경
    label = tf.reshape(label, [-1])

    return image, label

# 이미지 잡음 추가
@tf.function
def add_noise(image, label):
    #noise = tf.random.normal(shape=tf.shape(image), mean=0.5, stddev=0.2, dtype=tf.float32)     # 정규 분포
    noise = tf.random.uniform(shape=tf.shape(image), minval=-0.7, maxval=0.7, dtype=tf.float32)  # 균등 분포
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# 이미지 진동
@tf.function
def shake_image(image, label):
    return image, label

# 이미지 자르기
def cut_image(image, label):
    
    return image, label

# 좌표 클리핑
@tf.function
def coords_clipping(image, label):
    # 이미지 크기 조사
    width = tf.cast(tf.shape(image)[-2], dtype=tf.int16)
    height = tf.cast(tf.shape(image)[-3], dtype=tf.int16)

    # 이미지 중심점 계산
    img_cx = width / 2
    img_cy = height / 2

    # shape 변경
    label = tf.reshape(label, shape=(-1, 7))

    # 클리핑할 좌표들만 추출
    x1 = tf.clip_by_value(label[:, 0:1], 0, width - 1)
    y1 = tf.clip_by_value(label[:, 1:2], 0, height - 1)
    x2 = tf.clip_by_value(label[:, 2:3], 0, width - 1)
    y2 = tf.clip_by_value(label[:, 3:4], 0, height - 1)
    rx = label[:, 4:5]
    ry = label[:, 5:6]
    p  = label[:, 6:7]

    # 하나로 합치기
    label = tf.concat([x1, y1, x2, y2, rx, ry, p], axis=-1)
    label_x = tf.concat([x1, x2, rx], axis=-1)
    label_y = tf.concat([y1, y2, ry], axis=-1)

    # rx 또는 ry가 이미지 범위를 벗어나는지 조사
    mask1 = tf.reduce_any(label[:, 4:6] < 5, axis=-1, keepdims=True)        # rx, ry < 5
    mask2 = tf.reduce_any(label[:, 4:5] > width-6, axis=-1, keepdims=True)  # rx     > width-6
    mask3 = tf.reduce_any(label[:, 5:6] > height-6, axis=-1, keepdims=True) # ry     > height-6

    # 이미지 범위를 벗어나는 이미지들 바운딩 박스 값 초기화
    label_x = tf.where(mask1, tf.cast(img_cx, dtype=tf.int16), label_x)
    label_x = tf.where(mask2, tf.cast(img_cx, dtype=tf.int16), label_x)
    label_x = tf.where(mask3, tf.cast(img_cx, dtype=tf.int16), label_x)
    label_y = tf.where(mask1, tf.cast(img_cy, dtype=tf.int16), label_y)
    label_y = tf.where(mask2, tf.cast(img_cy, dtype=tf.int16), label_y)
    label_y = tf.where(mask3, tf.cast(img_cy, dtype=tf.int16), label_y)
    p       = tf.where(mask1, tf.constant(0, dtype=tf.int16),  p)
    p       = tf.where(mask2, tf.constant(0, dtype=tf.int16),  p)
    p       = tf.where(mask3, tf.constant(0, dtype=tf.int16),  p)

    # 하나로 합치기
    label = tf.concat([
        label_x[:, 0:1],
        label_y[:, 0:1],
        label_x[:, 1:2],
        label_y[:, 1:2],
        label_x[:, 2:3],
        label_y[:, 2:3],
        p
    ], axis=-1)

    # 원래 형태로 변경
    label = tf.reshape(label, [-1])

    return image, label

# 좌표 변환
@tf.function
def coords_convert(image, label):
    # shape 변경
    label = tf.reshape(label, shape=(-1, 7))

    # 데이터 분리, 자료형 변경
    x1 = tf.cast(label[:, 0:1], dtype=tf.float32)
    y1 = tf.cast(label[:, 1:2], dtype=tf.float32)
    x2 = tf.cast(label[:, 2:3], dtype=tf.float32)
    y2 = tf.cast(label[:, 3:4], dtype=tf.float32)
    rx = tf.cast(label[:, 4:5], dtype=tf.float32)
    ry = tf.cast(label[:, 5:6], dtype=tf.float32)
    p  = tf.cast(label[:, 6:7], dtype=tf.float32)

    # x1, y1, x2, y2, rx, ry, prob 를,
    # cx, cy,  w,  h, rx, ry, prob 로 변환
    cx = (x2 + x1) / 2.0
    cy = (y2 + y1) / 2.0
    w  = tf.abs(x2 - x1)
    h  = tf.abs(y2 - y1)

    # prob이 0이면 전부 0으로 변경
    w = tf.where(p==0, 0.0, w)
    h = tf.where(p==0, 0.0, h)

    # 하나로 합치기
    label = tf.concat([cx, cy, w, h, rx, ry, p], axis=-1)

    # 원래 형태로 변경
    label = tf.reshape(label, [-1])

    return image, label

# 좌표 스케일링
@tf.function
def coords_scaling(image, label):
    # 이미지 크기 조사
    width = tf.cast(tf.shape(image)[-2], dtype=tf.float32)
    height = tf.cast(tf.shape(image)[-3], dtype=tf.float32)

    # shape 변경
    label = tf.reshape(label, shape=(-1, 7))

    # 데이터 분리, 0~1 스케일링
    cx = label[:, 0:1] / width
    cy = label[:, 1:2] / height
    w  = label[:, 2:3] / width
    h  = label[:, 3:4] / height
    rx = label[:, 4:5] / width
    ry = label[:, 5:6] / height
    p  = label[:, 6:7]

    # 하나로 합치기
    label = tf.concat([cx, cy, w, h, rx, ry, p], axis=-1)

    # 원래 형태로 변경
    label = tf.reshape(label, [-1])

    return image, label
