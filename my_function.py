# 필요한 패키지
import tensorflow as tf             # 텐서플로
import tensorflow_addons as tfa     # 텐서플로 애드온
import pandas as pd                 # 판다스
import matplotlib.pyplot as plt     # 그래프 도구


# 데이터 추출 : 열 병합 추출
def column_extract(dataframe, keywords):
    # 빈 데이터 프레임 생성
    df = pd.DataFrame()
    # 기존 데이터 프레임 추출 시작
    for key in keywords:
        # 키워드 추출
        name_list = [s for s in dataframe.columns.to_numpy() if key in s]   
        # 추출된 키워드 열 끼리 병합
        for n in name_list:
            # 키워드 열 끼리 or 연산
            if key in df:   df[key.strip()] = df[key] | dataframe[n]
            else:           df[key.strip()] = dataframe[n]
    # 데이터 프레임 반환
    return df

# 데이터 추출 : 열 추출
def column_pick(dataframe, keywords):
    # 빈 데이터 프레임 생성
    df = pd.DataFrame()
    # 기존 데이터 프레임 추출 시작
    for key in keywords:
        # 키워드 추출
        name_list = [s for s in dataframe.columns.to_numpy() if key in s]
        # 추출된 키워드로 데이터 프레임 새로 만들기
        for n in name_list:
            df[n.strip()] = dataframe[n]
    # 데이터 프레임 반환
    return df

# 데이터셋 미리 보기
def review_dataset(dataset, lable_class, num_images):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_images):
            ax = plt.subplot(4, 8, i + 1)
            plt.imshow(images[i], cmap='gray')
            print(i, ' : ', [name for name, mask in zip(lable_class, tf.gather(labels[0], i).numpy()) if mask==1])
            plt.title(i)
            plt.axis('off')
    plt.show()

# tfds 전처리 : 데이터 잡음 추가
@tf.function
def add_noise(image, label):
    # 이미지 잡음
    #noise = tf.random.normal(shape=tf.shape(image), mean=0.5, stddev=0.2, dtype=tf.float32)     # 정규 분포
    noise = tf.random.uniform(shape=tf.shape(image), minval=-0.7, maxval=0.7, dtype=tf.float32)  # 균등 분포
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# tfds 전처리 : 데이터 회전
@tf.function
def rotate_image(image, label):
    # 이미지 회전
    angle = tf.random.uniform([], minval=-15, maxval=15, dtype=tf.float32)
    angle_rad = angle * (3.141592653589793 / 180.0)
    image = tfa.image.rotate(image, angle_rad)
    return image, label

# tfds 전처리 : 데이터 확대 및 축소
def scale_image(image, label):
    # 이미지 확대 및 축소
    pass

# tfds 전처리 : 데이터 이동
def shift_image(image, label):
    # 이미지 상하좌우로 이동
    pass

# tfds 전처리 : 데이터 진동
def shake_image(image, label):
    # 이미지 흔들림
    pass

# tfds 전처리 : 데이터 필터링
@tf.function
def filter_dataset(image, label):
    # label이 모두 0인 경우 필터링
    return tf.math.reduce_any(label != 0)

