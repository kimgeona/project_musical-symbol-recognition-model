# 필요한 패키지
import tensorflow as tf             # 텐서플로


# 사용자 손실 함수
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, numbers, alpha=1.0, **kwargs):
        # 전체 합
        total = sum(numbers)
        # 자신의 클래스가 차지하는 비중(비율) 구하기
        self.rate = [n / total for n in numbers]
        # 범위 제한
        epsilon = tf.keras.backend.epsilon()    
        self.rate = tf.clip_by_value(self.rate, epsilon, 1.0-epsilon)
        # 비율이 작은것이 큰 가중치를 가지도록 로그함수 통과
        self.rate = 1.0 - tf.math.log(self.rate)
        #
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        # y_pre 범위 제한(epsilon ~  (1.0 - epsilon))
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0-epsilon)
        # Binary Crossentropy 손실 계산
        loss = -(y_true * tf.math.log(y_pred) + (1-y_true) * tf.math.log(1-y_pred))
        # 배치 크기 계산
        batch_size = tf.shape(y_pred)[0]
        # 계산된 가중치 브로드 개스팅
        weights = tf.broadcast_to(self.rate, [batch_size, tf.shape(self.rate)[0]])
        # 가중치를 사용하여 손실을 계산함
        weighted_losses = tf.reduce_sum(weights * loss, axis=-1)
        weighted_losses = tf.reduce_mean(weighted_losses)
        return weighted_losses
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "rate":self.rate.numpy().tolist()}