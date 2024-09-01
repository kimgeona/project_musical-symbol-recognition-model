# 필요한 패키지
import tensorflow as tf             # 텐서플로


# 사용자 손실 함수
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """
    소수 레이블에 더 많은 가중치를 부과하는 손실 함수.
    """
    def __init__(self, total_count, class_count, name='WeightedBinaryCrossentropy', **kwargs):
        super(WeightedBinaryCrossentropy, self).__init__(name=name, **kwargs)
        # 가중치 계산
        self.weight_true = [n / total_count for n in class_count]
        self.weight_false = [1.0 - n for n in self.weight_true]
        # 가중치 범위 제한 (0+epsilon <= w <= 1)
        epsilon = tf.keras.backend.epsilon()
        self.weight_true = tf.clip_by_value(self.weight_true, epsilon, 1.0-epsilon)
        self.weight_false = tf.clip_by_value(self.weight_false, epsilon, 1.0-epsilon)
        # 가중치 로그함수 처리
        self.weight_true = 1.0 - tf.math.log(self.weight_true)
        self.weight_false = 1.0 - tf.math.log(self.weight_false)


    def call(self, y_true, y_pred):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        # y_pre 범위 제한 (0+epsilon <= y_pred <= 1-epsilon)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0-epsilon)
        # Binary Crossentropy loss 계산
        loss = -(y_true * tf.math.log(y_pred) + (1-y_true) * tf.math.log(1-y_pred))
        # loss * weight
        weighted_losses = tf.where(y_true==1.0, loss * self.weight_true, loss * self.weight_false)
        weighted_losses = tf.reduce_sum(weighted_losses, axis=-1)
        weighted_losses = tf.reduce_mean(weighted_losses)
        return weighted_losses
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "weight_true":self.weight_true.numpy().tolist(), "weight_false":self.weight_false.numpy().tolist()}