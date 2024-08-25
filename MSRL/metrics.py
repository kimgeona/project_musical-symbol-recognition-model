# 필요한 패키지
import tensorflow as tf             # 텐서플로


# 정확도
class Accuracy(tf.keras.metrics.Metric):
    """
    다중 레이블 y_true와 y_pred를 입력받아 정확도를 계산함.

    레이블중 하나라도 틀리면 해당 레이블 전체가 틀리것으로 간주하여 채점함.
    """
    def __init__(self, name='Accuracy', **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        # 전체 샘플 수(total), 올바르게 분류된 레이블 수(count)
        self.total = self.add_weight("total", initializer='zeros')
        self.count = self.add_weight("count", initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        # y_pred 예측값 제한
        y_pred = tf.round(y_pred)
        # y_true == y_pred 확인
        metric = tf.equal(y_true, y_pred)
        metric = tf.reduce_all(metric, axis=-1)
        metric = tf.cast(metric, dtype=tf.float32)
        # 확인 결과 저장(누적)
        self.total.assign_add(tf.cast(tf.size(metric), dtype=tf.float32))
        self.count.assign_add(tf.reduce_sum(metric))
        
    def result(self):
        # 정확도 계산
        return self.count / self.total
    
    def reset_state(self):
        # 누적 변수 초기화
        self.total.assign(0)
        self.count.assign(0)
        
    def get_config(self):
        base_config = super(Accuracy, self).get_config()
        return {**base_config}
    
# 정확도
class MeanAccuracy(tf.keras.metrics.Metric):
    """
    다중 레이블 y_true와 y_pred를 입력받아 정확도를 계산함.

    각각의 레이블 인덱스 별로 정확도를 구하고 평균적인 정확도를 구해줌.
    """
    def __init__(self, num_classes, name='MeanAccuracy', **kwargs):
        super(MeanAccuracy, self).__init__(name=name, **kwargs)
        # 레이블 클래스 갯수
        self.num_classes = num_classes
        # 전체 샘플 수(total), 올바르게 분류된 레이블 수(count)
        self.total = self.add_weight("total", initializer='zeros')
        self.count = self.add_weight("count", shape=(num_classes,), initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        # y_pred 예측값 제한
        y_pred = tf.round(y_pred)
        # y_true == y_pred 확인
        metric = tf.equal(y_true, y_pred)
        metric = tf.cast(metric, dtype=tf.float32)
        # 확인 결과 저장(누적)
        self.total.assign_add(tf.cast(tf.shape(metric)[0], dtype=tf.float32))
        self.count.assign_add(tf.reduce_sum(metric, axis=-2))
        
    def result(self):
        # 정확도 계산
        return tf.reduce_mean(self.count / self.total)
    
    def reset_state(self):
        # 누적 변수 초기화
        self.total.assign(0)
        self.count.assign(tf.zeros(shape=(self.num_classes,), dtype=tf.float32))
        
    def get_config(self):
        base_config = super(MeanAccuracy, self).get_config()
        return {**base_config, 'num_classes':int(self.num_classes)}