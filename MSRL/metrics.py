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
    
# 해밍 점수
class HammingScore(tf.keras.metrics.Metric):
    """
    다중 레이블 y_true와 y_pred를 입력받아 해밍점수를 계산함.

    올바르게 분류된 레이블의 비율을 구하는 방식.
    """
    def __init__(self, name='HammingScore', **kwargs):
        super(HammingScore, self).__init__(name, **kwargs)
        # 레이블이 올바르게 분류된 비율 합산(total), 올바르게 분류된 레이블 수(count)
        self.total = self.add_weight("total", initializer='zeros')
        self.count = self.add_weight("count", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        # y_pred 예측값 반올림
        y_pred = tf.round(y_pred)
        # 올바르게 분류된 비율 계산
        metric = tf.equal(y_true, y_pred)
        metric = tf.cast(metric, dtype=tf.float32)
        # 확인 결과 저장(누적)
        self.count.assign_add(tf.reduce_sum(metric))
        self.total.assign_add(tf.cast(tf.size(metric), dtype=tf.float32))
    
    def result(self):
        return self.count / self.total
    
    def reset_state(self):
        # 누적 변수 초기화
        self.total.assign(0)
        self.count.assign(0)
        
# 정밀도
class Precision(tf.keras.metrics.Metric):
    """
    다중 레이블 y_true와 y_pred를 입력받아 정밀도를 계산함.

    레이블중 하나라도 틀리면 해당 레이블 전체가 틀리것으로 간주하여 채점함.
    """
    pass

# 재현율
class Recall(tf.keras.metrics.Metric):
    """
    다중 레이블 y_true와 y_pred를 입력받아 재현율을 계산함.

    레이블중 하나라도 틀리면 해당 레이블 전체가 틀리것으로 간주하여 채점함.
    """
    pass