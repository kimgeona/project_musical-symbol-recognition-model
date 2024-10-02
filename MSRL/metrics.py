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

# 바운딩 박스 일치 정도
class IoU(tf.keras.metrics.Metric):
    """
    다중 레이블 y_true와 y_pred를 입력받아 IoU를 계산함.
    바운딩 박스 좌표가 애초에 없는(0으로 지정되어있는) y_true는 점수 합산에 들어가지 않음.
    """
    def __init__(self, name='IoU', **kwargs):
        super(IoU, self).__init__(name=name, **kwargs)
        # 전체 샘플 수(total), 올바르게 분류된 레이블 수(count)
        self.total = self.add_weight("total", initializer='zeros')
        self.count = self.add_weight("count", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
    
        # 바운딩 박스 좌표만 슬라이싱
        bounding_true = y_true[:, :, 0:4]
        bounding_pred = y_pred[:, :, 0:4]

        # 애초에 크기가 없는 박스 필터링
        bounding_true = tf.boolean_mask(bounding_true, self.box_area(bounding_true) > 0)
        bounding_pred = tf.boolean_mask(bounding_pred, self.box_area(bounding_true) > 0) # bounding_true를 기준으로 필터링

        # 텐서가 비어있는지 확인
        if tf.size(bounding_true)==0:
            return

        # IoU값 계산
        iou_nums = tf.shape(bounding_true)[0]
        iou = self.compute_IoU(bounding_true, bounding_pred)

        # 확인 결과 저장(누적)
        self.total.assign_add(tf.cast(tf.reduce_sum(iou), dtype=tf.float32))
        self.count.assign_add(tf.cast(iou_nums, dtype=tf.float32))
    
    def result(self):
        # 평균 IoU 정확도 계산
        return self.count / self.total
    
    def reset_state(self):
        # 누적 변수 초기화
        self.total.assign(0)
        self.count.assign(0)

    def get_config(self):
        base_config = super(IoU, self).get_config()
        return {**base_config}
    
    # 박스 넓이 계산
    def box_area(self, box):
        # 각각의 좌표로 나누기
        x1 = box[:, :, 0:1]
        y1 = box[:, :, 1:2]
        x2 = box[:, :, 2:3]
        y2 = box[:, :, 3:4]
        # 넓이 계산
        return tf.abs(tf.maximum(0.0, x2-x1) * tf.maximum(0.0, y2-y1))

    # IoU 계산
    def compute_IoU(self, box1, box2):
        # box 넓이 계싼
        box1_area = self.box_area(box1)
        box2_area = self.box_area(box2)

        # 교집합 영역의 박스 좌표 구함
        box3 = tf.concat([
            tf.maximum(box1[:, :, 0:1], box2[:, :, 0:1]),
            tf.maximum(box1[:, :, 1:2], box2[:, :, 1:2]),
            tf.minimum(box1[:, :, 2:3], box2[:, :, 2:3]),
            tf.minimum(box1[:, :, 3:4], box2[:, :, 3:4])
        ], axis=-1)
        
        # 교집합 영역의 넓이 계산
        intersection_area = tf.where(box2_area==0.0, 0.0, self.box_area(box3))

        # 합집합 영역 계산
        union_area = box1_area + box2_area - intersection_area

        # 0~1 사이값 리턴
        return intersection_area / union_area

# 중심 좌표 떨어진 정도
class PointDistance(tf.keras.metrics.Metric):
    """
    다중 레이블 y_true와 y_pred를 입력받아 중심좌표 예측이 얼마자 정답으로부터 멀어졌는지 계산함.
    """

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