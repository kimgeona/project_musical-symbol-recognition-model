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
        # Nan 값 확인
        tf.debugging.check_numerics(y_true, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(y_pred, "NaN or Inf is not allowed")

        # 배치 크기 알아내기
        batch_size = tf.shape(y_true)[0]

        # shape 변경
        y_true = tf.reshape(y_true, shape=(batch_size, -1, 7))
        y_pred = tf.reshape(y_pred, shape=(batch_size, -1, 7))

        # 바운딩 박스 좌표만 슬라이싱
        # (batch_size, -1, 7) -> (batch_size, -1)
        y_true = y_true[:, :, 6:7]
        y_pred = y_pred[:, :, 6:7]

        # y_pred 예측값 반올림
        y_pred = tf.round(y_pred)

        # 자료형 int16으로 변경
        y_true = tf.cast(y_true, dtype=tf.int16)
        y_pred = tf.cast(y_pred, dtype=tf.int16)

        # y_true == y_pred 확인
        metric = tf.equal(y_true, y_pred)
        metric = tf.reduce_all(metric, axis=[-2])
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
        # Nan 값 확인
        tf.debugging.check_numerics(y_true, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(y_pred, "NaN or Inf is not allowed")

        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # 배치 크기 알아내기
        batch_size = tf.shape(y_true)[0]

        # shape 변경
        y_true = tf.reshape(y_true, shape=(batch_size, -1, 7))
        y_pred = tf.reshape(y_pred, shape=(batch_size, -1, 7))

        # 바운딩 박스 좌표만 슬라이싱
        # (batch_size, -1, 7) -> (batch_size, -1)
        y_true = y_true[:, :, 6:7]
        y_pred = y_pred[:, :, 6:7]

        # y_pred 예측값 반올림
        y_pred = tf.round(y_pred)

        # 자료형 int16으로 변경
        y_true = tf.cast(y_true, dtype=tf.int16)
        y_pred = tf.cast(y_pred, dtype=tf.int16)

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
    def __init__(self, image_height, image_width, name='IoU', **kwargs):
        super(IoU, self).__init__(name=name, **kwargs)
        # 이미지 크기 저장
        self.image_height = tf.constant(image_height, dtype=tf.float32)
        self.image_width = tf.constant(image_width, dtype=tf.float32)

        # 전체 샘플 수(total), 올바르게 분류된 레이블 수(count)
        self.total = self.add_weight("total", initializer='zeros')
        self.count = self.add_weight("count", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
    
        # 배치 크기 알아내기
        batch_size = tf.shape(y_true)[0]

        # shape 변경
        y_true = tf.reshape(y_true, shape=(batch_size, -1, 7))
        y_pred = tf.reshape(y_pred, shape=(batch_size, -1, 7))

        # 스케일 복귀
        x1 = y_true[:, :, 0:1] * self.image_width
        y1 = y_true[:, :, 1:2] * self.image_height
        x2 = y_true[:, :, 2:3] * self.image_width
        y2 = y_true[:, :, 3:4] * self.image_height
        cx = y_true[:, :, 4:5] * self.image_width
        cy = y_true[:, :, 5:6] * self.image_height
        p  = y_true[:, :, 6:7]
        y_true = tf.concat([x1, y1, x2, y2, cx, cy, p], axis=-1)
        #
        x1 = y_pred[:, :, 0:1] * self.image_width
        y1 = y_pred[:, :, 1:2] * self.image_height
        x2 = y_pred[:, :, 2:3] * self.image_width
        y2 = y_pred[:, :, 3:4] * self.image_height
        cx = y_pred[:, :, 4:5] * self.image_width
        cy = y_pred[:, :, 5:6] * self.image_height
        p  = y_pred[:, :, 6:7]
        y_pred = tf.concat([x1, y1, x2, y2, cx, cy, p], axis=-1)

        # 바운딩 박스 좌표만 슬라이싱
        bounding_true = y_true[:, :, 0:4]
        bounding_pred = y_pred[:, :, 0:4]

        # IoU값 계산
        iou_nums = tf.reduce_sum(tf.cast(self.box_area(bounding_true)!=0.0, dtype=tf.int16))
        iou = tf.where(self.box_area(bounding_true)==0.0, 0.0, self.compute_IoU(bounding_true, bounding_pred))

        # 확인 결과 저장(누적)
        self.total.assign_add(tf.cast(tf.reduce_sum(iou), dtype=tf.float32))
        self.count.assign_add(tf.cast(iou_nums, dtype=tf.float32))
    
    def result(self):
        # 평균 IoU 정확도 계산
        return tf.where(self.count==0.0, 0.0, self.total / self.count)
    
    def reset_state(self):
        # 누적 변수 초기화
        self.total.assign(0)
        self.count.assign(0)

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
        return tf.where(union_area==0.0, 0.0, intersection_area / union_area)

# 중심 좌표 떨어진 정도
class PointDistance(tf.keras.metrics.Metric):
    """
    다중 레이블 y_true와 y_pred를 입력받아 중심좌표 예측이 얼마자 정답으로부터 멀어졌는지 계산함.
    """
    def __init__(self, image_height, image_width, name='PointDistance', **kwargs):
        super(PointDistance, self).__init__(name=name, **kwargs)
        # 이미지 크기 저장
        self.image_height = tf.constant(image_height, dtype=tf.float32)
        self.image_width = tf.constant(image_width, dtype=tf.float32)

        # 전체 샘플 수(total), 올바르게 분류된 레이블 수(count)
        self.total = self.add_weight("total", initializer='zeros')
        self.count = self.add_weight("count", initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
    
        # 배치 크기 알아내기
        batch_size = tf.shape(y_true)[0]

        # shape 변경
        y_true = tf.reshape(y_true, shape=(batch_size, -1, 7))
        y_pred = tf.reshape(y_pred, shape=(batch_size, -1, 7))

        # 바운딩 박스 존재 여부 슬라이싱
        probability = y_true[:, :, 6:7]

        # 악상기호 중심 좌표 슬라이싱 및 스케일 복귀 
        cx = y_true[:, :, 4:5] * self.image_width
        cy = y_true[:, :, 5:6] * self.image_height
        y_true = tf.concat([cx, cy], axis=-1)
        #
        cx = y_pred[:, :, 4:5] * self.image_width
        cy = y_pred[:, :, 5:6] * self.image_height
        y_pred = tf.concat([cx, cy], axis=-1)

        # 갯수와 중심점 사이 거리 구하기
        distance_nums = tf.reduce_sum(probability)
        distance = self.distance(y_true, y_pred) * probability

        # 확인 결과 저장(누적)
        self.total.assign_add(tf.cast(tf.reduce_sum(distance), dtype=tf.float32))
        self.count.assign_add(tf.cast(distance_nums, dtype=tf.float32))

    def distance(self, true, pred):
        # 좌표 추출
        x1 = true[:, :, 0:1]
        y1 = true[:, :, 1:2]
        x2 = pred[:, :, 0:1]
        y2 = pred[:, :, 1:2]

        # 입실론 값 생성
        epsilon = tf.keras.backend.epsilon()

        # 유클리디안 거리 계산
        distance = tf.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + epsilon)
        return distance
    
    def result(self):
        # 평균 거리 계산
        return tf.where(self.count==0.0, 0.0, self.total / self.count)
    
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