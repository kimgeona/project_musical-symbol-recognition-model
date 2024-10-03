# 필요한 패키지
import tensorflow as tf             # 텐서플로


# 사용자 손실 함수
class WeightedBC(tf.keras.losses.Loss):
    """
    WeightedBinaryCrossentropy : 가중치화된 이진 크로스 엔트로피.

    해당 클래스는 소수 레이블에 더 많은 가중치를 부과하여 손실을 계산한다.
    """
    def __init__(self, total_count, class_count, name='WeightedBC', **kwargs):
        super(WeightedBC, self).__init__(name=name, **kwargs)
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
    

#
class WeightedIoU(tf.keras.losses.Loss):
    """
    WeightedIoU : 가중치화된 IoU.

    해당 클래스는 바운딩 박스의 중첩 정도, 중심점 거리, 비율에 대한 손실을 계산한다.
    해당 클래스는 바운딩 박스의 시작 좌표와 끝 좌표 뒤바뀜에 대한 손실을 계산한다.
    해당 클래스는 이미지의 상대적 중심에 대한 손실을 계산한다.
    해당 클래스는 소수 레이블에 더 많은 가중치를 부과하여 손실을 계산한다.
    """
    def __init__(self, total_count, class_count, name='WeightedIoU', **kwargs):
        super(WeightedIoU, self).__init__(name=name, **kwargs)
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
        
        # shape을 (batch_size, class_count, 1) 으로 변경
        self.weight_true = tf.reshape(self.weight_true, shape=(-1, len(class_count), 1))
        self.weight_false = tf.reshape(self.weight_false, shape=(-1, len(class_count), 1))

    def call(self, y_true, y_pred):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # 배치 크기 알아내기
        batch_size = tf.shape(y_true)[0]

        # shape 변경
        y_true = tf.reshape(y_true, shape=(batch_size, -1, 7))
        y_pred = tf.reshape(y_pred, shape=(batch_size, -1, 7))

        # 바운딩 박스 좌표와 상대적 중심 좌표 분리
        box_true = y_true[:, :, 0:4]
        
        # 손실 계산
        loss = tf.where(self.box_area(box_true)==0, self.to_zero_loss(y_pred), self.iou_loss(y_true, y_pred))

        # 소수 레이블에 더 많은 가중치를 부여
        # : (batch_size, class_count, 1)
        weightedLoss = tf.where(box_true==0.0, loss * self.weight_false, loss * self.weight_true)

        # 각 클래스마다 계산된 손실 합산
        # : (batch_size,)
        sumLoss = tf.reduce_sum(weightedLoss, axis=[-1, -2])

        # 배치 크기만큼 손실 평균
        # : ()
        meanLoss = tf.reduce_mean(sumLoss, axis=-1)

        # 손실 반환
        return meanLoss

    # 0. 좌표 정렬
    def sort_coordinate(self, box):
        # 각각의 좌표로 나누기
        x1 = box[:, :, 0:1]
        y1 = box[:, :, 1:2]
        x2 = box[:, :, 2:3]
        y2 = box[:, :, 3:4]
        # 좌표 크기 비교
        new_x1 = tf.minimum(x1, x2)
        new_y1 = tf.minimum(y1, y2)
        new_x2 = tf.maximum(x1, x2)
        new_y2 = tf.maximum(y1, y2)
        # 텐서 하나로 만들기
        return tf.concat([new_x1, new_y1, new_x2, new_y2], axis=-1)

    # 1-1. 박스 넓이 계산
    def box_area(self, box):
        # 각각의 좌표로 나누기
        x1 = box[:, :, 0:1]
        y1 = box[:, :, 1:2]
        x2 = box[:, :, 2:3]
        y2 = box[:, :, 3:4]
        # 넓이 계산
        return tf.abs((x2-x1) * (y2-y1))

    # 1-2. IoU 계산
    def compute_IoU(self, box1, box2):
        # 예측 박스 좌표 정렬
        box2 = self.sort_coordinate(box2)

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
        intersection_area = self.box_area(box3)

        # 합집합 영역 계산
        union_area = box1_area + box2_area - intersection_area

        # 0~1 사이값 리턴
        return 1.0 - (intersection_area / union_area)

    # 2-1. 중심점 계산
    def get_center(self, box):
        # 각각의 좌표로 나누기
        x1 = box[:, :, 0:1]
        y1 = box[:, :, 1:2]
        x2 = box[:, :, 2:3]
        y2 = box[:, :, 3:4]
        # 중심점 계산
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy

    # 2-2. 박스 대각선 길이 계산
    def box_diagonal(self, box):
        # 각각의 좌표로 나누기
        x1 = box[:, :, 0:1]
        y1 = box[:, :, 1:2]
        x2 = box[:, :, 2:3]
        y2 = box[:, :, 3:4]
        # 대각선 길이 계산
        return tf.sqrt((x2 - x1) ** 2 + (y2 -y1) ** 2)

    # 2-3. 두 박스의 중심점 거리 계산 -> 0 이상의 값 리턴
    def center_distance(self, box1, box2):
        # 예측 박스 좌표 정렬
        box2 = self.sort_coordinate(box2)

        # 두 박스의 중심점 사이 거리
        b1_cx, b1_cy = self.get_center(box1)
        b2_cx, b2_cy = self.get_center(box2)
        box_distance = tf.sqrt((b2_cx - b1_cx) ** 2 + (b2_cy - b1_cy) ** 2)

        # 두 박스를 모두 포함하는 합집합 영역의 대각선 길이
        box4 = tf.concat([
            tf.minimum(box1[:, :, 0:1], box2[:, :, 0:1]),
            tf.minimum(box1[:, :, 1:2], box2[:, :, 1:2]),
            tf.maximum(box1[:, :, 2:3], box2[:, :, 2:3]),
            tf.maximum(box1[:, :, 3:4], box2[:, :, 3:4])
        ], axis=-1) 
        union_diagonal = self.box_diagonal(box4)
        
        # 0~1 사이값 리턴
        return box_distance / union_diagonal

    # 3-1. 각 박스의 너비와 높이 비율 계산 
    def box_ratio(self, box):
        # 각각의 좌표로 나누기
        x1 = box[:, :, 0:1]
        y1 = box[:, :, 1:2]
        x2 = box[:, :, 2:3]
        y2 = box[:, :, 3:4]
        # 너비와 높이 계산
        width = tf.abs(x2 - x1)
        height = tf.abs(y2 - y1)
        return height / (width + 1e-6) # 0 나누기 방지

    # 3-2. 종횡 비율 차이 계산
    def ratio_differences(self, box1, box2):
        # 예측 박스 좌표 정렬
        box2 = self.sort_coordinate(box2)

        # 각 박스 비율 계산
        b1_ratio = self.box_ratio(box1)
        b2_ratio = self.box_ratio(box2)

        # 0~1 사이값 리턴
        return tf.abs(b1_ratio - b2_ratio) / tf.maximum(b1_ratio, b2_ratio)

    # 4. 좌표 뒤바뀜 패널티 계산
    def coordinate_penalty(self, box2):
        # 각각의 좌표로 나누기
        x1 = box2[:, :, 0:1]
        y1 = box2[:, :, 1:2]
        x2 = box2[:, :, 2:3]
        y2 = box2[:, :, 3:4]

        # 예측 박스 좌표 바뀜 패널티 계산
        penalty_x = tf.maximum(0.0, x1 - x2)
        penalty_y = tf.maximum(0.0, y1 - y2)

        # 0~1 사이값 리턴
        return penalty_x + penalty_y

    # 5-1. IoU 손실 구하기
    def iou_loss(self, y_true, y_pred):
        # 바운딩 박스 좌표와 상대적 중심 좌표 분리
        box_true = y_true[:, :, 0:4]
        box_pred = y_pred[:, :, 0:4]
        point_true = y_true[:, :, 4:6]
        point_pred = y_pred[:, :, 4:6]

        # CIoU = IoU + 중심점 거리 + 종횡비 + 좌표 뒤바낌 패널티
        iou = self.compute_IoU(box_true, box_pred)          # 0~1 : IoU 계산
        distance = self.center_distance(box_true, box_pred) # 0~1 : 중심점 거리 계산
        ratio = self.ratio_differences(box_true, box_pred)  # 0~1 : 박스 비율 차이 계산
        penalty = self.coordinate_penalty(box_pred)         # 0~n :좌표 뒤바뀜 패널티

        # 중심 좌표 손실
        center_loss = self.box_diagonal(tf.concat([point_true, point_pred], axis=-1)) / self.box_diagonal(box_true)
        
        # 손실 계산
        return iou + distance + ratio + penalty + center_loss

    # 5-2. 0으로 수렴 손실 구하기
    def to_zero_loss(self, y_pred):
        # 모든 좌표 절대값 취하기
        y_pred = tf.abs(y_pred)

        # 로그 손실 구하기
        loss = tf.math.log(y_pred + 1.0)

        # 손실 합 반환
        return tf.reduce_sum(loss, axis=-1, keepdims=True)