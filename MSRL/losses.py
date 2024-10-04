# 필요한 패키지
import tensorflow as tf             # 텐서플로


# 사용자 손실 함수
class WeightedMultiTaskLoss(tf.keras.losses.Loss):
    """
    WeightedMultiTaskLoss : 가중치화된 다중 작업 손실.

    해당 클래스는 총 세가지 영역의 손실을 계산한다.

    x1, y1, x2, y2 :
    0~1 사이의 회귀 작업으로 Mean Absolute Error(MAE)를 계산한다.

    cx, cy:
    0~1 사이의 회귀 작업으로 Mean Absolute Error(MAE)를 계산한다.

    probability:
    0 또는 1의 다중 레이블 분류 작업으로, 이진 크로스 엔트로피 값을 계산한다.

    해당 클래스는 소수 레이블에 더 많은 가중치를 부과하여 손실을 계산한다.
    """
    def __init__(self, total_count, class_count, name='WeightedMultiTaskLoss', **kwargs):
        super(WeightedMultiTaskLoss, self).__init__(name=name, **kwargs)
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

        # Nan 값 확인
        tf.debugging.check_numerics(self.weight_true, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(self.weight_false, "NaN or Inf is not allowed")


    def call(self, y_true, y_pred):
        # 자료형 통일
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # Nan 값 확인
        tf.debugging.check_numerics(y_true, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(y_pred, "NaN or Inf is not allowed")

        # 배치 크기 알아내기
        batch_size = tf.shape(y_true)[0]

        # shape 변경
        y_true = tf.reshape(y_true, shape=(batch_size, -1, 7))
        y_pred = tf.reshape(y_pred, shape=(batch_size, -1, 7))

        # 세가지 영역으로 분리
        bounding_true = y_true[:, :, 0:4]       # x1, y1, x2, y2
        bounding_pred = y_pred[:, :, 0:4]       # 바운딩 박스 좌표
        center_true = y_true[:, :, 4:6]         # cx, cy
        center_pred = y_pred[:, :, 4:6]         # 악상기호 중심 좌표
        probability_true = y_true[:, :, 6:7]    # p
        probability_pred = y_pred[:, :, 6:7]    # 악상기호 존재 여부 

        # 손실 계산
        loss_1 = self.bounding_loss(bounding_true, bounding_pred)
        loss_2 = self.center_loss(center_true, center_pred)
        loss_3 = self.probility_loss(probability_true, probability_pred)
        tf.debugging.check_numerics(loss_1, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(loss_2, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(loss_3, "NaN or Inf is not allowed")

        # 손실
        loss = loss_1 + loss_2 + loss_3
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")

        # 가중치 곱
        loss_weighted = tf.where(probability_true==0.0, loss * self.weight_false, loss * self.weight_true)
        tf.debugging.check_numerics(loss_weighted, "NaN or Inf is not allowed")

        # 손실 합 : 각 배치마다 클래스들의 손실 합산
        loss_sum = tf.reduce_sum(loss_weighted, axis=[-1, -2])
        tf.debugging.check_numerics(loss_sum, "NaN or Inf is not allowed")

        # 손실 평균 : 배치들의 손실 평균
        loss_mean = tf.reduce_mean(loss_sum, axis=-1)
        tf.debugging.check_numerics(loss_mean, "NaN or Inf is not allowed")

        return loss_mean
    
    def bounding_loss(self, true, pred):
        # 두 좌표 차이 계산
        distance_abs = tf.abs(true - pred)

        # distance_abs 범위 제한 (0 + epsilon <= distance_abs <= 1 - epsilon)
        epsilon = tf.keras.backend.epsilon()
        distance_abs = tf.clip_by_value(1.0-distance_abs, epsilon, 1.0-epsilon)
        tf.debugging.check_numerics(distance_abs, "NaN or Inf is not allowed")

        # 차이가 클 수록 손실이 커지게
        loss = -tf.math.log(distance_abs)
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")

        # 손실 평균
        loss = tf.reduce_mean(loss, axis=-1, keepdims=True)
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")

        return loss
    
    def center_loss(self, true, pred):
        # 두 좌표 차이 계산
        distance_abs = tf.abs(true - pred)

        # distance_abs 범위 제한 (0 + epsilon <= distance_abs <= 1 - epsilon)
        epsilon = tf.keras.backend.epsilon()
        distance_abs = tf.clip_by_value(1.0-distance_abs, epsilon, 1.0-epsilon)
        tf.debugging.check_numerics(distance_abs, "NaN or Inf is not allowed")

        # 차이가 클 수록 손실이 커지게
        loss = -tf.math.log(distance_abs)
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")

        # 손실 평균
        loss = tf.reduce_mean(loss, axis=-1, keepdims=True)
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")
        
        return loss

    def probility_loss(self, true, pred):
        # 두 좌표 차이 계산 (이진 크로스 엔트로피)
        distance_abs = tf.abs(true - pred)

        # distance_abs 범위 제한 (0 + epsilon <= distance_abs <= 1 - epsilon)
        epsilon = tf.keras.backend.epsilon()
        distance_abs = tf.clip_by_value(1.0-distance_abs, epsilon, 1.0-epsilon)
        tf.debugging.check_numerics(distance_abs, "NaN or Inf is not allowed")

        # 차이가 클 수록 손실이 커지게
        loss = -tf.math.log(distance_abs)
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")

        # 손실 평균
        loss = tf.reduce_mean(loss, axis=-1, keepdims=True)
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")
        
        return loss

#
class WeightedIoU(tf.keras.losses.Loss):
    """
    WeightedIoU : 가중치화된 IoU.

    해당 클래스는 총 세가지 영역의 손실을 계산한다.

    x1, y1, x2, y2 :
    해당 클래스는 바운딩 박스의 중첩 정도, 중심점 거리, 비율에 대한 손실을 계산한다.
    해당 클래스는 바운딩 박스의 시작 좌표와 끝 좌표 뒤바뀜에 대한 손실을 계산한다.
    해당 클래스는 이미지의 상대적 중심에 대한 손실을 계산한다.

    cx, cy:
    설명

    probability:
    설명

    *해당 클래스에서 계산되는 손실들은 소수 레이블에 더 많은 가중치가 부과되도록 설계됨.
    """
    def __init__(self, image_height, image_width, total_count, class_count, name='WeightedIoU', **kwargs):
        super(WeightedIoU, self).__init__(name=name, **kwargs)
        # 이미지 크기 저장
        self.image_height = tf.constant(image_height, dtype=tf.float32)
        self.image_width = tf.constant(image_width, dtype=tf.float32)

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
        
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(y_true, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(y_pred, "NaN or Inf is not allowed")

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

        # 세가지 영역으로 분리
        bounding_true = y_true[:, :, 0:4]
        bounding_pred = y_pred[:, :, 0:4]
        center_true = y_true[:, :, 4:6]
        center_pred = y_pred[:, :, 4:6]
        probability_true = y_true[:, :, 6:7]
        probability_pred = y_true[:, :, 6:7]
        
        # 손실 계산
        # NOTE: 여기 제대로 계산되는게 맞겠지..?
        bounding_loss = tf.where(
            self.box_area(bounding_true)==0.0,          # 바운딩 박스 존재 여부
            self.to_zero_loss(bounding_pred),           # 존재 x : 바운딩 박스 크기를 0으로 만들도록 함
            self.iou_loss(bounding_true, bounding_pred) # 존재 o : 바운딩 박스가 정답과 같게끔 만들도록 함
        )
        center_loss = self.center_loss(center_true, center_pred)
        probability_loss = self.binary_crossentropy_loss(probability_true, probability_pred)
        
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(bounding_loss, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(center_loss, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(center_loss, "NaN or Inf is not allowed")

        # 총 손실
        # (batch_size, class_count, 1)
        total_loss = bounding_loss + center_loss + probability_loss
        tf.debugging.check_numerics(total_loss, "NaN or Inf is not allowed")

        # 소수 레이블에 더 많은 가중치를 부여
        # (batch_size, class_count, 1)
        weightedLoss = tf.where(probability_true==0.0, total_loss * self.weight_false, total_loss * self.weight_true)
        tf.debugging.check_numerics(weightedLoss, "NaN or Inf is not allowed")

        # 각 클래스마다 계산된 손실 합산
        # (batch_size,)
        sumLoss = tf.reduce_sum(weightedLoss, axis=[-1, -2])
        tf.debugging.check_numerics(sumLoss, "NaN or Inf is not allowed")

        # 배치 크기만큼 손실 평균
        # ()
        meanLoss = tf.reduce_mean(sumLoss, axis=-1)
        tf.debugging.check_numerics(meanLoss, "NaN or Inf is not allowed")

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
    
    # 1-2. 교집합이 있는지 확인
    def is_intersection(self, box1, box2):
        # 각각의 좌표로 나누기
        b1_x1 = box1[:, :, 0:1]
        b1_y1 = box1[:, :, 1:2]
        b1_x2 = box1[:, :, 2:3]
        b1_y2 = box1[:, :, 3:4]
        b2_x1 = box2[:, :, 0:1]
        b2_y1 = box2[:, :, 1:2]
        b2_x2 = box2[:, :, 2:3]
        b2_y2 = box2[:, :, 3:4]

        # 첫 번째 박스가 두 번째 박스와 겹치는지 확인
        condition1 = tf.logical_and(b1_x1 < b2_x2, b1_y1 < b2_y2)
        condition2 = tf.logical_and(b1_x2 > b2_x1, b1_y2 > b2_y1)
        
        # 두 번째 박스가 첫 번째 박스와 겹치는지 확인
        condition3 = tf.logical_and(b2_x1 < b1_x2, b2_y1 < b1_y2)
        condition4 = tf.logical_and(b2_x2 > b1_x1, b2_y2 > b1_y1)

        # 어느 한 조건이라도 만족하면 교집합이 있는 것
        has_inter = tf.logical_or(tf.logical_and(condition1, condition2), tf.logical_and(condition3, condition4))

        return has_inter

    # 1-3. IoU 계산
    def compute_IoU(self, box1, box2):
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(box1, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(box2, "NaN or Inf is not allowed")

        # 예측 박스 좌표 정렬
        box2 = self.sort_coordinate(box2)

        # box 넓이 계산
        box1_area = self.box_area(box1)
        box2_area = self.box_area(box2)

        # 교집합 영역이 있는지 검사
        intersection = self.is_intersection(box1, box2)

        # 교집합 영역의 박스 좌표 구함
        box3 = tf.concat([
            tf.maximum(box1[:, :, 0:1], box2[:, :, 0:1]),
            tf.maximum(box1[:, :, 1:2], box2[:, :, 1:2]),
            tf.minimum(box1[:, :, 2:3], box2[:, :, 2:3]),
            tf.minimum(box1[:, :, 3:4], box2[:, :, 3:4])
        ], axis=-1)
        
        # 교집합 영역의 넓이 계산
        intersection_area = tf.where(intersection, self.box_area(box3), 0.0)
        tf.debugging.check_numerics(intersection_area, "NaN or Inf is not allowed")

        # 합집합 영역 계산
        union_area = box1_area + box2_area - intersection_area
        tf.debugging.check_numerics(union_area, "NaN or Inf is not allowed")

        # iou 계산
        iou = tf.where(union_area==0.0, 0.0, intersection_area / union_area)
        tf.debugging.check_numerics(iou, "NaN or Inf is not allowed")

        # 0~1 사이값 리턴
        return 1.0 - iou

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
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(box1, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(box2, "NaN or Inf is not allowed")

        # 예측 박스 좌표 정렬
        box2 = self.sort_coordinate(box2)

        # 두 박스의 중심점 사이 거리
        b1_cx, b1_cy = self.get_center(box1)
        b2_cx, b2_cy = self.get_center(box2)
        box_distance = tf.sqrt((b2_cx - b1_cx) ** 2 + (b2_cy - b1_cy) ** 2)
        tf.debugging.check_numerics(box_distance, "NaN or Inf is not allowed")

        # 두 박스를 모두 포함하는 합집합 영역의 대각선 길이
        box4 = tf.concat([
            tf.minimum(box1[:, :, 0:1], box2[:, :, 0:1]),
            tf.minimum(box1[:, :, 1:2], box2[:, :, 1:2]),
            tf.maximum(box1[:, :, 2:3], box2[:, :, 2:3]),
            tf.maximum(box1[:, :, 3:4], box2[:, :, 3:4])
        ], axis=-1) 
        union_diagonal = self.box_diagonal(box4)
        tf.debugging.check_numerics(union_diagonal, "NaN or Inf is not allowed")

        # 0~1 사이값 리턴
        return tf.where(union_diagonal==0.0, 0.0, box_distance / union_diagonal)

    # 3-1. 각 박스의 너비와 높이 비율 계산 
    def box_width_height(self, box):
        # 각각의 좌표로 나누기
        x1 = box[:, :, 0:1]
        y1 = box[:, :, 1:2]
        x2 = box[:, :, 2:3]
        y2 = box[:, :, 3:4]

        # 너비와 높이 계산
        width = tf.abs(x2 - x1)
        height = tf.abs(y2 - y1)
        return width, height

    # 3-2. 각 박스의 너비와 높이 차이 계산
    def ratio_differences(self, box1, box2):
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(box1, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(box2, "NaN or Inf is not allowed")

        # 각 박스의 너비와 높이 계싼
        b1_w, b1_h = self.box_width_height(box1)
        b2_w, b2_h = self.box_width_height(box2)

        # 높이와 너비 차이 계산
        diff = tf.abs(b2_w - b1_w) + tf.abs(b2_h - b1_h)
        tf.debugging.check_numerics(diff, "NaN or Inf is not allowed")

        # 입실론 값 생성
        epsilon = tf.keras.backend.epsilon()

        # 로그 손실 구하기
        loss = tf.math.log(diff + 1.0 + epsilon)
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")

        # 0~1 사이값 리턴
        return loss

    # 4. 좌표 뒤바뀜 패널티 계산
    def coordinate_penalty(self, box2):
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(box2, "NaN or Inf is not allowed")

        # 각각의 좌표로 나누기
        x1 = box2[:, :, 0:1]
        y1 = box2[:, :, 1:2]
        x2 = box2[:, :, 2:3]
        y2 = box2[:, :, 3:4]

        # 예측 박스 좌표 바뀜 패널티 계산
        penalty_x = tf.maximum(0.0, x1 - x2)
        penalty_y = tf.maximum(0.0, y1 - y2)
        tf.debugging.check_numerics(penalty_x, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(penalty_y, "NaN or Inf is not allowed")

        # 0~1 사이값 리턴
        return penalty_x + penalty_y


    # 박스 존재 o : IoU 손실 구하기
    def iou_loss(self, true, pred):
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(true, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(pred, "NaN or Inf is not allowed")

        # CIoU = IoU + 중심점 거리 + 종횡비 + 좌표 뒤바낌 패널티
        iou = self.compute_IoU(true, pred)          # 0~1        : IoU 계산
        distance = self.center_distance(true, pred) # 0~1        : 중심점 거리 계산
        ratio = self.ratio_differences(true, pred)  # 0~log(n+1) : 박스 비율 차이 계산
        penalty = self.coordinate_penalty(pred)     # 0~n        : 좌표 뒤바뀜 패널티
        
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(iou, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(distance, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(ratio, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(penalty, "NaN or Inf is not allowed")

        # 손실 계산
        return iou + distance + ratio + penalty

    # 박스 존재 x : 박스 크기 0으로 만드는 손실 구하기
    def to_zero_loss(self, pred):
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(pred, "NaN or Inf is not allowed")

        # 모든 좌표 절대값 취하기
        pred = tf.abs(pred)

        # 좌표들 합 구하기
        pred = tf.reduce_sum(pred, axis=-1, keepdims=True)

        # 입실론 값 생성
        epsilon = tf.keras.backend.epsilon()

        # 로그 손실 구하기
        loss = tf.math.log(pred + 1.0 + epsilon)

        # Nan 값 생성시 에러
        tf.debugging.check_numerics(pred, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")

        # 손실 반환
        return loss
    
    # 중심 좌표 손실 구하기
    def center_loss(self, true, pred):
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(true, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(pred, "NaN or Inf is not allowed")

        # 두 점 사이 거리 구하기
        distance = self.box_diagonal(tf.concat([true, pred], axis=-1))

        # 입실론 값 생성
        epsilon = tf.keras.backend.epsilon()
        
        # log 정규화
        loss = tf.math.log(distance + 1.0 + epsilon)

        # Nan 값 생성시 에러
        tf.debugging.check_numerics(distance, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")

        # 손실 반환
        return loss
    
    # 이진 크로스 엔트로피 손실
    def binary_crossentropy_loss(self, true, pred):
        # Nan 값 생성시 에러
        tf.debugging.check_numerics(true, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(pred, "NaN or Inf is not allowed")

        # 입실론 값 생성
        epsilon = tf.keras.backend.epsilon()

        # 범위 제한 (0+epsilon <= y_pred <= 1-epsilon)
        pred = tf.clip_by_value(pred, epsilon, 1.0-epsilon)

        # Binary Crossentropy loss 계산
        loss = -(true * tf.math.log(pred) + (1-true) * tf.math.log(1-pred))

        # Nan 값 생성시 에러
        tf.debugging.check_numerics(pred, "NaN or Inf is not allowed")
        tf.debugging.check_numerics(loss, "NaN or Inf is not allowed")

        # 손실 반환
        return loss

