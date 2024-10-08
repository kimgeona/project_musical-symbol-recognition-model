# 필요 패키지들
import os
import cv2
import numpy as np
import torch
import torchvision.ops as ops
# 
from ultralytics import YOLO

# 이미지 저장
def save(
        image, dir,
        boxes, confidences, classes,
        class_names=None,
        font_scale=0.5, thickness=2
    ):
    # 바운딩 박스가 그려진 이미지 생성
    preview_image = draw_boxes_on_image(
        image, boxes, confidences, classes, class_names, font_scale, thickness
    )
    # 저장
    cv2.imwrite(dir, preview_image)

# 이미지 미리보기
def preview(
        image, 
        boxes, confidences, classes,
        class_names=None,
        window_height=900,
        font_scale=0.5, thickness=2
    ):
    # 창 크기 고정
    cv2.namedWindow('Preview Image', cv2.WINDOW_NORMAL)

    # 창 크기 계산
    height, width, channel = image.shape
    window_h = window_height
    window_w = int(width * (window_h / height))

    # 창 크기 조절
    cv2.resizeWindow('Preview Image', window_w, window_h)

    # 바운딩 박스가 그려진 이미지 생성
    preview_image = draw_boxes_on_image(
        image, boxes, confidences, classes, class_names, font_scale, thickness
    )

    # 새로운 크기로 이미지 리사이즈 (선명도 유지 위해 INTER_AREA 사용)
    preview_image = cv2.resize(preview_image, (window_w, window_h), interpolation=cv2.INTER_AREA)

    # 바운딩 박스가 그려진 이미지 미리보기
    cv2.imshow('Preview Image', preview_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 이미지에 바운딩 박스 그리기 (최대 20개 클래스까지 가능)
def draw_boxes_on_image(
        image, 
        boxes, confidences, classes,
        class_names=None,
        font_scale=0.5, thickness=2
    ):
    # 이미지 복사 (원본 이미지를 수정하지 않도록)
    img_copy = image.copy()

    # 클래스 ID와 색상을 매핑하는 딕셔너리
    class_colors = {
        0: (0, 255, 0),         # 클래스  0: 초록색 -> staff
        1: (192, 192, 192),     # 클래스  1: 회색   -> clef
        2: (255, 0, 255),       # 클래스  2: 자홍색 -> key
        3: (255, 255, 0),       # 클래스  3: 청록색 -> measure
        4: (0, 0, 255),         # 클래스  4: 빨간색 -> rest
        5: (128, 128, 0),       # 클래스  5: 올리브 -> time
        6: (255, 0, 0),         # 클래스  6: 파란색 -> note
        7: (75, 0, 130),        # 클래스  7: 인디고         -> accidental
        8: (255, 192, 203),     # 클래스  8: 핑크색         -> articulation
        9: (0, 255, 127),       # 클래스  9: 시안           -> dynamic
        10: (255, 165, 0),      # 클래스 10: 오렌지색       -> glissando
        11: (139, 69, 19),      # 클래스 11: 초콜릿색       -> octave
        12: (0, 0, 128),        # 클래스 12: 어두운 파란색  -> ornament
        13: (0, 128, 128),      # 클래스 13: 어두운 청록색  -> repetition
        14: (0, 255, 255),      # 클래스 14: 노란색         -> tie
        15: (128, 0, 0),        # 클래스 15: 어두운 빨간색
        16: (0, 128, 0),        # 클래스 16: 어두운 초록색
        17: (255, 105, 180),    # 클래스 17: 딸기색
        18: (255, 215, 0),      # 클래스 18: 금색
        19: (255, 20, 147),     # 클래스 19: 딥 핑크
    }

    # 박스, 점수, 클래스 순회하며 표시
    for i in range(len(boxes)):
        # 박스 좌표 (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, boxes[i])

        # 현재 클래스에 대한 색상 추출
        class_id = int(classes[i])
        color = class_colors[class_id]

        # 바운딩 박스 그리기
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)

        # 클래스 이름과 점수
        class_id = int(classes[i])
        confidence = float(confidences[i])
        label = f'{confidence:.2f}'  # 점수

        # 클래스 이름이 제공되었으면 추가
        if class_names is not None:
            label = f'{class_names[class_id]}: {confidence:.2f}'

        # 텍스트의 위치와 크기 설정
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img_copy, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, thickness=cv2.FILLED)
        cv2.putText(img_copy, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    #
    return img_copy


# 악상기호 대분류 모델
class SymbolDetector:
    def __init__(
        self, 
        image_dirs,             # 분석할 이미지 주소들
        note_confidences=0.7,   # 인식 대상의 정확도 임계값
        note_iou_threshold=0.8, # NMS IoU 임계값
    ):
        # 모델 불러오기
        model_dir = os.path.join(os.path.dirname(__file__), 'best.pt')
        self.model = YOLO(model_dir)
        
        # 이미지 분할 정보
        self.split_height = 416 # YOLO 모델 입력 크기 h
        self.split_width = 416  # YOLO 모델 입력 크기 w
        self.overlap = 128      # 겹치는 부분의 크기

        # 객체 탐지 파라미터 저장
        self.note_confidences=note_confidences      # 인식 대상의 정확도 임계값
        self.note_iou_threshold=note_iou_threshold  # NMS IoU 임계값

        # 이미지들 불러오기
        self.images = self.__load_images(image_dirs)
                
    # 이미지들 불러오기
    def __load_images(self, dirs):
        # 불러온 이미지들 목록
        images = []

        for dir in dirs:
            # 이미지 로드
            image = cv2.imread(dir)
            if image is None:
                print('이미지 "{dir}"를 찾을 수 없거나 불러올 수 없습니다.')
                continue

            # 이미지 색상 반전
            images.append(255 - image)

        # 불러온 이미지들 반환
        return images

    # 하나의 이미지에 대한 예측 수행
    def __predict(self, image, preview):
        # 이미지 높이 계산
        height, width, channels = image.shape

        # 검출할 데이터
        boxes = []
        confidences = []
        classes = []

        # 이미지 분할 및 YOLO 적용
        for y in range(0, height, self.split_height - self.overlap):
            for x in range(0, width, self.split_width - self.overlap):
                # 경계 체크 및 패딩 추가
                split_img = np.zeros((self.split_height, self.split_width, 3), dtype=np.uint8)
                img_patch = image[y:y + self.split_height, x:x + self.split_width]

                # 패딩 처리
                split_img[:img_patch.shape[0], :img_patch.shape[1]] = img_patch

                # 이미지의 픽셀값이 전부 0이면 인식 건너뛰기
                if np.all(split_img == 0):
                    continue  # 인식을 건너뜀

                # YOLO 예측
                result = self.model.predict(split_img, conf=self.note_confidences)
                
                # 결과 저장
                for r in result:
                    # 바운딩 박스의 좌표에 x, y 오프셋을 더하여 원본 이미지 좌표로 변환
                    xyxy = r.boxes.xyxy.clone()
                    xyxy[:, [0, 2]] += x
                    xyxy[:, [1, 3]] += y
                    # 텐서 값이 있는 것(유효한 박스들)들만 추가
                    if xyxy.numel() > 0:
                        boxes.append(xyxy)
                        confidences.append(r.boxes.conf)
                        classes.append(r.boxes.cls)
                
                # 인식되는 과정 미리보기
                if preview:
                    # 하나의 텐서로 만들기
                    if boxes:
                        boxes_tmp = torch.cat(boxes)
                        confidences_tmp = torch.cat(confidences)
                        classes_tmp = torch.cat(classes)
                    else:
                        boxes_tmp = torch.tensor([])
                        confidences_tmp = torch.tensor([])
                        classes_tmp = torch.tensor([])
                    # preview 이미지 생성
                    preview_image = draw_boxes_on_image(
                        255 - image,
                        boxes_tmp, confidences_tmp, classes_tmp
                    )
                    # 이미지 크기에 맞춰 윈도우 크기 변경
                    self.__preview_window_resize(preview_image)
                    # 이미지 화질 선명하기 크기 변경
                    preview_image = self.__resize_image(preview_image)
                    # preview 미리보기
                    cv2.imshow('Preview Image', preview_image)
                    cv2.waitKey(1)

        # 각 이미지마다 인식된 박스 데이터 결과를 하나의 텐서로 결합
        if boxes:
            boxes = torch.cat(boxes)
            confidences = torch.cat(confidences)
            classes = torch.cat(classes)
        else:
            boxes = torch.tensor([])
            confidences = torch.tensor([])
            classes = torch.tensor([])
        
        # 검출된 데이터들 반환
        return boxes, confidences, classes

    # 여러이미지의 바운딩 박스 통합
    def __assemble(self, boxes, confidences, classes):
        # 최종 결과
        result_boxes = []
        result_confidences = []
        result_classes = []

        # 클래스별로 NMS 적용 및 결과 통합 
        for class_id in classes.unique():
            # 특정 클래스만 필터링
            mask = classes == class_id

            # 박스 좌표, 점수, 클래스 필터링
            masked_boxes = boxes[mask]
            masked_confidences = confidences[mask]
            masked_classes = classes[mask]

            # NMS 적용
            # TODO: 특정 인덱스(ex: staff)는 겹치면 그냥 하나로 만들어버리기, 나머지는 NMS 적용
            nms_indices = ops.nms(masked_boxes, masked_confidences, self.note_iou_threshold) # iou_threshold=0.8 이상이면 합침

            # 최종 결과에 남은 박스 추가
            result_boxes.append(masked_boxes[nms_indices])
            result_confidences.append(masked_confidences[nms_indices])
            result_classes.append(masked_classes[nms_indices])

        # 각 클래스마다 추출된 NMS 결과를 하나의 텐서로 결합
        if result_boxes:
            result_boxes = torch.cat(result_boxes)
            result_confidences = torch.cat(result_confidences)
            result_classes = torch.cat(result_classes)
        else:
            result_boxes = torch.tensor([])
            result_confidences = torch.tensor([])
            result_classes = torch.tensor([])

        # 최종 결과 반환
        return result_boxes, result_confidences, result_classes

    # 이미지 크기에서 벗어나는 좌표 처리
    def __inspection(self, image, boxes, confidences, classes):
        # 이미지 높이 계산
        height, width, channels = image.shape

        # 기존 이미지의 너비와 높이 크기를 넘어가는 좌표 클리핑
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)   # x_min
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)  # y_min
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)   # x_max
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)  # y_max

        # 박스의 크기가 0인 것들 제거
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]

        # 너비와 높이가 모두 0이 아닌 박스만 필터링
        mask = (box_w > 0) & (box_h > 0)

        # 유효한 박스만 필터링
        boxes = boxes[mask]
        confidences = confidences[mask]
        classes = classes[mask]

        # 결과 반환
        return boxes, confidences, classes

    # 창 크기 계산
    def __preview_window_resize(self, image, window_height=900):
        # 창의 크기 조절
        cv2.namedWindow('Preview Image', cv2.WINDOW_NORMAL)  # 창 크기 고정

        # 창 크기 계산
        height, width, channel = image.shape
        self.window_h = window_height
        self.window_w = int(width * (self.window_h / height))
        cv2.resizeWindow('Preview Image', self.window_w, self.window_h)

    # 이미지 화질 선명하게 크기 변경
    def __resize_image(self, image):
        # 새로운 크기로 이미지 리사이즈 (선명도 유지 위해 INTER_AREA 사용)
        resized_image = cv2.resize(image, (self.window_w, self.window_h), interpolation=cv2.INTER_AREA)
        return resized_image

    # 객체 인식 수행
    def detact(self, preview=False):
        # 이미지들 에서 탐지된 객체
        self.datas = []

        # 객체 탐지 시작
        for image in self.images:
            # 이미지 미리보기
            if preview:
                self.__preview_window_resize(image) # 윈도우 크기 변경
                cv2.imshow('Preview Image', self.__resize_image(255 - image))
                cv2.waitKey(1)
            # 데이터 검출
            boxes, confidences, classes = self.__predict(image, preview)
            # 데이터 결합
            boxes, confidences, classes = self.__assemble(boxes, confidences, classes)
            # 데이터 검사
            boxes, confidences, classes = self.__inspection(image, boxes, confidences, classes)
            # 데이터 저장
            self.datas.append((boxes, confidences, classes))
            # 이미지 미리보기
            if preview:
                # preview 이미지 생성
                preview_image = draw_boxes_on_image(
                    255 - image,
                    boxes, confidences, classes
                )
                # 이미지 크기에 맞춰 윈도우 크기 변경
                self.__preview_window_resize(preview_image)
                # 이미지 화질 선명하기 크기 변경
                preview_image = self.__resize_image(preview_image)
                # preview 미리보기
                cv2.imshow('Preview Image', preview_image)
                cv2.waitKey(1000)
        
        # 이미지 미리보기 닫기
        if preview:
            cv2.waitKey(0)  # 키 입력 대기
            cv2.destroyAllWindows()  # 윈도우 닫기

        # 결과 반환
        return [255 - image for image in self.images], self.datas