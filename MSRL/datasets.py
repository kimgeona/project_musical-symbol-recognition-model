# 필요한 패키지
import os                           # 운영체제 관련
import shutil
import tensorflow as tf             # 텐서플로
import pandas as pd                 # 판다스
import matplotlib.pyplot as plt     # 그래프 도구
import MSRL.functions as myfn


# 데이터셋 미리 보기
def preview(image, true, pred):
    import numpy as np                  # 넘파이
    import cv2                          # OpenCV

    # batch 크기 알아내기
    batch_size = tf.shape(image)[0]

    # shape 변경
    true = tf.reshape(true, shape=(batch_size, -1, 7))
    pred = tf.reshape(pred, shape=(batch_size, -1, 7))

    # 원래 스케일로 복원 (0~255)
    image = tf.squeeze(image, axis=-1)  # (32, 512, 192)
    image = (1.0 - image) * 255  # 0~1 범위를 0~255 범위로 변환
    image = tf.clip_by_value(image, 0, 255)  # 0과 255로 클리핑
    image = tf.cast(image, tf.uint8)  # uint8로 변환

    # 이미지 크기 조사
    image_height = tf.shape(image[0])[0].numpy()
    image_width = tf.shape(image[0])[1].numpy()

    # 컬러로 바꿔야 하므로 grayscale 이미지를 채널을 3개로 복제
    image = tf.repeat(image[..., tf.newaxis], repeats=3, axis=-1)  # (32, 512, 192, 3)

    # 바운딩 박스를 그릴 색상을 준비
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255)]  # 다양한 색상 리스트

    # 배치에서 첫 번째 이미지만을 시각화 예시로 사용
    image_np = image[0].numpy().astype(np.uint8)  # 첫 번째 이미지를 numpy로 변환
    true_boxes = true[0].numpy()  # 첫 번째 이미지의 실제 바운딩 박스
    pred_boxes = pred[0].numpy()  # 첫 번째 이미지의 예측 바운딩 박스

    # 바운딩 박스를 그리는 함수
    def draw_bounding_boxes(image, boxes, color):
        for box in boxes:
            cx = box[0] * image_width
            cy = box[1] * image_height
            w  = box[2] * image_width
            h  = box[3] * image_height
            rx = box[4] * image_width
            ry = box[5] * image_height
            p  = box[6]
            if p < 0.8:
                continue
            start_point = (int(cx - (w / 2)), int(cy - (h / 2)))
            end_point = (int(cx + (w / 2)), int(cy + (h / 2)))
            relative_point = (int(rx), int(ry))
            cv2.rectangle(image, start_point, end_point, color, 2)
            cv2.drawMarker(image, relative_point, color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        return image

    # 실제 바운딩 박스 그리기 (파란색)
    image_with_true_boxes = draw_bounding_boxes(image_np.copy(), true_boxes, colors[0])

    # 예측 바운딩 박스 그리기 (빨간색)
    image_with_pred_boxes = draw_bounding_boxes(image_np.copy(), pred_boxes, colors[1])

    # 이미지를 시각화
    plt.figure(figsize=(6, 9))

    # 실제 바운딩 박스가 그려진 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(image_with_true_boxes)
    plt.title('True Bounding Boxes')
    plt.axis('off')

    # 예측 바운딩 박스가 그려진 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(image_with_pred_boxes)
    plt.title('Predicted Bounding Boxes')
    plt.axis('off')

    # 이미지 간격 조정
    plt.subplots_adjust(wspace=0.2)  # 이미지 사이의 간격 조정

    plt.show()

# 악상 기호 데이터셋 클래스
class MusicalSymbolDataset:
    # 초기화
    def __init__(self, dataset_dir):
        # 기존 데이터셋 주소 저장
        self.path = dataset_dir
        # csv 주소 생성
        csv_dir = [
            os.path.join(self.path, 'train', 'label.csv'),
            os.path.join(self.path, 'validation', 'label.csv'),
            os.path.join(self.path, 'test', 'label.csv')
        ]
        # 이미지 주소, 이미지 레이블
        self.img_dirs = []
        self.img_label = [pd.read_csv(dir) for dir in csv_dir]
        # 이미지 레이블 불러오기
        for i in range(len(3)):
            self.img_dirs.append(self.img_label[i]['name'].to_list())       # 이미지 주소 저장
            self.img_label[i] = self.img_label[i].drop(columns=['name'])    # 이미지 레이블 저장
        # 이미지 주소 생성
        self.img_dirs[0] = [os.path.join(self.path, 'train', name)      for name in self.img_dirs[0]]
        self.img_dirs[1] = [os.path.join(self.path, 'validation', name) for name in self.img_dirs[1]]
        self.img_dirs[2] = [os.path.join(self.path, 'test', name)       for name in self.img_dirs[2]]

    # 데이터 프레임 추출
    def __pick(self, dataframe, keywords):
        # 데이터 프레임 생성
        df = pd.DataFrame()
        # 데이터 프레임 구성
        for word in keywords:
            # Dataframe 에서 kewords 단어를 포함하는 열 이름 추출
            cols = [col for col in dataframe.columns if word in col]
            # keywords 단어들을 중심으로 열들 하나로 묶어 Dataframe에 저장
            if (len(keywords) > 1):
                for col in cols:
                    if word in df:  df[word.strip()] = df[word] | dataframe[col]
                    else:           df[word.strip()] = dataframe[col]
            # keywords 단어가 들어간 열들만 Dataframe 으로 만들기
            else:
                if cols:
                    df = dataframe[cols].copy()
                    df.columns = [col.strip() for col in cols]
        # 생성된 데이터 프레임 반환
        return df

    # 데이터셋 생성
    def __make(self):
        # 이미지 주소 데이터셋 생성
        ds_img = [tf.data.Dataset.from_tensor_slices(t) for t in self.img_dirs_edited]

        # 이미지 레이블 데이터셋 생성
        ds_label = [tf.data.Dataset.from_tensor_slices(t) for t in self.img_label_edited]
        
        # 하나의 데이터셋 생성
        dss = []
        for z in zip(ds_img, ds_label):
            dss.append(tf.data.Dataset.zip(z))

        # map : 전처리 적용
        for i, ds in enumerate(dss):
            # 마지막 데이터셋은 전처리 건너뛰기
            if i==len(dss)-1: 
                # test 전처리
                ds = ds.map(myfn.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)          # 이미지 불러오기
                ds = ds.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 이동
                ds = ds.map(myfn.cut_image,  num_parallel_calls=tf.data.experimental.AUTOTUNE)          # 이미지 잘라내기
                ds = ds.map(myfn.coords_clipping, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 좌표 잘라내기
                ds = ds.map(myfn.coords_convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 변환
                ds = ds.map(myfn.coords_scaling, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 스케일링
            else:
                # train, validation 전처리
                ds = ds.map(myfn.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)          # 이미지 불러오기
                #ds = ds.map(myfn.rotate_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)        # 이미지 회전
                #ds = ds.map(myfn.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 확대 및 축소
                ds = ds.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 이동
                ds = ds.map(myfn.cut_image,  num_parallel_calls=tf.data.experimental.AUTOTUNE)          # 이미지 잘라내기
                ds = ds.map(myfn.add_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)           # 이미지 잡음
                # ds = ds.map(myfn.shake_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 진동
                ds = ds.map(myfn.coords_clipping, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 좌표 잘라내기
                ds = ds.map(myfn.coords_convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 변환
                ds = ds.map(myfn.coords_scaling, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 스케일링

            # 전처리된 데이터셋 저장
            dss[i] = ds

        # cache
        for i, ds in enumerate(dss):
            ds = ds.cache() # 캐싱
            dss[i] = ds

        # shuffle
        for i, ds in enumerate(dss):
            ds = ds.shuffle(buffer_size=len(self.img_dirs[i]))  # 셔플
            dss[i] = ds

        # batch
        for i, ds in enumerate(dss):
            ds = ds.batch(32)
            dss[i] = ds

        # prefetch
        for i, ds in enumerate(dss):
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            dss[i] = ds

        # 생성된 데이터셋 저장
        if (len(ds)==1):
            self.tfds = dss[0]
        else:
            self.tfds = dss

        # 생성된 데이터셋 정보 출력
        self.ds_info()

    # 데이터셋 레이블 정보 출력
    def ds_info(self):
        for i, name in enumerate(['Train', 'Validation', 'Test']):
            print('-- TFDS {} ------------------'.format(name))
            # input
            print('input  : {}, {}'.format(self.img_dirs_edited[i].shape, self.img_dirs_edited[i].dtype))
            # output
            print('output : {}, {}'.format(self.img_label_edited[i].shape, self.img_label_edited[i].dtype))
            print('---------------------------------')
            # class info
            batch_size = self.img_label_edited[i].shape[0]
            tmp_tensor = tf.reshape(self.img_label_edited[i], shape=(batch_size, -1, 7))[:, :, 6]
            class_count = tf.reduce_sum(tmp_tensor, axis=0)
            class_count = class_count.numpy().tolist()
            print('total number of labels : {}'.format(batch_size))
            print('number of each class : {}'.format(class_count))
            print()

    # 데이터셋 1 : 전체 분류
    def ds_1(self, *, train_valid=True):
        # 이미지 레이블 클래스 변수 초기화
        self.label_classes = self.img_label.columns.to_numpy().tolist()

        # 이미지와 레이블 준비
        self.img_dirs_edited = tuple([tf.convert_to_tensor(self.img_dirs, dtype=tf.string)])
        self.img_label_edited = tuple([tf.convert_to_tensor(self.img_label.values, dtype=tf.int16)])

        # 데이터셋 생성
        self.__make()

        # 데이터셋 반환
        if train_valid:
            return self.tfds_train, self.tfds_validation
        else:
            return self.tfds
    
    # 데이터셋 2 : 카테고리 분류
    def ds_2(self, *, train_valid=True):
        # 이미지 레이블 클래스 변수 초기화
        self.label_classes = []

        # 데이터 프레임 추출
        df_list = [
            self.__pick(self.img_label, keywords=[ 
                'note',
                'accidental',
                'articulation',
                'dynamic',
                'octave',
                'ornament',
                'repetition',
                'clef',
                'key',
                'measure',
                'rest',
                'time'
            ])
        ]

        # 이미지와 레이블 준비
        self.img_dirs_edited = tuple([tf.convert_to_tensor(self.img_dirs, dtype=tf.string)])
        self.img_label_edited = tuple([tf.convert_to_tensor(df.values, dtype=tf.int16) for df in df_list])

        # 데이터셋 생성
        self.__make()

        # 데이터셋 반환
        if train_valid:
            return self.tfds_train, self.tfds_validation
        else:
            return self.tfds
    
    # 데이터셋 3 : 카테고리 분류, 상세 분류
    def ds_3(self, *, train_valid=True):
        # 이미지 레이블 클래스 변수 초기화
        self.label_classes = []

        # 데이터 프레임 추출
        df_list = [
            self.__pick(self.img_label, keywords=[ 
                'note',
                'accidental',
                'articulation',
                'dynamic',
                'octave',
                'ornament',
                'repetition',
                'clef',
                'key',
                'measure',
                'rest',
                'time'
            ]),
            self.__pick(self.img_label, keywords=['staff-']).drop(columns=['staff-half-left', 'staff-half-right']),
            self.__pick(self.img_label, keywords=['note']),
            self.__pick(self.img_label, keywords=['accidental']),
            self.__pick(self.img_label, keywords=['articulation']),
            self.__pick(self.img_label, keywords=['dynamic']),
            self.__pick(self.img_label, keywords=['octave']),
            self.__pick(self.img_label, keywords=['ornament']),
            self.__pick(self.img_label, keywords=['repetition']),
            self.__pick(self.img_label, keywords=['clef']),
            self.__pick(self.img_label, keywords=['key']),
            self.__pick(self.img_label, keywords=['measure']),
            self.__pick(self.img_label, keywords=['rest']),
            self.__pick(self.img_label, keywords=['time']),
        ]

        # 이미지와 레이블 준비
        self.img_dirs_edited = tuple([tf.convert_to_tensor(self.img_dirs, dtype=tf.string)])
        self.img_label_edited = tuple([tf.convert_to_tensor(df.values, dtype=tf.int16) for df in df_list])

        # 데이터셋 생성
        self.__make()

        # 데이터셋 반환
        if train_valid:
            return self.tfds_train, self.tfds_validation
        else:
            return self.tfds

    # 데이터셋 4 : 카테고리 분류 for Object Detection
    def ds_OD(self):
        # 이미지 레이블 클래스 변수 초기화
        self.label_classes = [
            'staff', 
            'note', 
            'accidental',
            'articulation',
            'dynamic',
            'glissando',
            'octave',
            'ornament',
            'repetition',
            'tie'
        ]

        # 데이터 프레임 추출
        df_list = []
        for i in range(len(self.dataset_dir)):
            picked_dfs = [self.__pick(self.img_label[i], keywords=[name]) for name in self.label_classes]
            picked_dfs = [self.__pick(df, keywords=['-x-1', '-y-1', '-x-2', '-y-2', '-cx', '-cy', '-probability']) for df in picked_dfs]
            for i, df in enumerate(picked_dfs):
                df.columns = [self.label_classes[i]+col for col in df.columns]
            picked_dfs = pd.concat(picked_dfs, axis=1)
            df_list.append(picked_dfs)
        
        # 이미지와 레이블 준비
        self.img_dirs_edited = [
            tf.convert_to_tensor(img_dir, dtype=tf.string) for img_dir in self.img_dirs
        ]
        self.img_label_edited = [
            tf.convert_to_tensor(df.values, dtype=tf.int16) for df in df_list
        ]

        # 데이터셋 생성
        self.__make()

        # 데이터셋 반환
        return self.tfds
    
    # 새로운 데이터셋 생성
    def ds_to_YOLO_format(self):
        # 이미지 레이블 클래스 변수 초기화
        self.label_classes = [
            'staff',
            'clef',
            'key',
            'measure',
            'rest',
            'time',
            'note', 
            'accidental',
            'articulation',
            'dynamic',
            'glissando',
            'octave',
            'ornament',
            'repetition',
            'tie'
        ]

        # 데이터 프레임 추출
        df_list = []
        for i in range(len(self.dataset_dir)):
            picked_dfs = [self.__pick(self.img_label[i], keywords=[name]) for name in self.label_classes]
            picked_dfs = [self.__pick(df, keywords=['-x-1', '-y-1', '-x-2', '-y-2', '-cx', '-cy', '-probability']) for df in picked_dfs]
            for i, df in enumerate(picked_dfs):
                df.columns = [self.label_classes[i]+col for col in df.columns]
            picked_dfs = pd.concat(picked_dfs, axis=1)
            df_list.append(picked_dfs)

        # 이미지와 레이블 준비
        self.img_dirs_edited = [
            tf.convert_to_tensor(img_dir, dtype=tf.string) for img_dir in self.img_dirs
        ]
        self.img_label_edited = [
            tf.convert_to_tensor(df.values, dtype=tf.int16) for df in df_list
        ]

        # 이미지 주소 데이터셋 생성
        ds_img = [tf.data.Dataset.from_tensor_slices(t) for t in self.img_dirs_edited]

        # 이미지 레이블 데이터셋 생성
        ds_label = [tf.data.Dataset.from_tensor_slices(t) for t in self.img_label_edited]
        
        # 하나의 데이터셋 생성
        dss = []
        for z in zip(ds_img, ds_label):
            dss.append(tf.data.Dataset.zip(z))

        # map : 전처리 적용
        for i, ds in enumerate(dss):
            # 마지막 데이터셋은 전처리 건너뛰기
            if i==len(dss)-1: 
                # test 전처리
                ds = ds.map(myfn.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)          # 이미지 불러오기
                #ds = ds.map(myfn.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 확대 및 축소
                ds = ds.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 이동
                ds = ds.map(myfn.cut_image_yolo,  num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 잘라내기
                ds = ds.map(myfn.coords_clipping, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 좌표 잘라내기
                ds = ds.map(myfn.coords_convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 변환
                ds = ds.map(myfn.coords_scaling, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 스케일링
                ds = ds.map(myfn.coords_delete, num_parallel_calls=tf.data.experimental.AUTOTUNE)       # 좌표 제거
            else:
                # train, validation 전처리
                ds = ds.map(myfn.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)          # 이미지 불러오기
                #ds = ds.map(myfn.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 확대 및 축소
                ds = ds.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 이동
                ds = ds.map(myfn.cut_image_yolo,  num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 잘라내기
                ds = ds.map(myfn.add_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)           # 이미지 잡음
                # ds = ds.map(myfn.shake_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 진동
                ds = ds.map(myfn.coords_clipping, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 좌표 잘라내기
                ds = ds.map(myfn.coords_convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 변환
                ds = ds.map(myfn.coords_scaling, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 스케일링
                ds = ds.map(myfn.coords_delete, num_parallel_calls=tf.data.experimental.AUTOTUNE)       # 좌표 제거

            # 전처리된 데이터셋 저장
            dss[i] = ds

        # cache
        for i, ds in enumerate(dss):
            ds = ds.cache() # 캐싱
            dss[i] = ds

        # # shuffle
        # for i, ds in enumerate(dss):
        #     ds = ds.shuffle(buffer_size=len(self.img_dirs[i]))  # 셔플
        #     dss[i] = ds

        # prefetch
        for i, ds in enumerate(dss):
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            dss[i] = ds

        # 새로운 데이터셋 주소 생성
        new_path = list(os.path.split(self.path))
        new_path = new_path[:-1] + ['TFDS_' + new_path[-1].split('_')[-1]]
        new_path = os.path.join(*new_path)

        # 저장
        for i, ds_name in enumerate(['train', 'validation', 'test']):
            # 데이터셋 이름 생성
            ds_path = os.path.join(new_path, ds_name)
            os.makedirs(ds_path, exist_ok=True)
            # 데이터셋에 이미지, 레이블 저장
            for count, (image, label) in enumerate(dss[i]):
                # 이미지, 레이블 주소 생성
                image_path = os.path.join(ds_path, f'{count}.png')
                image_path = image_path.encode('utf-8')
                label_path = os.path.join(ds_path, f'{count}.txt')
                label_path = label_path.encode('utf-8')
                # 이미지 저장
                scaled_image = tf.clip_by_value(image * 255.0, 0.0, 255.0)
                encoded_image = tf.image.encode_png(tf.cast(scaled_image, tf.uint8))
                tf.io.write_file(image_path, encoded_image)
                # 레이블 저장
                strs = []
                for index in tf.range(label.shape[0]):
                    # 해당 인덱스의 데이터 가져오기
                    data = label[index].numpy()
                    # probability가 1.0인 경우 확인
                    if data[0] == 1.0:
                        # cx, cy, w, h 값을 가져와서 문자열 생성
                        cx, cy, w, h = data[1:]
                        strs.append(f'{index.numpy()} {cx} {cy} {w} {h}')
                with open(label_path, 'w') as f:
                    f.write('\n'.join(strs))

     # 새로운 데이터셋 생성
    
    def save_as_YOLO11(self):
        # 이미지 레이블 목록
        self.label_classes = [
            'staff',
            'clef',
            'key',
            'measure',
            'rest',
            'time',
            'note', 
            'accidental',
            'articulation',
            'dynamic',
            'glissando',
            'octave',
            'ornament',
            'repetition',
            'tie'
        ]

        # 데이터 프레임 추출
        df_list = []
        for i in range(len(self.dataset_dir)):
            picked_dfs = [self.__pick(self.img_label[i], keywords=[name]) for name in self.label_classes]
            picked_dfs = [self.__pick(df, keywords=['-x-1', '-y-1', '-x-2', '-y-2', '-cx', '-cy', '-probability']) for df in picked_dfs]
            for i, df in enumerate(picked_dfs):
                df.columns = [self.label_classes[i]+col for col in df.columns]
            picked_dfs = pd.concat(picked_dfs, axis=1)
            df_list.append(picked_dfs)

        # 이미지와 레이블 준비
        self.img_dirs_edited = [
            tf.convert_to_tensor(img_dir, dtype=tf.string) for img_dir in self.img_dirs
        ]
        self.img_label_edited = [
            tf.convert_to_tensor(df.values, dtype=tf.int16) for df in df_list
        ]

        # 이미지 주소 데이터셋 생성
        ds_img = [tf.data.Dataset.from_tensor_slices(t) for t in self.img_dirs_edited]

        # 이미지 레이블 데이터셋 생성
        ds_label = [tf.data.Dataset.from_tensor_slices(t) for t in self.img_label_edited]
        
        # 하나의 데이터셋 생성
        dss = []
        for z in zip(ds_img, ds_label):
            dss.append(tf.data.Dataset.zip(z))

        # map : 전처리 적용
        for i, ds in enumerate(dss):
            # 마지막 데이터셋은 전처리 건너뛰기
            if i==len(dss)-1: 
                # test 전처리
                ds = ds.map(myfn.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)          # 이미지 불러오기
                #ds = ds.map(myfn.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 확대 및 축소
                ds = ds.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 이동
                ds = ds.map(myfn.cut_image_yolo,  num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 잘라내기
                ds = ds.map(myfn.coords_clipping, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 좌표 잘라내기
                ds = ds.map(myfn.coords_convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 변환
                ds = ds.map(myfn.coords_scaling, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 스케일링
                ds = ds.map(myfn.coords_delete, num_parallel_calls=tf.data.experimental.AUTOTUNE)       # 좌표 제거
            else:
                # train, validation 전처리
                ds = ds.map(myfn.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)          # 이미지 불러오기
                #ds = ds.map(myfn.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 확대 및 축소
                ds = ds.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 이동
                ds = ds.map(myfn.cut_image_yolo,  num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 잘라내기
                ds = ds.map(myfn.add_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)           # 이미지 잡음
                # ds = ds.map(myfn.shake_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 진동
                ds = ds.map(myfn.coords_clipping, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 좌표 잘라내기
                ds = ds.map(myfn.coords_convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 변환
                ds = ds.map(myfn.coords_scaling, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 좌표 스케일링
                ds = ds.map(myfn.coords_delete, num_parallel_calls=tf.data.experimental.AUTOTUNE)       # 좌표 제거

            # 전처리된 데이터셋 저장
            dss[i] = ds

        # cache
        for i, ds in enumerate(dss):
            ds = ds.cache() # 캐싱
            dss[i] = ds

        # # shuffle
        # for i, ds in enumerate(dss):
        #     ds = ds.shuffle(buffer_size=len(self.img_dirs[i]))  # 셔플
        #     dss[i] = ds

        # prefetch
        for i, ds in enumerate(dss):
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            dss[i] = ds

        # 새로운 데이터셋 주소 생성
        new_path = list(os.path.split(self.path))
        new_path = new_path[:-1] + ['TFDS_' + new_path[-1].split('_')[-1]]
        new_path = os.path.join(*new_path)

        # 저장
        for i, ds_name in enumerate(['train', 'validation', 'test']):
            # 데이터셋 이름 생성
            ds_path = os.path.join(new_path, ds_name)
            os.makedirs(ds_path, exist_ok=True)
            # 데이터셋에 이미지, 레이블 저장
            for count, (image, label) in enumerate(dss[i]):
                # 이미지, 레이블 주소 생성
                image_path = os.path.join(ds_path, f'{count}.png')
                image_path = image_path.encode('utf-8')
                label_path = os.path.join(ds_path, f'{count}.txt')
                label_path = label_path.encode('utf-8')
                # 이미지 저장
                scaled_image = tf.clip_by_value(image * 255.0, 0.0, 255.0)
                encoded_image = tf.image.encode_png(tf.cast(scaled_image, tf.uint8))
                tf.io.write_file(image_path, encoded_image)
                # 레이블 저장
                strs = []
                for index in tf.range(label.shape[0]):
                    # 해당 인덱스의 데이터 가져오기
                    data = label[index].numpy()
                    # probability가 1.0인 경우 확인
                    if data[0] == 1.0:
                        # cx, cy, w, h 값을 가져와서 문자열 생성
                        cx, cy, w, h = data[1:]
                        strs.append(f'{index.numpy()} {cx} {cy} {w} {h}')
                with open(label_path, 'w') as f:
                    f.write('\n'.join(strs))

# MSIG 데이터셋을 YOLOv11 데이터셋으로 변환해주는 클래스
class DatasetConvert:
    def __init__(self, MSIG_dataset_dir):
        # 기존 데이터셋 주소 저장
        self.path = MSIG_dataset_dir
        print('* 입력된 데이터셋')
        print('- {}'.format(self.path))
        print('')

        # 생성될 데이터세 주소 생성
        self.newPath = list(os.path.split(self.path))
        self.newPath = self.newPath[:-1] + ['TFDS_' + self.newPath[-1][5:]]
        self.newPath = os.path.join(*self.newPath)
        print('* 생성될 데이터셋')
        print('- {}'.format(self.newPath))
        print('')

        # 존재하는 디렉토리이면 제거
        if os.path.exists(self.newPath):
            shutil.rmtree(self.newPath)
            
        # 새로운 데이터셋 폴더 생성
        os.makedirs(os.path.join(self.newPath, 'train'))
        os.makedirs(os.path.join(self.newPath, 'validation'))
        os.makedirs(os.path.join(self.newPath, 'test'))

        print('* 새로운 데이터셋을 생성합니다.')
        # csv 주소 생성
        csv_dir = [
            os.path.join(self.path, 'train', 'label.csv'),
            os.path.join(self.path, 'validation', 'label.csv'),
            os.path.join(self.path, 'test', 'label.csv')
        ]

        # 이미지 주소, 이미지 레이블
        print('- csv 파일을 불러옵니다.')
        self.img_dirs = []
        self.img_labels = [pd.read_csv(dir) for dir in csv_dir]

        # 이미지 레이블 불러오기
        for i in range(3):
            self.img_dirs.append(self.img_labels[i]['name'].to_list())      # 이미지 주소 저장
            self.img_labels[i] = self.img_labels[i].drop(columns=['name'])  # 이미지 레이블 저장

        # 이미지 주소 생성
        self.img_dirs[0] = [os.path.join(self.path, 'train', name)      for name in self.img_dirs[0]]
        self.img_dirs[1] = [os.path.join(self.path, 'validation', name) for name in self.img_dirs[1]]
        self.img_dirs[2] = [os.path.join(self.path, 'test', name)       for name in self.img_dirs[2]]

        # 이미지 레이블 목록
        self.label_class = [
            'staff',
            'clef',
            'key',
            'measure',
            'rest',
            'time',
            'note', 
            'accidental',
            'articulation',
            'dynamic',
            'glissando',
            'octave',
            'ornament',
            'repetition',
            'tie'
        ]

        # 데이터 프레임 추출
        print('- csv 데이터를 추출합니다.')
        df_list = []
        for i in range(3):
            picked_dfs = [self.__pick(self.img_labels[i], keywords=[name]) for name in self.label_class]
            picked_dfs = [self.__pick(df, keywords=['-x-1', '-y-1', '-x-2', '-y-2', '-cx', '-cy', '-probability']) for df in picked_dfs]
            for i, df in enumerate(picked_dfs):
                df.columns = [self.label_class[i]+col for col in df.columns]
            picked_dfs = pd.concat(picked_dfs, axis=1)
            df_list.append(picked_dfs)

        # 이미지와 레이블 준비
        self.img_dirs_tfs = [
            tf.convert_to_tensor(img_dir, dtype=tf.string) for img_dir in self.img_dirs
        ]
        self.img_label_tfs = [
            tf.convert_to_tensor(df.values, dtype=tf.int16) for df in df_list
        ]

        # 이미지 주소 텐서플로우 데이터셋 생성
        img_dirs_tfdss = [tf.data.Dataset.from_tensor_slices(t) for t in self.img_dirs_tfs]

        # 이미지 레이블 텐서플로우 데이터셋 생성
        img_label_tfdss = [tf.data.Dataset.from_tensor_slices(t) for t in self.img_label_tfs]
        
        # 텐서플로우 데이터셋 생성
        print('- 텐서플로우 데이터셋을 생성합니다.')
        self.tfdss = []
        for z in zip(img_dirs_tfdss, img_label_tfdss):
            self.tfdss.append(tf.data.Dataset.zip(z))

        # map
        for i, tfds in enumerate(self.tfdss):
            # train, validation
            if i<2:
                tfds = tfds.map(myfn.DC_load_image,      num_parallel_calls=tf.data.experimental.AUTOTUNE) # 이미지 불러오기
                tfds = tfds.map(myfn.DC_image_shift,     num_parallel_calls=tf.data.experimental.AUTOTUNE) # 이미지 이동
                tfds = tfds.map(myfn.DC_image_cut,       num_parallel_calls=tf.data.experimental.AUTOTUNE) # 이미지 잘라내기
                tfds = tfds.map(myfn.DC_image_noise,     num_parallel_calls=tf.data.experimental.AUTOTUNE) # 이미지 잡음
                tfds = tfds.map(myfn.DC_coords_clipping, num_parallel_calls=tf.data.experimental.AUTOTUNE) # 좌표 잘라내기
                tfds = tfds.map(myfn.DC_coords_convert,  num_parallel_calls=tf.data.experimental.AUTOTUNE) # 좌표 변환
                tfds = tfds.map(myfn.DC_coords_scaling,  num_parallel_calls=tf.data.experimental.AUTOTUNE) # 좌표 스케일링
                tfds = tfds.map(myfn.DC_coords_delete,   num_parallel_calls=tf.data.experimental.AUTOTUNE) # 좌표 제거
            # test
            else:
                tfds = tfds.map(myfn.DC_load_image,      num_parallel_calls=tf.data.experimental.AUTOTUNE) # 이미지 불러오기
                tfds = tfds.map(myfn.DC_image_cut,       num_parallel_calls=tf.data.experimental.AUTOTUNE) # 이미지 잘라내기
                tfds = tfds.map(myfn.DC_coords_clipping, num_parallel_calls=tf.data.experimental.AUTOTUNE) # 좌표 잘라내기
                tfds = tfds.map(myfn.DC_coords_convert,  num_parallel_calls=tf.data.experimental.AUTOTUNE) # 좌표 변환
                tfds = tfds.map(myfn.DC_coords_scaling,  num_parallel_calls=tf.data.experimental.AUTOTUNE) # 좌표 스케일링
                tfds = tfds.map(myfn.DC_coords_delete,   num_parallel_calls=tf.data.experimental.AUTOTUNE) # 좌표 제거
            # 전처리된 데이터셋 저장
            self.tfdss[i] = tfds

        # NOTE: cash, prefetch 를 사용하면 커널 충돌이 일어남. 아마도 메모리 부족으로 생기는 것으로 예상됨.
        # cash
        # for i, tfds in enumerate(self.tfdss):
        #     tfds = tfds.cache()
        #     self.tfdss[i] = tfds

        # # shuffle
        # for i, ds in enumerate(dss):
        #     ds = ds.shuffle(buffer_size=len(self.img_dirs[i]))  # 셔플
        #     dss[i] = ds

        # # prefetch
        # for i, tfds in enumerate(self.tfdss):
        #     tfds = tfds.prefetch(tf.data.experimental.AUTOTUNE)
        #     self.tfdss[i] = tfds

        # 저장
        print('- 이미지와 레이블을 저장합니다.')
        for tfds in self.tfdss:
            for dir, image, label in tfds:
                # dir : 이미지와 레이블을 저장할 주소 생성
                image_path = dir.numpy().decode('utf-8')
                image_path = list(os.path.split(image_path))
                image_path[0] = image_path[0].replace(self.path, self.newPath)
                image_path = os.path.join(*image_path)
                label_path = list(os.path.split(image_path))
                label_path[1] = label_path[1].replace('.png', '.txt')
                label_path = os.path.join(*label_path)
                # image : 이미지 저장
                scaled_image = tf.clip_by_value(image * 255.0, 0.0, 255.0)
                encoded_image = tf.image.encode_png(tf.cast(scaled_image, tf.uint8))
                tf.io.write_file(image_path, encoded_image)
                # label : 레이블 저장
                strs = []
                for index in tf.range(label.shape[0]):
                    # 해당 인덱스의 데이터 가져오기
                    data = label[index].numpy()
                    # probability가 1.0인 경우 확인
                    if data[0] == 1.0:
                        # cx, cy, w, h 값을 가져와서 문자열 생성
                        cx, cy, w, h = data[1:]
                        strs.append(f'{index.numpy()} {cx} {cy} {w} {h}')
                with open(label_path, 'w') as f:
                    f.write('\n'.join(strs))
                # 정보 출력
                # print(f"Directory: {image_path}")
                # print(f"Image shape: {image.shape}")  # 이미지 텐서 크기 출력
                # print(f"Label shape: {label.shape}")  # 레이블 텐서 크기 출력
        print('- 생성을 완료하였습니다.')
        

    # 데이터 프레임 추출
    def __pick(self, dataframe, keywords):
        # 데이터 프레임 생성
        df = pd.DataFrame()
        # 데이터 프레임 구성
        for word in keywords:
            # Dataframe 에서 kewords 단어를 포함하는 열 이름 추출
            cols = [col for col in dataframe.columns if word in col]
            # keywords 단어들을 중심으로 열들 하나로 묶어 Dataframe에 저장
            if (len(keywords) > 1):
                for col in cols:
                    if word in df:  df[word.strip()] = df[word] | dataframe[col]
                    else:           df[word.strip()] = dataframe[col]
            # keywords 단어가 들어간 열들만 Dataframe 으로 만들기
            else:
                if cols:
                    df = dataframe[cols].copy()
                    df.columns = [col.strip() for col in cols]
        # 생성된 데이터 프레임 반환
        return df

# MSIG 데이터셋을 하나의 데이터셋으로 합쳐주는 클래스
class DatasetAssemble:
    def __init__(self, dataset_dirs, new_name):
        # 새로운 데이터셋 경로 계산
        new_dataset_dirs = os.path.join(os.path.split(dataset_dirs[0])[0], new_name)

        # 데이터셋 정보 출력
        print(f'* 합칠 데이터셋')
        for i, dir in enumerate(dataset_dirs):
            print(f': {dir}')
        print()
        print(f'* 생성될 데이터셋')
        print(f': {new_dataset_dirs}')
        print()

        # 존재하는 디렉토리이면 제거
        if os.path.exists(new_dataset_dirs):
            shutil.rmtree(new_dataset_dirs)
            
        # 새로운 데이터셋 폴더 생성
        os.makedirs(os.path.join(new_dataset_dirs, 'train'))
        os.makedirs(os.path.join(new_dataset_dirs, 'validation'))
        os.makedirs(os.path.join(new_dataset_dirs, 'test'))

        #
        csv_train = []
        csv_validation = []
        csv_test = []

        # csv 파일 불러오기 및 이미지 복사
        print(f'* 데이터셋 생성을 시작합니다')
        for dataset_dir in dataset_dirs:
            # csv 파일 읽기
            train_df = pd.read_csv(os.path.join(dataset_dir, 'train', 'label.csv'))
            valid_df = pd.read_csv(os.path.join(dataset_dir, 'validation', 'label.csv'))
            test_df  = pd.read_csv(os.path.join(dataset_dir, 'test', 'label.csv'))
            # csv 파일에 명시되어 있는 이미지들 조사
            train_img_dirs = [os.path.join(dataset_dir, 'train', dir) for dir in train_df['name'].tolist()]
            valid_img_dirs = [os.path.join(dataset_dir, 'validation', dir) for dir in valid_df['name'].tolist()]
            test_img_dirs  = [os.path.join(dataset_dir, 'test', dir) for dir in test_df['name'].tolist()]
            # 이미지 복사
            print(f'- {dataset_dir} 폴더의 이미지를 복사합니다.')
            for dir in train_img_dirs:
                shutil.copyfile(dir, os.path.join(
                    new_dataset_dirs,                                               # newDatasetName
                    'train',                                                        # train
                    os.path.split(dataset_dir)[-1] + '-' + os.path.split(dir)[-1]   # preDatasetName-0.png
                ))
            for dir in valid_img_dirs:
                shutil.copyfile(dir, os.path.join(
                    new_dataset_dirs,                                               # newDatasetName
                    'validation',                                                   # validation
                    os.path.split(dataset_dir)[-1] + '-' + os.path.split(dir)[-1]   # preDatasetName-0.png
                ))
            for dir in test_img_dirs:
                shutil.copyfile(dir, os.path.join(
                    new_dataset_dirs,                                               # newDatasetName
                    'test',                                                         # test
                    os.path.split(dataset_dir)[-1] + '-' + os.path.split(dir)[-1]   # preDatasetName-0.png
                ))
            # name 열 이름 수정
            train_df['name'] = os.path.split(dataset_dir)[-1] + '-' + train_df['name']
            valid_df['name'] = os.path.split(dataset_dir)[-1] + '-' + valid_df['name']
            test_df['name']  = os.path.split(dataset_dir)[-1] + '-' + test_df['name']
            # csv 파일 추가
            csv_train.append(train_df)
            csv_validation.append(valid_df)
            csv_test.append(test_df)

        # 데이터 프레임 하나로 합치기
        print('- CSV 파일들을 하나로 합칩니다.')
        df_train       = pd.concat(csv_train, ignore_index=True, sort=False).fillna(0)
        df_validation  = pd.concat(csv_validation, ignore_index=True, sort=False).fillna(0)
        df_test        = pd.concat(csv_test, ignore_index=True, sort=False).fillna(0)
        df_train[df_train.columns.difference(['name'])]           = df_train[df_train.columns.difference(['name'])].astype(int)
        df_validation[df_validation.columns.difference(['name'])] = df_validation[df_validation.columns.difference(['name'])].astype(int)
        df_test[df_test.columns.difference(['name'])]             = df_test[df_test.columns.difference(['name'])].astype(int)

        # csv 파일로 저장하기
        print('- CSV 파일들을 저장합니다.')
        df_train.to_csv(os.path.join(new_dataset_dirs, 'train', 'label.csv'), index=False)
        df_validation.to_csv(os.path.join(new_dataset_dirs, 'validation', 'label.csv'), index=False)
        df_test.to_csv(os.path.join(new_dataset_dirs, 'test', 'label.csv'), index=False)

        #
        print('- 생성을 완료하였습니다.')