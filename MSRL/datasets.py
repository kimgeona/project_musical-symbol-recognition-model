# 필요한 패키지
import os                           # 운영체제 관련
import tensorflow as tf             # 텐서플로
import pandas as pd                 # 판다스
import matplotlib.pyplot as plt     # 그래프 도구
import MSRL.functions as myfn


# (작성중) 데이터셋 미리 보기
def preview(dataset, lable_class, num_images):
    return
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_images):
            ax = plt.subplot(4, 8, i + 1)
            plt.imshow(images[i], cmap='gray')
            print(i, ' : ', [name for name, mask in zip(lable_class, tf.gather(labels[0], i).numpy()) if mask==1])
            plt.title(i)
            plt.axis('off')
    plt.show()

# 악상 기호 데이터셋 클래스
class MusicalSymbolDataset:
    # 초기화
    def __init__(self, dataset_dir=os.path.join('.', 'datasets', 'MSIG_1.1.5_note-recognition')):
        # 데이터셋 주소 저장
        self.dataset_dir = [
            os.path.join(dataset_dir, 'train'),
            os.path.join(dataset_dir, 'validation'),
            os.path.join(dataset_dir, 'test')
        ]
        # 데이터셋 레이블 주소 저장
        self.csv_dir = [
            os.path.join(self.dataset_dir[0], 'label.csv'),
            os.path.join(self.dataset_dir[1], 'label.csv'),
            os.path.join(self.dataset_dir[2], 'label.csv')
        ]
        # 이미지 주소, 이미지 레이블
        self.img_dirs = []
        self.img_label = [pd.read_csv(dir) for dir in self.csv_dir]
        # 이미지 레이블 불러오기
        for i in range(len(self.dataset_dir)):
            self.img_label[i]['name_int'] = self.img_label[i]['name'].str.extract('(\d+)').astype(int)  # CSV 파일 불러오기
            self.img_label[i] = self.img_label[i].sort_values('name_int')                               # name 열을 정수값으로 추출
            self.img_label[i].reset_index(drop=True, inplace=True)                                      # 정수값을 기준으로 정렬
            self.img_dirs.append(self.img_label[i]['name'].to_list())                                   # 현재 데이터 순서로 인덱스 초기화
            self.img_label[i] = self.img_label[i].drop(columns=['name', 'name_int'])                    # 필요 없는 name, name_int 열 제거
        # 이미지 주소 생성
        for i in range(len(self.dataset_dir)):
            self.img_dirs[i] = [os.path.join(self.dataset_dir[i], name) for name in self.img_dirs[i]]

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
                continue
            # 전처리
            ds = ds.map(myfn.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)          # 이미지 불러오기
            #ds = ds.map(myfn.rotate_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)        # 이미지 회전
            #ds = ds.map(myfn.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 확대 및 축소
            #ds = ds.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 이동
            ds = ds.map(myfn.add_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)           # 이미지 잡음
            # ds = ds.map(myfn.shake_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)         # 이미지 진동
            ds = ds.map(myfn.coords_clipping, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 좌표 잘라내기
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
