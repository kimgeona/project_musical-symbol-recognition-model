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
    def __init__(self, dataset_dir=os.path.join('.', 'datasets', 'MusicalSymbol-v.1.1.5')):
        # 데이터셋 주소 정보 생성
        self.dataset_dir = dataset_dir                          # 데이터셋 주소
        self.csv_dir = os.path.join(dataset_dir, 'label.csv')   # 데이터셋 레이블 주소
        # 이미지 파일 목록 구하기
        self.img_dirs = os.listdir(self.dataset_dir)
        self.img_dirs = [file.split('.')[0] for file in self.img_dirs if file.split('.')[1] == 'png']
        self.img_dirs = [int(file) for file in self.img_dirs]
        self.img_dirs.sort()
        self.img_dirs = [os.path.join(self.dataset_dir, str(file) + '.png') for file in self.img_dirs]
        # 이미지 레이블 불러오기
        self.img_label = pd.read_csv(self.csv_dir)                                              # CSV 파일 불러오기
        self.img_label['name_int'] = self.img_label['name'].str.extract('(\d+)').astype(int)    # name 열을 정수값으로 추출
        self.img_label = self.img_label.sort_values('name_int')                                 # 정수값을 기준으로 정렬
        self.img_label.reset_index(drop=True, inplace=True)                                     # 현재 데이터 순서로 인덱스 초기화
        self.img_label = self.img_label.drop(columns=['name', 'name_int'])                      # 필요 없는 name, name_int 열 제거
        # 이미지 레이블 클래스 변수
        self.label_classes = []

    # 데이터 프레임 추출
    def __pick(self, dataframe, keywords):
        # 데이터 프레임 생성
        df = pd.DataFrame()
        # 데이터 프레임 구성
        for word in keywords:
            # Dataframe 에서 kewords 단어를 포함하는 열 이름 추출
            cols = [col for col in dataframe.columns.to_numpy() if word in col]
            # keywords 단어들을 중심으로 열들 하나로 묶어 Dataframe에 저장
            if (len(keywords) > 1):
                for col in cols:
                    if word in df:  df[word.strip()] = df[word] | dataframe[col]
                    else:           df[word.strip()] = dataframe[col]
            # keywords 단어가 들어간 열들만 Dataframe 으로 만들기
            else:
                for col in cols:
                    df[col.strip()] = dataframe[col]
        # 레이블 클래스 저장
        self.label_classes = self.label_classes + df.columns.to_numpy().tolist()
        # 생성된 데이터 프레임 반환
        return df

    # 데이터셋 생성
    def __make(self):
        # 개별 데이터셋 생성
        if len(self.img_dirs_edited) > 1:   ds_img = tf.data.Dataset.from_tensor_slices(self.img_dirs_edited)
        else:                               ds_img = tf.data.Dataset.from_tensor_slices(self.img_dirs_edited[0])
        if len(self.img_label_edited) > 1:  ds_label = tf.data.Dataset.from_tensor_slices(self.img_label_edited)
        else:                               ds_label = tf.data.Dataset.from_tensor_slices(self.img_label_edited[0])

        # 하나의 데이터셋 생성
        ds = tf.data.Dataset.zip((ds_img, ds_label))

        # map : 전처리 적용
        ds = ds.map(myfn.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 이미지 불러오기
        ds = ds.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 이동
        ds = ds.map(myfn.rotate_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)    # 이미지 회전
        ds = ds.map(myfn.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 확대 및 축소
        ds = ds.map(myfn.shake_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 진동
        ds = ds.map(myfn.add_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)       # 이미지 잡음

        # cache
        ds = ds.cache() # 캐싱

        # shuffle
        ds = ds.shuffle(buffer_size=len(self.img_dirs))  # 셔플

        # 생성된 데이터셋 저장
        self.tfds = ds

        # 훈련, 검증 데이터셋 저장
        self.tfds_validation = ds.take(320).batch(32).prefetch(tf.data.experimental.AUTOTUNE)  # 훈련 데이터셋
        self.tfds_train = ds.skip(320).batch(32)                                               # 검증 데이터셋

        # 생성된 데이터셋 정보 출력
        self.ds_info()

    # 데이터셋 레이블 정보 출력
    def ds_info(self):
        print('-- TFDS label shape ------------------')
        for i, t in enumerate(self.img_label_edited):
            print('ds {:<2} : {}'.format(i, t.shape))
        print()
        print('-- TFDS label class count ------------------')
        for i, t in enumerate(self.img_label_edited):
            print('ds {:<2} : ['.format(i), end='')
            for a, b in enumerate(tf.reduce_sum(t, axis=0).numpy()):
                if a==0 : print('{}'.format(b), end='')
                else    : print(', {}'.format(b), end='')
            print(']')
        print()
        print('-- MODEL input node ------------------')
        model_in = []
        for i, t in enumerate(self.img_dirs_edited):
            if isinstance(self.tfds.element_spec[0], tuple):
                model_in.append(list(t.shape) + list(self.tfds.element_spec[0][i].shape))
            else:
                model_in.append(list(t.shape) + list(self.tfds.element_spec[0].shape))
        print(model_in)
        print()
        print('-- MODEL output node ------------------')
        model_out = []
        for t in self.img_label_edited:
            model_out.append([t.shape[-1]])
        print(model_out)

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
