# 필요한 패키지
import os                           # 운영체제 관련
import tensorflow as tf             # 텐서플로
import tensorflow_addons as tfa     # 텐서플로 애드온
import pandas as pd                 # 판다스
import my_function as myfn

# 데이터셋 주소
dataset_dir = os.path.join('.', 'datasets', 'MusicalSymbol-v.1.1.5')
csv_dir = os.path.join(dataset_dir, 'label.csv')

# 데이터 프레임
def pick(dataframe, keywords):
    # 빈 데이터 프레임 생성
    df = pd.DataFrame()
    # 기존 데이터 프레임 추출 시작
    for key in keywords:
        # 키워드 추출
        name_list = [s for s in dataframe.columns.to_numpy() if key in s]
        #
        if (len(keywords) > 1):
            # 추출된 키워드 열 끼리 병합
            for n in name_list:
                # 키워드 열 끼리 or 연산
                if key in df:   df[key.strip()] = df[key] | dataframe[n]
                else:           df[key.strip()] = dataframe[n]
        else:
            # 추출된 키워드로 데이터 프레임 새로 만들기
            for n in name_list:
                df[n.strip()] = dataframe[n]
    # 데이터 프레임 반환
    return df

# 이미지 레이블 카테고리화
def label_categorize(df):
    # 데이터 프레임 정리
    df_list = [
        pick(df, keywords=[ 
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
        pick(df, keywords=['staff-']).drop(columns=['staff-half-left', 'staff-half-right']),
        pick(df, keywords=['note']),
        pick(df, keywords=['accidental']),
        pick(df, keywords=['articulation']),
        pick(df, keywords=['dynamic']),
        pick(df, keywords=['octave']),
        pick(df, keywords=['ornament']),
        pick(df, keywords=['repetition']),
        pick(df, keywords=['clef']),
        pick(df, keywords=['key']),
        pick(df, keywords=['measure']),
        pick(df, keywords=['rest']),
        pick(df, keywords=['time']),
    ]

    # Pandas 데이터 프레임을 TensorFlow 텐서로 변환
    df_tensor = tuple([tf.convert_to_tensor(df.values, dtype=tf.int16) for df in df_list])

    # 생성된 데이터셋 정보 출력
    print('생성된 데이터셋 정보')
    print('--------------------------------')
    for i, t in enumerate(df_tensor):
        print('out shape {:<3} : {}'.format(i, t.shape))
    print('--------------------------------')
    print('[', end='')
    for i, t in enumerate(df_tensor):
        if i==0 : print('{}'.format(t.shape[-1]), end='')
        else    : print(', {}'.format(t.shape[-1]), end='')
    print(']')
    # 
    return df_tensor

# 이미지 레이블 준비
def prepare_label(csv_dir):
    # csv 데이터 불러오기
    df = pd.read_csv(csv_dir)                                       # CSV 파일 불러오기
    df['name_int'] = df['name'].str.extract('(\d+)').astype(int)    # name 열을 정수값으로 추출
    df = df.sort_values('name_int')                                 # 정수값을 기준으로 정렬
    df.reset_index(drop=True, inplace=True)                         # 현재 데이터 순서로 인덱스 초기화
    df = df.drop(columns=['name', 'name_int'])                      # 필요 없는 name, name_int 열 제거
    # 레이블 카테고리화
    df = label_categorize(df)
    # 텐서플로 데이터셋으로 변환
    ds = tf.data.Dataset.from_tensor_slices(df)
    return ds

# 이미지 이름 준비
def prepare_image(dataset_dir):
    # 이미지 주소 불러오기
    files = os.listdir(dataset_dir)
    files = [file.split('.')[0] for file in files if file.split('.')[1] == 'png']
    files = [int(file) for file in files]
    files.sort()
    files = [os.path.join(dataset_dir,str(file)+'.png') for file in files]
    # 텐서플로 데이터셋으로 변환
    ds = tf.data.Dataset.from_tensor_slices(files)
    return ds

# 데이터셋 생성
def make():
    # 개별 데이터셋 생성
    ds_img = prepare_image(dataset_dir)
    ds_label = prepare_label(csv_dir)

    # 하나의 데이터셋 생성
    ds = tf.data.Dataset.zip((ds_img, ds_label))

    # map
    ds = ds.map(myfn.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)      # 이미지 불러오기
    ds = ds.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 이동
    ds = ds.map(myfn.rotate_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)    # 이미지 회전
    ds = ds.map(myfn.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 확대 및 축소
    ds = ds.map(myfn.shake_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)     # 이미지 진동
    ds = ds.map(myfn.add_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)       # 이미지 잡음

    # cache
    ds = ds.cache() # 캐싱

    # shuffle
    ds = ds.shuffle(buffer_size=25000)  # 셔플

    # 데이터셋 반환
    return ds


