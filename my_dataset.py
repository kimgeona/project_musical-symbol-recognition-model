# 필요한 패키지
import os                           # 운영체제 관련
import tensorflow as tf             # 텐서플로
import tensorflow_addons as tfa     # 텐서플로 애드온
import pandas as pd                 # 판다스
import my_function as myfn

# 데이터셋 주소
dataset_dir = os.path.join('.', 'datasets', 'MusicalSymbol-v.1.1.5')

# 레이블 텐서
t_dfs = None

# CSV 파일 Pandas 데이터 프레임을 불러오기
def load_dataframe(dataset_dir):

    # CSV 파일 Pandas 데이터 프레임으로 불러오기
    csv_dir = os.path.join(dataset_dir, 'label.csv')    # CSV 파일 주소 생성
    df = pd.read_csv(csv_dir)                           # CSV 파일 불러오기

    # 데이터 프레임 수정
    df['name_int'] = df['name'].str.extract('(\d+)').astype(int)    # name 열을 정수값으로 추출
    df = df.sort_values('name_int')                                 # 정수값을 기준으로 정렬
    df.reset_index(drop=True, inplace=True)                         # 현재 데이터 순서로 인덱스 초기화
    df = df.drop(columns=['name', 'name_int'])                      # 필요 없는 name, name_int 열 제거

    # 데이터 프레임 반환
    return df

# 데이터 프레임 텐서 데이터셋으로 준비
def dataframe_to_tensor(dataframe):
    # 데이터 프레임 정리
    dfs = [
        myfn.column_extract(dataframe, keywords=[ 
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
        myfn.column_pick(dataframe, keywords=['staff-']).drop(columns=['staff-half-left', 'staff-half-right']),
        myfn.column_pick(dataframe, keywords=['note']),
        myfn.column_pick(dataframe, keywords=['accidental']),
        myfn.column_pick(dataframe, keywords=['articulation']),
        myfn.column_pick(dataframe, keywords=['dynamic']),
        myfn.column_pick(dataframe, keywords=['octave']),
        myfn.column_pick(dataframe, keywords=['ornament']),
        myfn.column_pick(dataframe, keywords=['repetition']),
        myfn.column_pick(dataframe, keywords=['clef']),
        myfn.column_pick(dataframe, keywords=['key']),
        myfn.column_pick(dataframe, keywords=['measure']),
        myfn.column_pick(dataframe, keywords=['rest']),
        myfn.column_pick(dataframe, keywords=['time']),
    ]

    # Pandas 데이터 프레임을 TensorFlow 텐서로 변환
    t_dfs = [tf.convert_to_tensor(df.values, dtype=tf.int16) for df in dfs ]

    # 준비된 데이터셋 출력
    print('전체 분류')
    print('--------------------------------')
    print('df_all          :', t_dfs[0].shape)
    print()
    print('세부 분류')
    print('--------------------------------')
    print('df_pitch        :', t_dfs[1].shape)
    print('df_note         :', t_dfs[2].shape)
    print('df_accidental   :', t_dfs[3].shape)
    print('df_articulation :', t_dfs[4].shape)
    print('df_dynamic      :', t_dfs[5].shape)
    print('df_octave       :', t_dfs[6].shape)
    print('df_ornament     :', t_dfs[7].shape)
    print('df_repetition   :', t_dfs[8].shape)
    print('df_clef         :', t_dfs[9].shape)
    print('df_key          :', t_dfs[10].shape)
    print('df_measure      :', t_dfs[11].shape)
    print('df_rest         :', t_dfs[12].shape)
    print('df_time         :', t_dfs[13].shape)

    # 텐서 반환
    return t_dfs

# tfds 생성 후 반환
def make():
    # 데이터셋 생성
    dataset = tf.data.Dataset.list_files(os.path.join(dataset_dir, '*.png'))

    # map : make_data
    dataset = dataset.map(make_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # map : shift_image
    #dataset = dataset.map(myfn.shift_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # map : rotate_image
    dataset = dataset.map(myfn.rotate_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # map : scale_image
    #dataset = dataset.map(myfn.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # map : shake_image
    #dataset = dataset.map(myfn.shake_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # map : add_noise
    dataset = dataset.map(myfn.add_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # filter
    #dataset = dataset.filter(myfn.filter_dataset)

    # cache
    dataset = dataset.cache()

    # shuffle
    dataset = dataset.shuffle(buffer_size=25000)

    # 데이터셋 반환
    return dataset

# 데이터 (image, lable) 생성
@tf.function
def make_data(img_path):
    global t_dfs

    # 데이터셋 준비 되어 있는지 확인
    if t_dfs == None:
        df = load_dataframe(dataset_dir)    # 데이터셋 불러오기
        t_dfs = dataframe_to_tensor(df)     # 데이터셋 가공
    
    # 타겟 생성
    image = tf.io.read_file(img_path)               # 파일 로드
    image = tf.image.decode_png(image, channels=1)  # png 파일로 변환
    image = tf.cast(image, tf.float32)              # uint8 -> float32
    image = image / 255.0                           # 0~1 로 정규화
    image = 1.0 - image                             # 흑백 이미지 반전

    # 파일 이름에서 인덱스 얻기
    lable_index = tf.strings.split(img_path, os.path.sep)[-1]           # 파일 이름 추출
    lable_index = tf.strings.split(lable_index, '.')[0]                 # 확장자 제거
    lable_index = tf.strings.to_number(lable_index, out_type=tf.int32)  # 숫자 변환

    # 레이블 생성
    lable = tuple(tf.gather(t_df, lable_index) for t_df in t_dfs)       # t_dfs 에서 lable_index 번째 label 추출

    # 타겟과 레이블 반환
    return (image, lable)





