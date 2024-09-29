# 필요한 패키지
import tensorflow as tf             # 텐서플로

# inception module
def inception_module(x, filters):
    # 1x1 conv
    out1_1 = tf.keras.layers.Conv2D(filters[0], 1, strides=1, padding='same', activation='relu')(x)
    # 1x1 conv -> 3x3 conv
    out2_1 = tf.keras.layers.Conv2D(filters[1], 1, strides=1, padding='same', activation='relu')(x)
    out2_2 = tf.keras.layers.Conv2D(filters[2], 3, strides=1, padding='same', activation='relu')(out2_1)
    # 1x1 conv -> 5x5 conv
    out3_1 = tf.keras.layers.Conv2D(filters[3], 1, strides=1, padding='same', activation='relu')(x)
    out3_2 = tf.keras.layers.Conv2D(filters[4], 5, strides=1, padding='same', activation='relu')(out3_1)
    # 3x3 max_pool -> 1x1 conv
    out4_1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    out4_2 = tf.keras.layers.Conv2D(filters[5], 1, strides=1, padding='same', activation='relu')(out4_1)

    # out1_1 + out2_2 + out3_2 + out4_4
    output = tf.keras.layers.Concatenate(axis=-1)([out1_1, out2_2, out3_2, out4_2])
    return output

def layer_out(inputs, n, layer_name, pre_outputs=None, i=None):
    # 이미지 인식 층
    layer = tf.keras.layers.Dense(64, activation='relu')(inputs)
    layer = tf.keras.layers.Dense(32, activation='relu')(layer)

    # 출력층
    if pre_outputs is not None and i is not None: 
        layer_out = tf.keras.layers.Dense(n, activation='softmax')(layer)
    else:
        layer_out = tf.keras.layers.Dense(n, activation='softmax', name=layer_name)(layer)

    # 마스크 처리
    if pre_outputs is not None and i is not None:
        #mask = tf.gather(pre_outputs, i, axis=1)
        #mask = tf.expand_dims(mask, axis=1)
        mask = tf.keras.layers.Reshape((-1,))(pre_outputs[:, i])
        layer_out = tf.keras.layers.Multiply(name=layer_name)([layer_out, mask])
    
    return layer_out

def msrm(out_shape):
    # 모델 입력
    inputs = tf.keras.layers.Input(shape=(512, 192, 1))

    # 모델 층 : 대분류
    layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu')(inputs)
    layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(layer_1)
    layer_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(layer_1)
    layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(layer_1)
    layer_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(layer_1)
    layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(layer_1)

    # 모델 층 : 상세분류 
    layer_2 = inception_module(layer_1, [64, 128, 96, 32, 16, 32])
    layer_2 = inception_module(layer_2, [128, 192, 128, 96, 32, 64])
    layer_2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(layer_2)
    layer_2 = inception_module(layer_2, [192, 208, 96, 48, 16, 64])
    layer_2 = inception_module(layer_2, [160, 224, 112, 64, 24, 64])
    layer_2 = inception_module(layer_2, [128, 256, 128, 64, 24, 64])
    layer_2 = inception_module(layer_2, [112, 288, 144, 64, 32, 64])
    layer_2 = inception_module(layer_2, [256, 320, 160, 128, 32, 128])
    layer_2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(layer_2)
    layer_2 = inception_module(layer_2, [256, 320, 160, 128, 32, 128])
    layer_2 = inception_module(layer_2, [384, 384, 192, 128, 48, 128])
    layer_2 = tf.keras.layers.AveragePooling2D(pool_size=(16, 6))(layer_2)
    layer_2 = tf.keras.layers.Flatten()(layer_2)

    # 모델 출력 : 대분류
    layer_all = tf.keras.layers.Flatten()(layer_1)
    layer_all = tf.keras.layers.Dense(256, activation='relu')(layer_all)
    layer_all = tf.keras.layers.Dense(128, activation='relu')(layer_all)
    layer_all = tf.keras.layers.Dense(out_shape[0], activation='sigmoid', name='all')(layer_all)

    # 모델 출력 : 상세분류
    layer_pitch         = layer_out(layer_2, out_shape[1], 'pitch')
    layer_note          = layer_out(layer_2, out_shape[2], 'note', layer_all, 0)
    layer_accidental    = layer_out(layer_2, out_shape[3], 'accidental', layer_all, 1)
    layer_articulation  = layer_out(layer_2, out_shape[4], 'articulation', layer_all, 2)
    layer_dynamic       = layer_out(layer_2, out_shape[5], 'dynamic', layer_all, 3)
    layer_octave        = layer_out(layer_2, out_shape[6], 'octave', layer_all, 4)
    layer_ornament      = layer_out(layer_2, out_shape[7], 'ornament', layer_all, 5)
    layer_repetition    = layer_out(layer_2, out_shape[8], 'repetition', layer_all, 6)
    layer_clef          = layer_out(layer_2, out_shape[9], 'clef', layer_all, 7)
    layer_key           = layer_out(layer_2, out_shape[10], 'key', layer_all, 8)
    layer_measure       = layer_out(layer_2, out_shape[11], 'measure', layer_all, 9)
    layer_rest          = layer_out(layer_2, out_shape[12], 'rest', layer_all, 10)
    layer_time          = layer_out(layer_2, out_shape[13], 'time', layer_all, 11)

    # 모델 생성
    model = tf.keras.Model(inputs=[inputs], outputs=[
        layer_all,
        layer_pitch,
        layer_note,
        layer_accidental,
        layer_articulation,
        layer_dynamic,
        layer_octave,
        layer_ornament,
        layer_repetition,
        layer_clef,
        layer_key,
        layer_measure,
        layer_rest,
        layer_time
    ], name='MusicalSymbolRecognitionModel')

    # 모델 반환
    return model

# 악상기호 인식 모델 클래스
class MusicalSymbolModel:
    """
    악상기호 인식 모델을 만들어서 반환해주는 클래스

    모델 1 : 전체 분류 모델
    모델 2 : 악상기호 대분류 모델
    모델 3 : 악상기호 대분류 + 상세분류 모델
    모델 4 : 악상기호 대분류 모델 for YOLO
    """
    #

    # 모델 2 : CNN
    def model_2_CNN(self, input_shape=(512, 192, 1), output_units=12):
        """
        악상기호의 대분류(카테고리 분류)를 위해 설계한 모델.
        에포크를 많이 잡아서 학습을 하여도 정확도가 12%를 웃돌고 손실이 내려갈 기미가 안보여 콜백함수에 의해 중간에 학습이 멈춰짐.
        """
        # 모델 입력
        input = tf.keras.layers.Input(shape=input_shape)

        # 첫 번째 합성곱 블록
        layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=32, strides=1, padding='same', activation='relu')(input)
        layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(layer_1)
        flatten_1 = tf.keras.layers.Flatten()(layer_1)
        dense_1 = tf.keras.layers.Dense(32, activation='relu')(flatten_1)

        # 두 번째 합성곱층 블록
        layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=16, strides=1, padding='same', activation='relu')(layer_1)
        layer_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(layer_2)
        flatten_2 = tf.keras.layers.Flatten()(layer_2)
        dense_2 = tf.keras.layers.Dense(64, activation='relu')(flatten_2)

        # 세 번째 합성곱층 블록
        layer_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(layer_2)
        layer_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(layer_3)
        flatten_3 = tf.keras.layers.Flatten()(layer_3)
        dense_3 = tf.keras.layers.Dense(128, activation='relu')(flatten_3)

        # 각 출력들 연결
        concatenate = tf.keras.layers.Concatenate()([dense_1, dense_2, dense_3])

        # 드롭아웃
        dropout = tf.keras.layers.Dropout(0.2)(concatenate)

        # 출력층
        output = tf.keras.layers.Dense(output_units, activation='sigmoid')(dropout)
        
        # 모델 생성
        model = tf.keras.Model(inputs=[input], outputs=[output], name='MusicalSymbolRecognitionModel_2_CNN')

        # 모델 반환
        return model
    
    # 모델 2 : CNN + RNN
    def model_2_CRNN(self, input_shape=(512, 192, 1), num_classes=12):
        """
        악상기호들이 등장하는 패턴이 의존적인 규칙이 있고,
        이러한 모습이 한글 문자 인식(조합된 문자 인식)과 비슷하다는 생각이 들어 알아보니 찾은 모델.

        CNN 층으로 다양한 패턴을 학습하고, RNN 층으로 이 패턴들의 상관관계를 분석하는 것이 핵심.
        RNN 층이 악상기호의 의존적으로 그려지는 패턴을 캐치할 것 같다는 생각을 하였고 이에 예제를 그대로 긁어다 GPT한테 부탁해서 만든 모델.
        """
        # 모델 입력
        input = tf.keras.layers.Input(shape=input_shape)

        # Conv1_1 * MaxPooling
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # (256, 96, 64)

        # Conv2_1 * MaxPooling
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # (128, 48, 128)

        # Conv3_1 * Conv3_2 * MaxPooling
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # (64, 24, 256)

        # Conv4_1 * BatchNormalization
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)  # (64, 24, 512)

        # Conv5_1 * BatchNormalization * MaxPooling
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # (32, 12, 512)

        # Conv6_1
        x = tf.keras.layers.Conv2D(512, (2, 2), padding='same', activation='relu')(x)  # (32, 12, 512)
        
        # Reshape
        x = tf.keras.layers.Reshape(target_shape=(32, 12 * 512))(x)  # (32, 12 * 512)

        # Bidirectional LSTM * Bidirectional LSTM
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)  # (32, 256)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)  # (32, 256)

        # TimeDistributed(Dense) - Dense 레이어에서 각 타임 스텝마다 출력을 생성
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='sigmoid'))(x)  # (32, 12)

        # GlobalAveragePooling1D를 사용해 시퀀스의 차원을 없애기
        output = tf.keras.layers.GlobalAveragePooling1D()(x)  # (12)

        # 모델 생성
        model = tf.keras.Model(inputs=[input], outputs=[output], name='MusicalSymbolRecognitionModel_2_CRNN')

        # 모델 반환
        return model
    
    # 모델 2 : CNN + MHSA
    def model_2_CNN_MHA(self, input_shape=(512, 192, 1), num_classes=12):
        """
        model_2_CRNN 에서 특징들의 상관관계를 분석하는 RNN 층의 성능 효과를 보고,
        연관관계 분석에 탁월한 성능을 보이는 Self-Attention을 이용하면 좋을 것 같다는 생각을 하게 됨.
        다양한 연관관계를 찾을 수 있도록 어텐션을 Multi-Head로 구성하는 층을 이용함.
        특히 특성맵 사이의 분석에 용이하도록 합성곱층의 출력 텐서의 차원을 재정렬함.

        성능은 CRNN보다 매우 좋은 점수를 보여주고 있지만, 과적합이 조금 심하게 적용되고 있는 모습이 보여지고 있음.
        validation 데이터셋을 지금보다 많이 준비를 하고 과대적합을 방지하는 코드를 구성하는게 좋을 것 으로 보임.
        """
        # 모델 입력
        input = tf.keras.layers.Input(shape=input_shape)

        # Conv1_1 * MaxPooling
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # (256, 96, 64)

        # Conv2_1 * MaxPooling
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # (128, 48, 128)

        # Conv3_1 * Conv3_2 * MaxPooling
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # (64, 24, 256)

        # Conv4_1 * BatchNormalization
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)  # (64, 24, 512)

        # Conv5_1 * BatchNormalization * MaxPooling
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # (32, 12, 512)

        # Conv6_1
        x = tf.keras.layers.Conv2D(512, (2, 2), padding='same', activation='relu')(x)  # (32, 12, 512)
        
        # (중요)Permute(transpose)를 사용하여 특성맵끼리의 관계를 파악하기 위한 차원 재정렬
        x = tf.keras.layers.Permute(dims=(3, 1, 2))(x) # (512, 32, 12)

        # (중요)Reshape
        x = tf.keras.layers.Reshape(target_shape=(512, 12 * 32))(x)  # (512, 12 * 32)

        # Multi-Head Self-Attention
        x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(query=x, value=x, key=x)

        # dense1_1
        x = tf.keras.layers.Dense(256, activation='relu')(x)

        # GlobalAveragePooling1D를 사용해 시퀀스의 차원을 없애기
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # dense2_1
        output = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

        # 모델 생성
        model = tf.keras.Model(inputs=[input], outputs=[output], name='MusicalSymbolRecognitionModel_2_CNN_MHA')

        # 모델 반환
        return model

    # 모델 2
    def model_2(self, input_shape=(512, 192, 1), num_classes=12):
        """
        실제로 사용할 모델
        """
        # 모델 입력
        input = tf.keras.layers.Input(shape=input_shape)

        # Conv1_1 * MaxPooling2D * Conv1_2 * MaxPooling2D * BatchNormalization
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input)    # (512, 192, 64)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)                               # (256, 96, 64)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)       # (256, 96, 128)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)                               # (128, 48, 128)
        x = tf.keras.layers.BatchNormalization()(x)                                         

        # inception
        x1 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x2 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x2)
        x3 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x3 = tf.keras.layers.Conv2D(128, 5, padding='same', activation='relu')(x3)
        x4 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x4 = tf.keras.layers.Conv2D(128, 7, padding='same', activation='relu')(x4)
        x5 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x5 = tf.keras.layers.Conv2D(128, 9, padding='same', activation='relu')(x5)
        x6 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
        x6 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x6)
        x = tf.keras.layers.Concatenate(axis=-1)([x1, x2, x3, x4, x5, x6])                  # (128, 48, 640)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)                                    # (64, 24, 640)                               

        # Conv2_1 * Conv2_2 * BatchNormalization * MaxPooling2D
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)       # (64, 24, 256)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)       # (64, 24, 256)
        x = tf.keras.layers.BatchNormalization()(x) 
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)                               # (32, 12, 256)

        # Permute * Reshape * Multi-Head-Self-Attention
        x = tf.keras.layers.Permute(dims=(3, 1, 2))(x)                                              # (256, 32, 12)
        x = tf.keras.layers.Reshape(target_shape=(256, 32 * 12))(x)                                 # (256, 32 * 12)
        x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(query=x, value=x, key=x)    # (256, 32 * 12)
        #x = tf.keras.layers.Reshape(target_shape=(256, 32, 12))(x)                                  # (256, 32, 12)
        #x = tf.keras.layers.Permute(dims=(2, 3, 1))(x)                                              # (256, 32, 12)

        # dense1_1
        x = tf.keras.layers.Dense(256, activation='relu')(x)                    # (256, 256)

        # GlobalAveragePooling1D를 사용해 시퀀스의 차원을 없애기
        x = tf.keras.layers.GlobalAveragePooling1D()(x)                         # (256,)

        # dense2_1
        output = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)    # (12,)

        # 모델 생성
        model = tf.keras.Model(inputs=[input], outputs=[output], name='MusicalSymbolRecognitionModel_2')

        # 모델 반환
        return model

    # 모델 4 : Object Detection with CNN + MHSA
    def model_OD(self, input_shape=(512, 192, 1), num_classes=10):
        """
        기존의 합성곱층과 멀티 헤드 셀프 어텐션 층을 이용한 다중 분류 모델 구조에서,
        YOLO모델의 객체 위치 추정 방식을 결합하여 Object Detection 모델을 새로 개발함.

        악상 기호 이미지들의 크기와 비율 그리고 악보의 크기들이 보통 비슷한 점을 착안하여 이미지의 
        크기를 다르게 입력받는 YOLO모델과는 다르게 기존의 방식 그대로 고정 크기로 잘라 입력받아
        객체를 탐지하는 방향으로 설계하기로 함.

        객체(악상기호)에 대한 위치 추정은 공간적 특징을 잘 잡아낼 수 있도록 Dense 층을 사용하지 않았고, 
        해당 객체(악상기호)의 존재 여부, 즉 다중 분류는 합성곱, 멀티 헤드 셀프 어텐션, 완전 연결 층을 이용하도록 설계함.
        """
        # 모델 입력
        input = tf.keras.layers.Input(shape=input_shape)

        # Conv1_1 * MaxPooling2D * Conv1_2 * MaxPooling2D * BatchNormalization
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input)    # (512, 192, 64)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)                               # (256, 96, 64)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)       # (256, 96, 128)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)                               # (128, 48, 128)
        x = tf.keras.layers.BatchNormalization()(x)                                         

        # inception
        x1 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x2 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x2)
        x3 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x3 = tf.keras.layers.Conv2D(128, 5, padding='same', activation='relu')(x3)
        x4 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x4 = tf.keras.layers.Conv2D(128, 7, padding='same', activation='relu')(x4)
        x5 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x)
        x5 = tf.keras.layers.Conv2D(128, 9, padding='same', activation='relu')(x5)
        x6 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
        x6 = tf.keras.layers.Conv2D(64,  1, padding='same', activation='relu')(x6)
        x = tf.keras.layers.Concatenate(axis=-1)([x1, x2, x3, x4, x5, x6])                  # (128, 48, 640)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)                                    # (64, 24, 640)                               

        # Conv2_1 * Conv2_2 * BatchNormalization * MaxPooling2D
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)       # (64, 24, 256)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)       # (64, 24, 256)
        x = tf.keras.layers.BatchNormalization()(x) 
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)                               # (32, 12, 256)

        # Permute * Reshape * Multi-Head-Self-Attention * Reshape * Permute
        x = tf.keras.layers.Permute(dims=(3, 1, 2))(x)                                              # (256, 32, 12)
        x = tf.keras.layers.Reshape(target_shape=(256, 32 * 12))(x)                                 # (256, 32 * 12)
        x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(query=x, value=x, key=x)    # (256, 32 * 12)

        # Object Positioning : Reshape * Permute
        x_op = tf.keras.layers.Reshape(target_shape=(256, 32, 12))(x)                               # (256, 32, 12)
        x_op = tf.keras.layers.Permute(dims=(2, 3, 1))(x_op)                                        # (32, 12, 256)

        # Object Positioning : Conv3_1 * GlobalAveragePooling2D
        x_op = tf.keras.layers.Conv2D(num_classes * 6, 1, padding='same', activation=None)(x_op)    # (32, 12, num_classes * 6)
        x_op = tf.keras.layers.GlobalAveragePooling2D()(x_op)                                       # (1, num_classes * 6)

        # Object Multiclass Classification : dense1_1 * GlobalAveragePooling1D * dense2_1
        x_omc = tf.keras.layers.Dense(256, activation='relu')(x)                    # (256, 256)
        x_omc = tf.keras.layers.GlobalAveragePooling1D()(x_omc)                     # (256,)
        x_omc = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x_omc)     # (num_classes,)

        # [Object Positioning] 출력값이 [Object Multiclass Classification] 출력값에 의해 제어
        x_omc_expanded = tf.expand_dims(x_omc, axis=2)
        x_op_reshaped = tf.reshape(x_op, shape=(-1, num_classes, 6))
        x_combined = tf.reshape(x_omc_expanded * x_op_reshaped, shape=(-1, num_classes * 6))

        # 모델 생성
        model = tf.keras.Model(inputs=[input], outputs=[x_combined, x_omc], name='MSRM_ObjectDetection')

        # 모델 반환
        return model