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