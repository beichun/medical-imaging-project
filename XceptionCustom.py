# modified from https://github.com/yanchummar/xception-keras/blob/master/xception_model.py
# and https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D


def XceptionCustom(input_shape,
                   activation_name='selu',
                   include_top=False,
                   pooling='avg'):
    """customized Xception architecture.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    # Returns
        A Keras model instance.
    """
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation(activation_name)(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_name)(x)

    residual = Conv2D(128, (1, 1), strides=(
        2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_name)(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 2 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(
        2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 3
    x = Activation(activation_name)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_name)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 3 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(
        2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 4
    x = Activation(activation_name)(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_name)(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Block 5 - 12
    for i in range(8):
        residual = x

        x = Activation(activation_name)(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation(activation_name)(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation(activation_name)(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(
        2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 13
    x = Activation(activation_name)(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_name)(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 13 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Block 14
    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_name)(x)

    # Block 14 part 2
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_name)(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    inputs = img_input

    # Create model
    model = Model(inputs, x, name='xception_custom')

    return model
