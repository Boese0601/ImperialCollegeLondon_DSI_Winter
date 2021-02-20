from __future__ import absolute_import
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake
from keras_unet_collection.losses import focal_tversky,dice
from keras_unet_collection._backbone_zoo import backbone_zoo, bach_norm_checker
from keras_unet_collection._model_unet_2d import UNET_left, UNET_right

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import warnings
# TODO: another score function?
def dice_score(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def unet(input_size=(240, 240, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[dice_score])

    return model


# unet++
def unet_plus_plus(input_size=(240, 240, 3), base_filter_num=32):
    inputs = Input(input_size)
    conv0_0 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv0_0 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv0_0)

    conv1_0 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv1_0 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_0)

    up1_0 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_0)
    merge00_10 = concatenate([conv0_0, up1_0], axis=-1)
    conv0_1 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge00_10)
    conv0_1 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_1)

    conv2_0 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv2_0 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_0)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv2_0)

    up2_0 = Conv2DTranspose(base_filter_num * 2, (2, 2), strides=(2, 2), padding='same')(conv2_0)
    merge10_20 = concatenate([conv1_0, up2_0], axis=-1)
    conv1_1 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge10_20)
    conv1_1 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_1)

    up1_1 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_1)
    merge01_11 = concatenate([conv0_0, conv0_1, up1_1], axis=-1)
    conv0_2 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge01_11)
    conv0_2 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_2)

    conv3_0 = Conv2D(base_filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv3_0 = Conv2D(base_filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_0)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3_0)

    up3_0 = Conv2DTranspose(base_filter_num * 4, (2, 2), strides=(2, 2), padding='same')(conv3_0)
    merge20_30 = concatenate([conv2_0, up3_0], axis=-1)
    conv2_1 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge20_30)
    conv2_1 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_1)

    up2_1 = Conv2DTranspose(base_filter_num * 2, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    merge11_21 = concatenate([conv1_0, conv1_1, up2_1], axis=-1)
    conv1_2 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge11_21)
    conv1_2 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_2)

    up1_2 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_2)
    merge02_12 = concatenate([conv0_0, conv0_1, conv0_2, up1_2], axis=-1)
    conv0_3 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge02_12)
    conv0_3 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_3)

    conv4_0 = Conv2D(base_filter_num * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv4_0 = Conv2D(base_filter_num * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        conv4_0)

    up4_0 = Conv2DTranspose(base_filter_num * 8, (2, 2), strides=(2, 2), padding='same')(conv4_0)
    merge30_40 = concatenate([conv3_0, up4_0], axis=-1)
    conv3_1 = Conv2D(base_filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge30_40)
    conv3_1 = Conv2D(base_filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_1)

    up3_1 = Conv2DTranspose(base_filter_num * 4, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    merge21_31 = concatenate([conv2_0, conv2_1, up3_1], axis=-1)
    conv2_2 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge21_31)
    conv2_2 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_2)

    up2_2 = Conv2DTranspose(base_filter_num * 2, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    merge12_22 = concatenate([conv1_0, conv1_1, conv1_2, up2_2], axis=-1)
    conv1_3 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge12_22)
    conv1_3 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_3)

    up1_3 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_3)
    merge03_13 = concatenate([conv0_0, conv0_1, conv0_2, conv0_3, up1_3], axis=-1)
    conv0_4 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge03_13)
    conv0_4 = Conv2D(base_filter_num, 3, activation='relu', padg='same', kernel_initializer='he_normal')(conv0_4)

    conv0_4 = Conv2D(1, 1, activation='sigmoid')(conv0_4)

    model = Model(inputs=inputs, outputs=conv0_4)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[dice_score])

    return model



# unet+++
# Is GELU available here?
def unet_3plus_2d_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate,
                       stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True,
                       backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
                       name='unet3plus'):


    depth_ = len(filter_num_down)

    X_encoder = []
    X_decoder = []

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        # downsampling levels
        for i, f in enumerate(filter_num_down[1:]):
            # UNET-like downsampling
            X = UNET_left(X, f, kernel_size=3, stack_num=stack_num_down, activation=activation,
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
            X_encoder.append(X)

    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder)

        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_ - 1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder) + 1

        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor
            X = X_encoder[-1]

            # extra downsamplings
            for i in range(depth_ - depth_encode):
                i_real = i + depth_encode

                X = UNET_left(X, filter_num_down[i_real], stack_num=stack_num_down, activation=activation, pool=pool,
                              batch_norm=batch_norm, name='{}_down{}'.format(name, i_real + 1))
                X_encoder.append(X)

    # treat the last encoded tensor as the first decoded tensor
    X_decoder.append(X_encoder[-1])

    # upsampling levels
    X_encoder = X_encoder[::-1]

    depth_decode = len(X_encoder) - 1

    # loop over upsampling levels
    for i in range(depth_decode):

        f = filter_num_skip[i]

        # collecting tensors for layer fusion
        X_fscale = []

        # for each upsampling level, loop over all available downsampling levels (similar to the unet++)
        for lev in range(depth_decode):

            # counting scale difference between the current down- and upsampling levels
            pool_scale = lev - i - 1  # -1 for python indexing

            # deeper tensors are obtained from **decoder** outputs
            if pool_scale < 0:
                pool_size = 2 ** (-1 * pool_scale)

                X = decode_layer(X_decoder[lev], f, pool_size, unpool,
                                 activation=activation, batch_norm=batch_norm,
                                 name='{}_up_{}_en{}'.format(name, i, lev))

            # unet skip connection (identity mapping)
            elif pool_scale == 0:

                X = X_encoder[lev]

            # shallower tensors are obtained from **encoder** outputs
            else:
                pool_size = 2 ** (pool_scale)

                X = encode_layer(X_encoder[lev], f, pool_size, pool, activation=activation,
                                 batch_norm=batch_norm, name='{}_down_{}_en{}'.format(name, i, lev))

            # a conv layer after feature map scale change
            X = CONV_stack(X, f, kernel_size=3, stack_num=1,
                           activation=activation, batch_norm=batch_norm,
                           name='{}_down_from{}_to{}'.format(name, i, lev))

            X_fscale.append(X)

            # layer fusion at the end of each level
        # stacked conv layers after concat. BatchNormalization is fixed to True

        X = concatenate(X_fscale, axis=-1, name='{}_concat_{}'.format(name, i))
        X = CONV_stack(X, filter_num_aggregate, kernel_size=3, stack_num=stack_num_up,
                       activation=activation, batch_norm=True, name='{}_fusion_conv_{}'.format(name, i))
        X_decoder.append(X)

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation
    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_aggregate, stack_num=stack_num_up, activation=activation,
                           unpool=unpool, batch_norm=batch_norm, concat=False,
                           name='{}_plain_up{}'.format(name, i_real))
            X_decoder.append(X)

    # return decoder outputs
    return X_decoder

def unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto',
                  stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                  batch_norm=False, pool=True, unpool=True, deep_supervision=False,
                  backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus'):
    depth_ = len(filter_num_down)

    verbose = False

    if filter_num_skip == 'auto':
        verbose = True
        filter_num_skip = [filter_num_down[0] for num in range(depth_ - 1)]

    if filter_num_aggregate == 'auto':
        verbose = True
        filter_num_aggregate = int(depth_ * filter_num_down[0])

    if verbose:
        print('Automated hyper-parameter determination is applied with the following details:\n----------')
        print('\tNumber of convolution filters after each full-scale skip connection: filter_num_skip = {}'.format(
            filter_num_skip))
        print('\tNumber of channels of full-scale aggregated feature maps: filter_num_aggregate = {}'.format(
            filter_num_aggregate))

    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)

    X_encoder = []
    X_decoder = []

    IN = Input(input_size)

    X_decoder = unet_3plus_2d_base(IN, filter_num_down, filter_num_skip, filter_num_aggregate,
                                   stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation,
                                   batch_norm=batch_norm, pool=pool, unpool=unpool,
                                   backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
                                   freeze_batch_norm=freeze_batch_norm, name=name)
    X_decoder = X_decoder[::-1]

    if deep_supervision:

        # ----- frozen backbone issue checker ----- #
        if ('{}_backbone_'.format(backbone) in X_decoder[0].name) and freeze_backbone:
            backbone_warn = '\n\nThe deepest UNET 3+ deep supervision branch ("sup0") directly connects to a frozen backbone.\nTesting your configurations on `keras_unet_collection.base.unet_plus_2d_base` is recommended.'
            warnings.warn(backbone_warn);
        # ----------------------------------------- #

        OUT_stack = []
        L_out = len(X_decoder)

        print(
            '----------\ndeep_supervision = True\nnames of output tensors are listed as follows (the last one is the final output):')

        # conv2d --> upsampling --> output activation.
        # index 0 is final output
        for i in range(1, L_out):

            pool_size = 2 ** (i)

            X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv_{}'.format(name, i - 1))(X_decoder[i])

            X = decode_layer(X, n_labels, pool_size, unpool,
                             activation=None, batch_norm=False, name='{}_output_sup{}'.format(name, i - 1))

            if output_activation:
                print('\t{}_output_sup{}_activation'.format(name, i - 1))

                if output_activation == 'Sigmoid':
                    X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i - 1))(X)
                else:
                    activation_func = eval(output_activation)
                    X = activation_func(name='{}_output_sup{}_activation'.format(name, i - 1))(X)
            else:
                if unpool is False:
                    print('\t{}_output_sup{}_trans_conv'.format(name, i - 1))
                else:
                    print('\t{}_output_sup{}_unpool'.format(name, i - 1))

            OUT_stack.append(X)

        X = CONV_output(X_decoder[0], n_labels, kernel_size=3,
                        activation=output_activation, name='{}_output_final'.format(name))
        OUT_stack.append(X)

        if output_activation:
            print('\t{}_output_final_activation'.format(name))
        else:
            print('\t{}_output_final'.format(name))

        model = Model([IN, ], OUT_stack)

    else:
        OUT = CONV_output(X_decoder[0], n_labels, kernel_size=3,
                          activation=activation, name='{}_output_final'.format(name))

        model = Model([IN, ], [OUT, ])

    return model

def unet_3p(input_size=(240, 240, 3)):
    inputs = Input(input_size)
    outputs = unet_3plus_2d(input_size,2,[32,64, 128, 256, 512],filter_num_skip = [32, 32, 32, 32],filter_num_aggregate = 160,stack_num_down = 2,stack_num_up = 1,deep_supervision=True,backbone='DenseNet201',weights=None
                            ,pool='max', unpool=False,batch_norm=True,freeze_batch_norm=False,freeze_backbone=False,activation='GELU')(inputs)
    outputs = Conv2D(1, 1, activation='sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[dice_score])
    return model
# deep supervision loss function