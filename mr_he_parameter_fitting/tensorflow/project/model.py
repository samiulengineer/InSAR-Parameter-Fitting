import os
import math
from numpy import dtype
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from einops import rearrange
import segmentation_models as sm
from tensorflow.keras.models import Model
import keras_unet_collection.models as kuc
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU, add, Conv2D, PReLU, ReLU, Concatenate, Activation, MaxPool2D, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

class NewCnn(Model):

    def __init__(self,
                 in_channels=1,
                 lr=1e-3,
                 loss_type='mse',
                 *args,
                 **kwargs):
        super().__init__()
        self.lr = lr
        self.in_channels = in_channels
        self.loss_type = loss_type
        kernel_size = 3
        padding = 1

        self.conv1 = Conv2D(64, kernel_size=kernel_size, padding='same')
        self.conv2 = Conv2D(64, kernel_size=kernel_size, padding='same')
        self.conv3 = Conv2D(64, kernel_size=kernel_size, padding='same')
        self.conv4 = Conv2D(64, kernel_size=kernel_size, padding='same')
        self.conv5 = Conv2D(64, kernel_size=kernel_size, padding='same')
        self.conv6 = Conv2D(64, kernel_size=kernel_size, padding='same')
        self.conv7 = Conv2D(64, kernel_size=kernel_size, padding='same')
        self.conv8 = Conv2D(2,  kernel_size=kernel_size, padding='same')

        self.cnn = Conv2D(1, kernel_size=3, padding='same')

        self.bn1 = BatchNormalization(64)
        self.bn2 = BatchNormalization(64) 
        self.bn3 = BatchNormalization(64)
        self.bn4 = BatchNormalization(64)
        self.bn5 = BatchNormalization(64)
        self.bn6 = BatchNormalization(64)

    def call(self, filt_ifg_phase, coh, ddays, bperps):  # forward propagation
        # def forward(self, ddays, bperps):  # forward propagation
        ''' filt_ifg_phase and coh model design '''

        [B, H, W, N] = filt_ifg_phase.shape

        filt_ifg_phase = ReLU(self.conv1(filt_ifg_phase))
        filt_ifg_phase = ReLU(self.bn1(self.conv2(filt_ifg_phase)))
        filt_ifg_phase = ReLU(self.bn2(self.conv3(filt_ifg_phase)))
        filt_ifg_phase = ReLU(self.bn3(self.conv4(filt_ifg_phase)))
        filt_ifg_phase = ReLU(self.bn4(self.conv5(filt_ifg_phase)))
        filt_ifg_phase = ReLU(self.bn5(self.conv6(filt_ifg_phase)))
        filt_ifg_phase = ReLU(self.bn6(self.conv7(filt_ifg_phase)))
        filt_ifg_phase = self.conv8(filt_ifg_phase)

        coh = ReLU(self.conv1(coh))
        coh = ReLU(self.bn1(self.conv2(coh)))
        coh = ReLU(self.bn2(self.conv3(coh)))
        coh = ReLU(self.bn3(self.conv4(coh)))
        coh = ReLU(self.bn4(self.conv5(coh)))
        coh = ReLU(self.bn5(self.conv6(coh)))
        coh = ReLU(self.bn6(self.conv7(coh)))
        coh = self.conv8(coh)

        concat_phase_coh = ReLU(concatenate([filt_ifg_phase, coh], axis=1))

        cnn_concat_phase_coh = ReLU(self.cnn(concat_phase_coh))

        ddays = tf.reshape(ddays, [B, 1, 1, N]) * tf.ones([B, H, W, N])
        bperps = tf.reshape(bperps, [B, 1, 1, N]) * tf.ones([B, H, W, N])

        ddays = ReLU(self.conv1(ddays))
        ddays = ReLU(self.bn1(self.conv2(ddays)))
        ddays = ReLU(self.bn2(self.conv3(ddays)))
        ddays = ReLU(self.bn3(self.conv4(ddays)))
        ddays = ReLU(self.bn4(self.conv5(ddays)))
        ddays = ReLU(self.bn5(self.conv6(ddays)))
        ddays = ReLU(self.bn6(self.conv7(ddays)))
        ddays = self.conv8(ddays)

        bperps = ReLU(self.conv1(bperps))
        bperps = ReLU(self.bn1(self.conv2(bperps)))
        bperps = ReLU(self.bn2(self.conv3(bperps)))
        bperps = ReLU(self.bn3(self.conv4(bperps)))
        bperps = ReLU(self.bn4(self.conv5(bperps)))
        bperps = ReLU(self.bn5(self.conv6(bperps)))
        bperps = ReLU(self.bn6(self.conv7(bperps)))
        bperps = self.conv8(bperps)

        concat_ddays_bperps = ReLU(concatenate((ddays, bperps), axis=1))

        cnn_concat_ddays_bperps = ReLU(self.cnn(concat_ddays_bperps))

        concat_all = ReLU(
            concatenate((cnn_concat_phase_coh, cnn_concat_ddays_bperps), axis=1))

        return concat_all


class RBDN(Model):
    def __init__(self,
                 height,
                 width,
                 in_channels=1,
                 lr=1e-3,
                 loss_type='mse',
                 *args,
                 **kwargs):
        super().__init__()
        self.height = height
        self.width = width
        self.lr = lr
        self.in_channels = in_channels
        self.loss_type = loss_type
        kernel_size = 3
        padding = 1

        self.conv_input = Conv2D(64, kernel_size=9, padding="same")

        self.conv_middle = Conv2D(64, kernel_size=3, padding="same")

        self.conv_concat = Conv2D(64, kernel_size=3, padding="same")

        self.pooling_Layer = MaxPool2D((2,2))

        self.unpool_layer = UpSampling2D((2,2))

        self.deconv_layer = Conv2DTranspose(64, kernel_size=3, padding="same")

        self.deconv_output = Conv2DTranspose(1, kernel_size=9, padding="same")

        self.bn_layer = BatchNormalization(axis=-1, epsilon=1e-3)

    def call(self, inputs):  # forward propagation
        # def forward(self, ddays, bperps):  # forward propagation
        ''' filt_ifg_phase and coh model design '''

        # concat_all = ReLU(
        #     concatenate((filt_ifg_phase, coh, ddays, bperps), axis=1))

        # Model data pass start

        conv1 = Activation("relu")(self.bn_layer(self.conv_input(inputs)))

        pool1 = self.pooling_Layer(conv1)
        pool1 = Activation("relu")(pool1)

        convB11 = Activation("relu")(self.bn_layer(self.conv_middle(pool1)))

        poolB1 = self.pooling_Layer(convB11)
        poolB1 = Activation("relu")(poolB1)

        convB21 = Activation("relu")(self.bn_layer(self.conv_middle(poolB1)))

        poolB2 = self.pooling_Layer(convB21)
        poolB2 = Activation("relu")(poolB2)

        convB31 = Activation("relu")(self.bn_layer(self.conv_middle(poolB2)))

        poolB3 = self.pooling_Layer(convB31)
        poolB3 = Activation("relu")(poolB3)

        convB32 = Activation("relu")(self.bn_layer(self.conv_middle(poolB3)))

        unpoolB3 = Activation("relu")(self.unpool_layer(convB32))

        deconvB31 = Activation("relu")(self.bn_layer(self.deconv_layer(unpoolB3)))

        concat_poolB2_deconv31 = concatenate((poolB2, deconvB31), axis=-1)

        convB22 = Activation("relu")(self.bn_layer(
            self.conv_concat(concat_poolB2_deconv31)))

        unpoolB2 = Activation("relu")(self.unpool_layer(convB22))

        deconvB21 = Activation("relu")(self.bn_layer(self.deconv_layer(unpoolB2)))

        concat_poolB1_deconvb21 = concatenate(
            (poolB1, deconvB21), axis=-1)

        convB12 = Activation("relu")(self.bn_layer(
            self.conv_concat(concat_poolB1_deconvb21)))

        unpoolB1 = Activation("relu")(self.unpool_layer(convB12))

        deconvB11 = Activation("relu")(self.bn_layer(self.deconv_layer(unpoolB1)))

        concat_pool1_deconvb11 = concatenate((pool1, deconvB11), axis=-1)

        conv21 = Activation("relu")(self.bn_layer(
            self.conv_concat(concat_pool1_deconvb11)))

        conv22 = Activation("relu")(self.bn_layer(self.conv_middle(conv21)))
        conv23 = Activation("relu")(self.bn_layer(self.conv_middle(conv22)))
        conv24 = Activation("relu")(self.bn_layer(self.conv_middle(conv23)))
        conv25 = Activation("relu")(self.bn_layer(self.conv_middle(conv24)))
        conv26 = Activation("relu")(self.bn_layer(self.conv_middle(conv25)))
        conv27 = Activation("relu")(self.bn_layer(self.conv_middle(conv26)))
        conv28 = Activation("relu")(self.bn_layer(self.conv_middle(conv27)))
        conv29 = Activation("relu")(self.bn_layer(self.conv_middle(conv28)))

        unpool1 = Activation("relu")(self.unpool_layer(conv29))

        deconv1 = Activation("relu")(self.deconv_output(unpool1))

        return deconv1
    
    def summary(self):

        input = Input(shape=(self.height, self.width, self.in_channels))
        model =  Model(inputs=input, outputs=self.call(input))

        return model.summary()