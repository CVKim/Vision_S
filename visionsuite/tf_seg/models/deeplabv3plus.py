import tensorflow as tf

from ._custom_layers_and_blocks import ConvolutionBnActivation, AtrousSeparableConvolutionBnReLU, AtrousSpatialPyramidPoolingV3

class DeepLabV3plus(tf.keras.Model):

    def __init__(
        self,
        n_classes,
        base_model,
        output_layers,
        height=None,
        width=None,
        channel=3,
        filters=64,
        final_activation="softmax",
        backbone_trainable=True,
        output_stride=16,
        dilations=[6, 12, 18],
        crl=False,
        channel_multipler=1,
        add_low_level=[0,1,2],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.output_stride = output_stride
        self.dilations = dilations
        self.height = height
        self.width = width
        self.channel = channel
        self.crl = crl
        self.add_low_level = add_low_level

        if "V2" not in base_model.name:
            ### for eff-v1
            self.low_level_index = 1
            if self.output_stride == 8:
                self.upsample2d_1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
                output_layers = output_layers[:3]
                self.dilations = [2 * rate for rate in dilations]
                assert max(self.add_low_level) == 1, ValueError(
                    "When output_stride is 8, the maximum of add_low_level is 1"
                )
            elif self.output_stride == 16:
                self.upsample2d_1 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
                output_layers = output_layers[:4]
                self.dilations = dilations
                assert max(self.add_low_level) == 2, ValueError(
                    "When output_stride is 16, the maximum of add_low_level is 2"
                )
            else:
                raise ValueError("'output_stride' must be one of (8, 16), got {}".format(self.output_stride))
        else:
            self.low_level_index = 0
            self.upsample2d_1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
            output_layers = output_layers[:2]
            assert max(self.add_low_level) == 1, ValueError(
                f"the maximum of add_low_level is 1, not {self.add_low_level}"
            )
            if self.output_stride == 8:
                self.dilations = [2 * rate for rate in dilations]
            elif self.output_stride == 16:
                self.dilations = dilations
            else:
                raise ValueError("'output_stride' must be one of (8, 16), got {}".format(self.output_stride))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers, name=base_model.name)

        # FPN conv.
        fpn_output_channel = self.backbone.get_layer(
            output_layers[self.low_level_index].name.split("/")[0]
        ).output.shape.as_list()[-1]
        self.fpn_conv_0 = tf.keras.layers.Conv2D(fpn_output_channel, (1, 1), strides=(1, 1), padding="same")
        self.fpn_conv_2 = tf.keras.layers.Conv2D(fpn_output_channel, (1, 1), strides=(1, 1), padding="same")

        # Define Layers
        self.atrous_sepconv_bn_relu_1 = AtrousSeparableConvolutionBnReLU(
            dilation=2,
            filters=filters,
            kernel_size=3,
            channel_multiplier=channel_multipler,
        )
        self.atrous_sepconv_bn_relu_2 = AtrousSeparableConvolutionBnReLU(
            dilation=2,
            filters=filters,
            kernel_size=3,
            channel_multiplier=channel_multipler,
        )
        self.aspp = AtrousSpatialPyramidPoolingV3(self.dilations, filters)

        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, 1)
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(64, 1)

        self.upsample2d_2 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")

        self.concat = tf.keras.layers.Concatenate(axis=3)

        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, 3)
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, 3)
        self.conv1x1_bn_sigmoid = ConvolutionBnActivation(self.n_classes, 1, post_activation="linear")

        self.final_activation = tf.keras.layers.Activation(final_activation, name="output")

    def call(self, inputs, training=None, mask=None):
        outputs = self.backbone(inputs)
        x = outputs[-1]

        low_level_features = outputs[self.low_level_index]
        for index in self.add_low_level:
            if index == self.low_level_index:
                continue

            if "V2" not in self.backbone.name:
                if index == 0:
                    feature = self.fpn_conv_0(outputs[index])
                elif index == 2:
                    feature = self.fpn_conv_2(outputs[index])
                else:
                    NotImplementedError(f"index({index}) for add_low_leve is not yet considered")
            else:
                feature = self.fpn_conv_0(outputs[index])
            feature = tf.image.resize(feature, tf.shape(outputs[self.low_level_index])[1:3], method="bilinear")
            low_level_features += feature

        # Encoder Module
        encoder = self.atrous_sepconv_bn_relu_1(x, training)
        encoder = self.aspp(encoder, training)
        encoder = self.conv1x1_bn_relu_1(encoder, training)
        encoder = self.upsample2d_1(encoder)

        # Decoder Module
        decoder_low_level_features = self.atrous_sepconv_bn_relu_2(low_level_features, training)
        decoder_low_level_features = self.conv1x1_bn_relu_2(decoder_low_level_features, training)

        decoder = self.concat([decoder_low_level_features, encoder])

        decoder = self.conv3x3_bn_relu_1(decoder, training)
        decoder = self.conv3x3_bn_relu_2(decoder, training)
        decoder = self.conv1x1_bn_sigmoid(decoder, training)

        decoder = self.upsample2d_2(decoder)
        decoder = self.final_activation(decoder)

        return decoder

    def model(self):

        if not self.crl:
            x = tf.keras.layers.Input(shape=(self.height, self.width, self.channel))
            return tf.keras.Model(inputs=[x], outputs=self.call(x))
        else:
            x = tf.keras.layers.Input(shape=(self.height, self.width, self.channel))
            input_threshold = tf.keras.Input((self.n_classes), name="threshold")
            branch_outputs = [0 for i in range(self.n_classes)]

            for c in range(self.n_classes):
                branch_outputs[c] = self.call(x)[:, :, :, c : c + 1]
                branch_outputs[c] = tf.where(branch_outputs[c] > input_threshold[0, c], 1.0, 0.0)

            _crl_output = tf.keras.layers.Concatenate(axis=-1)(branch_outputs)
            # output_threshold = tf.math.argmax(output_threshold, axis=-1, name="output2")
            argmax_layer = Argmax("output2")
            crl_output = argmax_layer(inputs=_crl_output, axis=-1)

            return tf.keras.Model(inputs=[x, input_threshold], outputs=crl_output)
