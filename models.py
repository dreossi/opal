import nn_utils

class tinyNN:

    ouput = None

    def __init__(self, x):

        num_classes = 2
        num_channels = 3

        filter_size_conv1 = 3
        num_filters_conv1 = 32

        filter_size_conv2 = 3
        num_filters_conv2 = 32

        filter_size_conv3 = 3
        num_filters_conv3 = 64

        fc_layer_size = 128

        layer_conv1 = nn_utils.create_convolutional_layer(input=x,
                       num_input_channels=num_channels,
                       conv_filter_size=filter_size_conv1,
                       num_filters=num_filters_conv1)

        layer_conv2 = nn_utils.create_convolutional_layer(input=layer_conv1,
                       num_input_channels=num_filters_conv1,
                       conv_filter_size=filter_size_conv2,
                       num_filters=num_filters_conv2)

        layer_conv3 = nn_utils.create_convolutional_layer(input=layer_conv2,
                       num_input_channels=num_filters_conv2,
                       conv_filter_size=filter_size_conv3,
                       num_filters=num_filters_conv3)

        layer_flat = nn_utils.create_flatten_layer(layer_conv3)

        layer_fc1 = nn_utils.create_fc_layer(input=layer_flat,
                             num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                             num_outputs=fc_layer_size,
                             use_relu=True)

        layer_fc2 = nn_utils.create_fc_layer(input=layer_fc1,
                             num_inputs=fc_layer_size,
                             num_outputs=num_classes,
                             use_relu=False)

        self.output = layer_fc2


class mediumNN:

    ouput = None

    def __init__(self, x):

        num_classes = 2
        num_channels = 3

        filter_size_conv1 = 3
        num_filters_conv1 = 32

        filter_size_conv2 = 3
        num_filters_conv2 = 32

        filter_size_conv3 = 3
        num_filters_conv3 = 32

        filter_size_conv4 = 3
        num_filters_conv4 = 64

        fc_layer_size = 128

        layer_conv1 = nn_utils.create_convolutional_layer(input=x,
                       num_input_channels=num_channels,
                       conv_filter_size=filter_size_conv1,
                       num_filters=num_filters_conv1)

        layer_conv2 = nn_utils.create_convolutional_layer(input=layer_conv1,
                       num_input_channels=num_filters_conv1,
                       conv_filter_size=filter_size_conv2,
                       num_filters=num_filters_conv2)

        layer_conv3 = nn_utils.create_convolutional_layer(input=layer_conv2,
                       num_input_channels=num_filters_conv2,
                       conv_filter_size=filter_size_conv3,
                       num_filters=num_filters_conv3)

        layer_conv4 = nn_utils.create_convolutional_layer(input=layer_conv3,
                       num_input_channels=num_filters_conv3,
                       conv_filter_size=filter_size_conv4,
                       num_filters=num_filters_conv4)

        layer_flat = nn_utils.create_flatten_layer(layer_conv4)

        layer_fc1 = nn_utils.create_fc_layer(input=layer_flat,
                             num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                             num_outputs=fc_layer_size,
                             use_relu=True)

        layer_fc2 = nn_utils.create_fc_layer(input=layer_fc1,
                             num_inputs=fc_layer_size,
                             num_outputs=num_classes,
                             use_relu=False)

        self.output = layer_fc2
