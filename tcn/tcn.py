import inspect
from typing import List

from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, SpatialDropout2D
from tensorflow import nest
from tensorflow_addons.layers import WeightNormalization
from keras.utils.layer_utils import count_params


def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


class ResidualBlock(Layer):

    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs):
        """Defines the residual block for the WaveNet TCN
        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block; defines dilation in temporal dimension only, spatial is for now 1
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _build_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlock. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape
            spatial_kernel_sizes = 0

            for k in range(2):
                if len(self.res_output_shape) != 4:
                    # make a Conv1D layer
                    name = 'conv1D_{}'.format(k)
                    # name scope used to make sure weights get unique names
                    spatial_kernel_sizes += 1
                    with K.name_scope(name):
                        conv = Conv1D(
                            filters=self.nb_filters,
                            kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate,
                            padding=self.padding,
                            name=name,
                            kernel_initializer=self.kernel_initializer
                        )
                        if self.use_weight_norm:
                            # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
                            with K.name_scope('norm_{}'.format(k)):
                                conv = WeightNormalization(conv)
                        self._build_layer(conv)
                else:
                    # make a Conv2D layer by
                    # > ZeroPadding2D (to the start; only if padding 'causal')
                    # > Conv2D
                    this_input_shape = self.res_output_shape
                    kernel_size = self.res_output_shape[-2] if self.res_output_shape[-2] < self.kernel_size else self.kernel_size
                    spatial_kernel_sizes += kernel_size
                    if self.padding == 'causal':
                        with K.name_scope('zeropadding2D_{}'.format(k)):
                            zeropad = ZeroPadding2D(
                                padding=(((kernel_size-1)*self.dilation_rate, 0), (0, 0)))
                            self._build_layer(zeropad)
                    name = 'conv2D_{}'.format(k)
                    # name scope used to make sure weights get unique names
                    with K.name_scope(name):
                        conv = Conv2D(
                            filters=self.nb_filters,
                            kernel_size=kernel_size,
                            # dilation rate in spatial dim is always 1, dilation_rate defines time dim rate
                            dilation_rate=(self.dilation_rate, 1),
                            padding='valid',
                            name=name,
                            kernel_initializer=self.kernel_initializer
                        )
                        if self.use_weight_norm:
                            # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
                            with K.name_scope('norm_{}'.format(k)):
                                conv = WeightNormalization(conv)
                        self._build_layer(conv)

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._build_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._build_layer(LayerNormalization())
                    elif self.use_weight_norm:
                        pass  # done above.

                self._build_layer(Activation(self.activation))
                if len(self.res_output_shape) == 4:
                    self._build_layer(SpatialDropout2D(rate=self.dropout_rate))
                else:
                    self._build_layer(SpatialDropout1D(rate=self.dropout_rate))

            if len(input_shape) != 4:
                if self.nb_filters != input_shape[-1]:
                    # 1x1 conv to match the shapes (channel dimension).
                    name = 'matching_conv1D'
                    with K.name_scope(name):
                        # make and build this layer separately because it directly uses input_shape
                        self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                                       kernel_size=1,
                                                       padding='same',
                                                       name=name,
                                                       kernel_initializer=self.kernel_initializer)
                        self.shape_match_conv.build(input_shape)
                        self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)
                else:
                    self.shape_match_conv = None
            else:
                # 2D 1x1 conv to match the shapes (channel dimension).
                name = 'matching_conv2D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv2D(filters=self.nb_filters,
                                                   kernel_size=(1, spatial_kernel_sizes-1),  # (1, input_shape[-2]),
                                                   padding='valid',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)
                    self.shape_match_conv.build(input_shape)
                    self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self._build_layer(Activation(self.activation))
            self.final_activation = Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)
        x_skip = x
        x = inputs
        if self.shape_match_conv:
            x = self.shape_match_conv(x)
            self.layers_outputs.append(x)
        res_x = layers.add([x, x_skip])
        self.layers_outputs.append(res_x)

        res_act_x = self.final_activation(res_x)
        self.layers_outputs.append(res_act_x)
        res_act_x.set_shape(self.res_output_shape)
        x_skip.set_shape(self.res_output_shape)
        return [res_act_x, x_skip, self.shape_match_conv]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]


class TCN(Layer):
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers. Can be a list.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 **kwargs):

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.skip_connections = []
        self.layers = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')

        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations)

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        # initialize parent class
        super(TCN, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * self.nb_stacks * sum(self.dilations)

    def build(self, input_shape):

        # member to hold current output shape of the layer for building purposes
        self.build_output_shape = input_shape

        # list to hold all the member ResidualBlocks
        self.layers = []

        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.layers.append(ResidualBlock(dilation_rate=d,
                                                 nb_filters=res_block_filters,
                                                 kernel_size=self.kernel_size,
                                                 padding=self.padding,
                                                 activation=self.activation,
                                                 dropout_rate=self.dropout_rate,
                                                 use_batch_norm=self.use_batch_norm,
                                                 use_layer_norm=self.use_layer_norm,
                                                 use_weight_norm=self.use_weight_norm,
                                                 kernel_initializer=self.kernel_initializer,
                                                 name='residual_block_{}'.format(len(self.layers))))
                # build newest residual block
                self.layers[-1].build(self.build_output_shape)
                self.build_output_shape = self.layers[-1].res_output_shape

        self.output_slice_index = None
        if self.padding == 'same':
            time = self.build_output_shape.as_list()[1]
            if time is not None:  # if time dimension is defined. e.g. shape = (bs, 500, input_dim).
                self.output_slice_index = int(self.build_output_shape.as_list()[1] / 2)
            else:
                # It will known at call time. c.f. self.call.
                self.padding_same_and_time_dim_unknown = True
        else:
            self.output_slice_index = -1  # causal case.

        self.computed_output_shape = self.build_output_shape
        if len(self.build_output_shape) != 4:
            if not self.return_sequences:
                self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :])
                self.computed_output_shape = self.build_output_shape[:1]+self.build_output_shape[2:]
        else:
            if self.build_output_shape.as_list()[-2] == 1:
                # (also) remove to-1-reduced spatial dimension with slicer
                if not self.return_sequences:
                    self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, 0, :])
                    self.computed_output_shape = self.build_output_shape[:1]+self.build_output_shape[3:]
                else:
                    self.slicer_layer = Lambda(lambda tt: tt[:, :, 0, :])
                    self.computed_output_shape = self.build_output_shape[:2]+self.build_output_shape[3:]
            elif not self.return_sequences:
                self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :, :])
                self.computed_output_shape = self.build_output_shape[:1]+self.build_output_shape[2:]

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        return self.computed_output_shape

    def call(self, inputs, training=None):
        x = inputs
        self.layers_outputs = [x]
        self.skip_connections = []
        match_convs = []
        for layer in self.layers:
            x, skip_out, match_conv = layer(x, training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)
            match_convs.append(match_conv)

        if self.use_skip_connections:
            if not any(match_convs):
                x = layers.add(self.skip_connections)
            else:
                x = self.skip_connections[0]
                for i, skip in enumerate(self.skip_connections[1:]):
                    if match_convs[i+1]:
                        x = match_convs[i+1](x)
                    x = layers.add([x, skip])
            self.layers_outputs.append(x)

        if self.slicer_layer:
            # case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        x.set_shape(self.compute_output_shape(inputs))
        return x

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config


def compiled_tcn(num_feat,  # type: int
                 num_classes,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 output_len=1,  # type: int
                 padding='causal',  # type: str
                 use_skip_connections=False,  # type: bool
                 return_sequences=True,
                 regression=False,  # type: bool
                 dropout_rate=0.05,  # type: float
                 name='tcn',  # type: str,
                 kernel_initializer='he_normal',  # type: str,
                 activation='relu',  # type:str,
                 opt='adam',
                 lr=0.002,
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False):
    # type: (...) -> Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).
    Classification uses a sparse categorical loss. Please input class ids and not one-hot encodings.

    Args:
        num_feat: The number of features of your input, i.e. the last dimension of: (batch_size, timesteps, input_dim).
        num_classes: The size of the final dense layer, how many classes we are predicting.
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        max_len: The maximum sequence length, use None if the sequence length is dynamic.
        padding: The padding to use in the convolutional layers.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        regression: Whether the output should be continuous or discrete.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        activation: The activation used in the residual blocks o = Activation(x + F(x)).
        name: Name of the model. Useful when having multiple TCN.
        kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        opt: Optimizer name.
        lr: Learning rate.
        use_batch_norm: Whether to use batch normalization in the residual layers or not.
        use_layer_norm: Whether to use layer normalization in the residual layers or not.
        use_weight_norm: Whether to use weight normalization in the residual layers or not.
    Returns:
        A compiled keras TCN.
    """

    dilations = adjust_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, kernel_initializer, use_batch_norm, use_layer_norm,
            use_weight_norm, name=name)(input_layer)

    def get_opt():
        if opt == 'adam':
            return optimizers.Adam(lr=lr, clipnorm=1.)
        elif opt == 'rmsprop':
            return optimizers.RMSprop(lr=lr, clipnorm=1.)
        else:
            raise Exception('Only Adam and RMSProp are available here')

    if not regression:
        # classification
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)

        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

        model.compile(get_opt(), loss='sparse_categorical_crossentropy', metrics=[accuracy])
    else:
        # regression
        x = Dense(output_len)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(get_opt(), loss='mean_squared_error')

    return model


def tcn_full_summary(model: Model, expand_residual_blocks=True):
    def append_table(table, layer, output_shape):
        table.append(
            (
                layer.name + "(" + layer.__class__.__name__ + ")",
                f"{output_shape}",
                f"{layer.count_params()}"
            ))
    table = [("Layer (type)", "Output Shape", "Param #")]
    for layer in model.layers:
        if isinstance(layer, TCN):
            for residual_block in layer.layers:
                if isinstance(residual_block, ResidualBlock):
                    if expand_residual_blocks:
                        for sublayer in residual_block.layers:
                            try:
                                output_shape = sublayer.output_shape
                            except AttributeError:
                                output_shape = 'multiple'
                            except RuntimeError:  # output_shape unknown in Eager mode.
                                output_shape = '?'
                            append_table(table, sublayer, output_shape)
                    else:
                        append_table(table, residual_block,
                                     residual_block.res_output_shape)
                else:
                    raise "Non ResidualBlock layer in TCN"
        else:
            append_table(table, layer, layer.output_shape)
    print(f"Model: \"{model.name}\"")
    lengths = [max(map(len, map(str, x))) for x in zip(*table)]
    lengths = [max(x) for x in zip(lengths, [28, 25, 10])]
    fmt = ' '.join('{:<%d}' % l for l in lengths)
    line_len = sum(lengths) + len(lengths) - 1
    print('_' * line_len)
    print(fmt.format(*table[0]))  # header
    print('=' * line_len)
    for i in range(1, len(table)):
        row = table[i]
        if i != 1:
            print('_' * line_len)
        print(fmt.format(*row))
    print('=' * line_len)
    print(f"Total params: {model.count_params()}")
    print(
        f"Trainable params: {count_params(model.trainable_weights)}")
    print(
        f"Non-trainable params: {count_params(model.non_trainable_variables)}")


def print_summary(model, line_length=None, positions=None, print_fn=None, expand_depth=0):
    """Prints a summary of a model.
    Args:
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
    """
    if print_fn is None:
        print_fn = print

    if model.__class__.__name__ == 'Sequential':
        sequential_like = True
    elif not model._is_graph_network:
        # We treat subclassed models as a simple sequence of layers, for logging
        # purposes.
        sequential_like = True
    else:
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (len(v) == 1 and
                                len(nest.flatten(v[0].keras_inputs)) > 1):
                # if the model has multiple nodes
                # or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break
            nodes += v
        if sequential_like:
            # search for shared layers
            for layer in model.layers:
                flag = False
                for node in layer._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break

    if sequential_like:
        line_length = line_length or 65
        positions = positions or [.45, .85, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape', 'Param #']
    else:
        line_length = line_length or 98
        positions = positions or [.33, .55, .67, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape',
                      'Param #', 'Connected to']
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print_fn(line)

    print_fn('Model: "{}"'.format(model.name))
    print_fn('_' * line_length)
    print_row(to_display, positions)
    print_fn('=' * line_length)

    def print_layer_summary(layer):
        """Prints a summary for a single layer.
        Args:
            layer: target layer.
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'multiple'
        except RuntimeError:  # output_shape unknown in Eager mode.
            output_shape = '?'
        name = layer.name
        cls_name = layer.__class__.__name__
        if not layer.built and not getattr(layer, '_is_graph_network', False):
            # If a subclassed model has a layer that is not called in Model.call, the
            # layer will not be built and we cannot call layer.count_params().
            print(f"Subclassed layer: {layer.name}")
            params = '0 (unused)'
        else:
            params = layer.count_params()
        fields = [name + ' (' + cls_name + ')', output_shape, params]
        print_row(fields, positions)

    def print_layer_summary_with_connections(layer):
        """Prints a summary for a single layer (including topological connections).
        Args:
            layer: target layer.
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'multiple'
        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue

            for inbound_layer, node_index, tensor_index, _ in node.iterate_inbound():
                connections.append('{}[{}][{}]'.format(inbound_layer.name, node_index,
                                                       tensor_index))

        name = layer.name
        cls_name = layer.__class__.__name__
        if not connections:
            first_connection = ''
        else:
            first_connection = connections[0]
        fields = [
            name + ' (' + cls_name + ')', output_shape,
            layer.count_params(), first_connection
        ]
        print_row(fields, positions)
        if len(connections) > 1:
            for i in range(1, len(connections)):
                fields = ['', '', '', connections[i]]
                print_row(fields, positions)

    def print_expanded_summary(layer, expand, is_last):
        if expand > 0 and hasattr(layer, "layers"):
            for i in range(len(layer.layers)):
                print_expanded_summary(
                    layer.layers[i], expand-1, is_last and i == len(layers)-1)
        else:
            if sequential_like:
                print_layer_summary(layer)
            else:
                print_layer_summary_with_connections(layer)
            if is_last:
                print_fn('=' * line_length)
            else:
                print_fn('_' * line_length)

    layers = model.layers
    for i in range(len(layers)):
        print_expanded_summary(layers[i], expand_depth, i == len(layers)-1)

    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)

    non_trainable_count = count_params(model.non_trainable_weights)

    print_fn('Total params: {:,}'.format(
        trainable_count + non_trainable_count))
    print_fn('Trainable params: {:,}'.format(trainable_count))
    print_fn('Non-trainable params: {:,}'.format(non_trainable_count))
    print_fn('_' * line_length)
