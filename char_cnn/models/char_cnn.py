from collections import namedtuple

import paddle
import paddle.nn as nn


__all__ = ['CharCNN']


class CharCNN(nn.Layer):
    def __init__(self, input_dim: int, num_classes: int, dropout: float, max_length: int):
        super(CharCNN, self).__init__()

        self.dropout_fn = nn.Dropout2D(dropout)

        def make_conv(input_dim: int, output_dim: int, kernel_size: int, pool: bool) -> nn.Layer:
            if pool:
                return nn.Sequential(
                    nn.Conv1D(input_dim, output_dim, kernel_size=kernel_size),
                    nn.ReLU(),
                    nn.MaxPool1D(3)
                )
            else:
                return nn.Sequential(
                    nn.Conv1D(input_dim, output_dim, kernel_size=kernel_size),
                    nn.ReLU(),
                )

        LayerShape = namedtuple('LayerShape', ['input', 'output', 'kernel', 'pool'])
        layers = [
            LayerShape(input=input_dim, output=1024, kernel=7, pool=True),
            LayerShape(input=1024, output=1024, kernel=7, pool=True),
            *[LayerShape(input=1024, output=1024, kernel=3, pool=False) for _ in range(3)],
            LayerShape(input=1024, output=1024, kernel=3, pool=True),
        ]

        def build_layer(shape: LayerShape) -> nn.Layer:
            return make_conv(shape.input, shape.output, shape.kernel, shape.pool)

        layers = [build_layer(s) for s in layers]
        self.convs = nn.LayerList(layers)

        input_shape = [128, max_length, input_dim]
        conv_output_dim = self._conv_output_dim(input_shape)

        self.mlp = nn.Sequential(
            nn.Linear(conv_output_dim, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        self.apply(self.__init_weights)

    def __init_weights(self, layer: nn.Layer):
        if isinstance(layer, nn.Conv1D) or isinstance(layer, nn.Linear):
            new_weights = paddle.normal(0.0, 0.05, shape=[layer.weight.shape])
            layer.weight.set_value(new_weights)

    def __conv_output_dim(self, input_shape):
        x = paddle.randn(input_shape, dtype=paddle.float32)
        x = self.conv_encoder(x)
        return x.shape[1]

    def conv_encoder(self, x: paddle.Tensor) -> paddle.Tensor:
        for conv in self.convs:
            x = conv(x)
        return paddle.reshape(x, [x.shape[0], -1])

    def forward(self, x, **kwargs):
        x = self.dropout_fn(x)
        x = self.conv_encoder(x)
        x = self.mlp(x)
        return x
