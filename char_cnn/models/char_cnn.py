from collections import namedtuple

import paddle
import paddle.nn as nn


__all__ = ['CharCNN']


class CharCNN(nn.Layer):
    def __init__(self, max_length: int, vocab_size: int, num_classes: int, dropout: float = 0.5, unk_id: int = 0):
        super(CharCNN, self).__init__()
        self.vocab_size = vocab_size
        self.unk_id = unk_id

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
            LayerShape(input=vocab_size, output=1024, kernel=7, pool=True),
            LayerShape(input=1024, output=1024, kernel=7, pool=True),
            *[LayerShape(input=1024, output=1024, kernel=3, pool=False) for _ in range(3)],
            LayerShape(input=1024, output=1024, kernel=3, pool=True),
        ]

        def build_layer(shape: LayerShape) -> nn.Layer:
            return make_conv(shape.input, shape.output, shape.kernel, shape.pool)

        layers = [build_layer(s) for s in layers]
        self.convs = nn.LayerList(layers)

        input_shape = [128, vocab_size, max_length]  # 'NCL' format
        conv_output_dim = self.__conv_output_dim(input_shape)

        self.mlp = nn.Sequential(
            nn.Linear(conv_output_dim, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        self.apply(self.__init_weights)

    def __init_weights(self, layer: nn.Layer):
        if isinstance(layer, nn.Conv1D) or isinstance(layer, nn.Linear):
            new_weights = paddle.normal(0.0, 0.05, shape=layer.weight.shape)
            layer.weight.set_value(new_weights)

    def __conv_output_dim(self, input_shape):
        x = paddle.randn(input_shape, dtype=paddle.float32)
        x = self.conv_encoder(x)
        return x.shape[1]

    def conv_encoder(self, x: paddle.Tensor) -> paddle.Tensor:
        for conv in self.convs:
            orig_shape = x.shape
            x = conv(x)
            now_shape = x.shape
            # print(f'{orig_shape} === {conv} ===> {now_shape}')
        return paddle.reshape(x, [x.shape[0], -1])

    def quantize(self, ids: paddle.Tensor) -> paddle.Tensor:
        x = nn.functional.one_hot(ids, num_classes=self.vocab_size)
        x[:, self.unk_id] = 0.0
        return paddle.transpose(x, [0, 2, 1])

    def forward(self, ids, **kwargs):
        x = self.quantize(ids)
        # x = self.dropout_fn(x)
        x = self.conv_encoder(x)
        x = self.mlp(x)
        return x
