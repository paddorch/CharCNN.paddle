#!/usr/bin/env python3
import paddle
import paddle.nn as nn


class CharCNN(paddle.nn.Layer):
    def __init__(self, num_features, num_classes, dropout):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1D(num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1D(kernel_size=3, stride=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1D(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1D(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1D(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1D(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1D(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1D(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc3 = nn.Linear(1024, num_classes)
        self.log_softmax = nn.LogSoftmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.reshape([x.shape[0], -1])
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        x = self.log_softmax(x)

        return x


def _test():
    model = CharCNN(1, 1, 1)


if __name__ == "__main__":
    _test()
