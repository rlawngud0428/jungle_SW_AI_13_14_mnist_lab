# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import *


class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        """
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        # TODO: params dict를 만들고 Affine/BatchNorm/ReLU/Dropout layer를 순서대로 구성하세요.
        # 권장 구조: 784 -> 512 -> 256 -> 10
        # self.layers는 OrderedDict로 만들고, self.grads는 params와 같은 key를 갖게 합니다.
        # raise NotImplementedError("NeuralNetwork.__init__을 구현하세요.")
        # 가중치 초기화 He 가중치 초기화
        self.hidden_size_list = [784, 512, 256, 10]
        self.hidden_layer_num = 3
        self.params = {}
        self.params["W1"] = np.random.randn(784, 512) * np.sqrt(2 / 784)
        self.params["b1"] = np.zeros(512)
        self.params["W2"] = np.random.randn(512, 256) * np.sqrt(2 / 512)
        self.params["b2"] = np.zeros(256)
        self.params["W3"] = np.random.randn(256, 10) * np.sqrt(2 / 256)
        self.params["b3"] = np.zeros(10)

        activation_layer = ReLU
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num):
            self.layers["Affine" + str(idx)] = Affine(
                self.params["W" + str(idx)], self.params["b" + str(idx)]
            )
            if True:
                self.params["gamma" + str(idx)] = np.ones(self.hidden_size_list[idx])
                self.params["beta" + str(idx)] = np.zeros(self.hidden_size_list[idx])
                self.layers["BatchNorm" + str(idx)] = BatchNorm(
                    self.params["gamma" + str(idx)], self.params["beta" + str(idx)]
                )

            self.layers["Activation_function" + str(idx)] = activation_layer()

            if True:
                self.layers["Dropout" + str(idx)] = Dropout(dropout_ratio)

        idx = self.hidden_layer_num
        self.layers["Affine" + str(idx)] = Affine(
            self.params["W" + str(idx)], self.params["b" + str(idx)]
        )

        self.last_layer = SoftmaxWithLoss()

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                x = layer.forward(x, train)
            else:
                x = layer.forward(x)
        return Softmax.forward(self, x)

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        self.grads = {}
        for name, layer in reversed(list(self.layers.items())):
            dout = layer.backward(dout)
            if isinstance(layer, Affine):
                idx = name[6:]
                self.grads["W" + idx] = layer.dW
                self.grads["b" + idx] = layer.db
            elif isinstance(layer, BatchNorm):
                idx = name[9:]
                self.grads["gamma" + idx] = layer.dgamma
                self.grads["beta" + idx] = layer.dbeta
        return dout

    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)
