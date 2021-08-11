# CharCNN.paddle
A PaddlePaddle implementation of CharCNN.

## 一、简介
简单的介绍模型，以及模型的主要架构或主要功能，如果能给出效果图，可以在简介的下方直接贴上图片，展示模型效果，然后另起一行，按如下格式给出论文及链接。

论文:title

## 二、复现精度

## 三、数据集
给出数据集的链接，然后按格式描述数据集大小与数据集格式即可。

格式如下：
- 数据集大小：关于数据集大小的描述，如类别，数量，图像大小等等；
- 数据格式：关于数据集格式的说明

## 四、环境依赖
主要分为两部分介绍，一部分是支持的硬件，另一部分是框架等环境的要求，格式如下：

- 硬件：
- 框架：
    - PaddlePaddle >= 2.0.0

## 五、快速开始
需要给出快速训练、预测、使用预训练模型预测的使用说明；

### Train
1. 准备数据集并划分`dev`集：
```shell
bash split_data.sh data/ag_news/train.csv
```

2. 训练
```shell
bash train_ag_news.sh
```

### Test
```shell
bash eval_ag_news.sh
bash eval_yahoo_answers.sh
bash eval_amz_full.sh
```

## 六、代码结构与详细说明
需要用一小节描述整个项目的代码结构，用一小节描述项目的参数说明，之后各个小节详细的描述每个功能的使用说明；

## 七、模型信息
以表格的信息，给出模型相关的信息