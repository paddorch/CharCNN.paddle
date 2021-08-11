# CharCNN.paddle
A PaddlePaddle implementation of CharCNN.

## 一、简介
简单的介绍模型，以及模型的主要架构或主要功能，如果能给出效果图，可以在简介的下方直接贴上图片，展示模型效果，然后另起一行，按如下格式给出论文及链接。

论文:title

## 二、复现精度

|  数据集            | 论文 error rate | 复现 error rate | 对比  |
|--------------------|-----------------|-----------------|-------|
| AG’s News          | 13.39           | 10.17           | +3.22 |
| Yahoo! Answers     | 28.80           |                 | +     |
| Amazon Review Full | 40.45           |                 | +     |

## 三、数据集

![](images/datasets.png)

## 四、环境依赖

- 硬件：建议用 GPU
- 框架：
    - PaddlePaddle >= 2.0.0
    - 详见 `requirements.txt`

## 五、快速开始

### Train
1. 下载数据集到 `/data` 文件夹，并将训练集划分为 `train` 和 `dev`集：
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

### 模型下载链接

- [Yahoo! Answers]()
- [Amazon Review Full]()

> 将模型分别放置于 `output/models_yahoo_answers/` 和 `output/models_amz_full` 目录下，如上运行 `eval` bash 脚本即可测试模型。

[comment]: <> (## 六、代码结构与详细说明)

[comment]: <> (需要用一小节描述整个项目的代码结构，用一小节描述项目的参数说明，之后各个小节详细的描述每个功能的使用说明；)

[comment]: <> (## 七、模型信息)

[comment]: <> (以表格的信息，给出模型相关的信息)

## References
```bibtex
@article{zhang2015character,
  title={Character-level convolutional networks for text classification},
  author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  journal={Advances in neural information processing systems},
  volume={28},
  pages={649--657},
  year={2015}
}
```

- https://github.com/srviest/char-cnn-text-classification-pytorch
