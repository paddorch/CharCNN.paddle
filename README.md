# CharCNN.paddle
A PaddlePaddle implementation of CharCNN.

## 1. Introduction

![](images/model.png)

论文: [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626v3.pdf)

## 2. Results

|  Datasets          | Paper error rate| Our error rate  | constract |
|--------------------|-----------------|-----------------|--------|
| AG’s News          | 13.39           | 10.17           | +3.22  |
| Yahoo! Answers     | 28.80           |                 | +      |
| Amazon Review Full | 40.45           | 38.97           | +1.48  |

## 3. Dataset

![](images/datasets.png)

Format:
```
"class idx","sentence or text to be classified"  
```

Samples are separated by newline.

Example:
```shell
"3","Fears for T N pension after talks, Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
"4","The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com)","SPACE.com - TORONTO, Canada -- A second\team of rocketeers competing for the  #36;10 million Ansari X Prize, a contest for\privately funded suborbital space flight, has officially announced the first\launch date for its manned rocket."
```

## 4. Requirement

- Python >= 3
- PaddlePaddle >= 2.0.0
- see `requirements.txt`

## 5. Usage

### Train
1. 下载数据集到 `/data` 文件夹，并将训练集划分为 `train` 和 `dev`集：
```shell
bash split_data.sh data/ag_news/train.csv
```

2. start train
```shell
bash train_ag_news.sh
```

### Download Trained model

- [Yahoo! Answers]()
- [Amazon Review Full]()

> 将模型分别放置于 `output/models_yahoo_answers/` 和 `output/models_amz_full` 目录下，如下运行 `eval` bash 脚本即可测试模型。

### Test
```shell
bash eval_ag_news.sh
bash eval_yahoo_answers.sh
bash eval_amz_full.sh
```

[comment]: <> (## 六、代码结构与详细说明)

[comment]: <> (需要用一小节描述整个项目的代码结构，用一小节描述项目的参数说明，之后各个小节详细的描述每个功能的使用说明；)

[comment]: <> (## 七、模型信息)

[comment]: <> (以表格的信息，给出模型相关的信息)

## Implementation Details
### Data Augumentation
We use [nlpaug](https://github.com/makcedward/nlpaug) to augment data, specifically, we substitute similar word according to `WordNet`.

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
