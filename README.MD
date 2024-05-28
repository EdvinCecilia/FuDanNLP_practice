# 从零开始NLP

我个人的NLP初学者的练习，可以帮助我快速入门，本项目是针对[FuDanNLP的NLP-Beginner：自然语言处理入门练习](https://github.com/FudanNLP/nlp-beginner)。

## 实验任务

### 任务一：基于机器学习的文本分类

- **实现方式**：使用NumPy实现基于logistic/softmax regression的文本分类。
- **数据集**：Classify the sentiment of sentences from the Rotten Tomatoes dataset
- 参考资料《神经网络与深度学习》 第2/3章
- **已完成**：是
- **详情**：[exp1 README](./exp1/README.MD)

### 任务二：基于深度学习的文本分类

- **实现方式**：使用PyTorch，实现CNN、RNN的文本分类。
- 参考资料
  - [PyTorch官方网站](https://pytorch.org/)
  - Convolutional Neural Networks for Sentence Classification [论文链接](https://arxiv.org/abs/1408.5882)
- **待完成**：是

### 任务三：基于注意力机制的文本匹配

- **实现方式**：使用Pytorch，实现ESIM模型（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。
- 参考资料
  - 《神经网络与深度学习》 第7章
  - Reasoning about Entailment with Neural Attention [论文链接](https://arxiv.org/pdf/1509.06664v1.pdf)
- **待完成**：是

### 任务四：基于LSTM+CRF的序列标注

- **实现方式**：使用Pytorch，用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。
- 参考资料
  - 《神经网络与深度学习》 第6、11章
- **待完成**：是

### 任务五：基于神经网络的语言模型

- **实现方式**：使用Pytorch，用LSTM、GRU来训练字符级的语言模型，计算困惑度。
- 参考资料
  - 《神经网络与深度学习》 第6、15章
- **待完成**：是