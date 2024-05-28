# 基于机器学习的文本分类

## 项目简介

本项目实现了一个基于Logistic回归的文本分类模型，使用n-gram作为特征表示。数据处理、特征提取、模型训练和预测均在Python中实现。本项目是针对[FuDanNLP的NLP-Beginner：自然语言处理入门练习](https://github.com/FudanNLP/nlp-beginner)的任务一：基于机器学习的文本分类。

## 文件结构

- `main.py`: 主程序文件，负责数据处理、模型训练和评估。
- `data_process.py`: 数据处理模块，包括数据加载、n-gram特征提取、训练集和验证集的划分。
- `model.py`: 定义了Logistic回归模型、softmax函数、交叉熵损失函数和梯度计算。
- `n_gram.py`: 实现n-gram词汇表的构建和文本转换为n-gram向量的功能。

## 环境依赖

- Python 3.x
- NumPy
- Pandas（主要用于数据读取的部分）
- scikit-learn（主要用于评估准确率）
- tqdm（可视化进度）

可以使用以下命令安装依赖：

```
bash
Copy code
pip install numpy pandas scikit-learn tqdm
```

## 使用方法

### 1. 数据准备

请确保数据文件`train.tsv`位于`data`文件夹中，数据格式如下：

```
pythonCopy codePhraseID    SentenceID   Phrase  Sentiment
1           1            A series...    1
2           1            series of...   2
...
```

### 2. 运行模型

在命令行中运行以下命令来训练模型并评估准确率：

```
python ./main.py
```

### 3. 代码说明

#### `main.py`

主程序文件，负责数据处理、模型训练和评估。

- `train()`: 处理数据、训练模型并评估准确率。
- `if __name__ == '__main__':`: 运行训练函数。

#### `data_process.py`

数据处理模块，包括数据加载、n-gram特征提取、训练集和验证集的划分。

- `process_data(file_path, n, sample_rate=0.1)`: 加载并处理数据。
- `train_test_split(X, y, test_size=0.2, random_state=None)`: 划分训练集和验证集。

#### `model.py`

定义了Logistic回归模型、softmax函数、交叉熵损失函数和梯度计算。

- `softmax(z)`: 计算softmax函数值。
- `categorical_cross_entropy(y_true, y_pred)`: 计算分类交叉熵损失函数。
- `compute_gradient(X, y_true, y_pred)`: 计算梯度。
- `LogisticRegression`类: 实现Logistic回归模型，包括`fit`和`predict`方法。

#### `n_gram.py`

实现n-gram词汇表的构建和文本转换为n-gram向量的功能。

- `build_ngram_vocab(data, n)`: 构建n-gram词汇表。
- `text_to_ngram_vector(text, ngram_vocab, n)`: 将文本转换为n-gram向量。

## 结果与评估

运行`main.py`会输出模型在验证集上的准确率。可以通过调整`LogisticRegression`中的`learning_rate`和`num_iterations`参数来优化模型性能。