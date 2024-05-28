# 任务一：基于机器学习的文本分类
# Author: Edvin Yang
# Time: 2024-05-22
# Description: This script builds an n-gram vocabulary and converts text to an n-gram vector representation.
import numpy as np
import string

def build_ngram_vocab(data, n):
    """
    构建n-gram词汇表。
    
    Args:
        data (List[Tuple[str, int]]): 原始数据集，每个元素为(文本, 标签)的元组。
        n (int): n-gram中的n值，即每个n-gram包含的单词数。
    
    Returns:
        Dict[Tuple[str, ...], int]: 构建的n-gram词汇表，键为n-gram元组，值为对应的索引值。
    
    """
    # 初始化n-gram词汇表
    ngram_vocab = {}
    # 初始化索引值
    index = 0
    
    # 遍历数据集
    for text, label in data:
        # 将文本转换为小写，并去除标点符号
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        # 按空格分割成单词列表
        tokens = text.split()
        
        # 遍历单词列表，构建n-gram
        for i in range(len(tokens) - n + 1):
            # 构建n-gram元组
            ngram = tuple(tokens[i:i+n])
            # 如果n-gram不在词汇表中
            if ngram not in ngram_vocab:
                # 将n-gram添加到词汇表中，并为其分配索引值
                ngram_vocab[ngram] = index
                # 索引值递增
                index += 1
    return ngram_vocab

def text_to_ngram_vector(text, ngram_vocab, n):
    """
    将文本转换为n-gram向量。
    
    Args:
        text (str): 需要转换的文本。
        ngram_vocab (dict): n-gram词汇表，键为n-gram元组，值为对应的索引。
        n (int): n-gram的阶数。
    
    Returns:
        np.ndarray: 转换后的n-gram向量，形状为(len(ngram_vocab),)。
    
    """
    # 初始化一个与ngram_vocab长度相同的零向量
    vector = np.zeros(len(ngram_vocab))
    # 将文本转换为小写并分割成单词列表
    tokens = text.lower().split()
    # 遍历tokens列表，从第一个单词开始到倒数第n个单词结束
    for i in range(len(tokens) - n + 1):
        # 取出当前位置开始的n个单词组成的ngram
        ngram = tuple(tokens[i:i+n])
        # 如果ngram在ngram_vocab中
        if ngram in ngram_vocab:
            # 将vector中对应ngram的索引位置的计数加1
            vector[ngram_vocab[ngram]] += 1
    return vector

# 以下为测试用例
if __name__ == '__main__':

    # 示例数据
    data = [
        ("I love this movie", 1),
        ("This film was terrible", 0),
        ("Amazing acting", 1),
        ("I hate this film", 0),
        ("Best movie ever", 1)
    ]

    n = 2  # 设置n-gram的大小
    ngram_vocab = build_ngram_vocab(data, n)
    print("n-gram词汇表:", ngram_vocab)

    # 处理所有文本
    X = np.array([text_to_ngram_vector(text, ngram_vocab, n) for text, label in data])
    y = np.array([label for text, label in data])

    print("n-gram向量:\n", X)
    print("标签:\n", y)

    # 添加偏置项
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    print("添加偏置项后的n-gram向量:\n", X)
