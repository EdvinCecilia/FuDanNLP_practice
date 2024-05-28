import numpy as np
import pandas as pd
from n_gram import build_ngram_vocab, text_to_ngram_vector
from tqdm import tqdm

# 数据处理部分

def process_data(file_path, n, sample_rate=0.1):
    """
    将指定文件中的数据处理为n-gram向量表示，并返回向量矩阵X、标签数组y和n-gram词汇表ngram_vocab。
    
    Args:
        file_path (str): 数据文件的路径。
        n (int): n-gram中n的值。
        sample_rate (float, optional): 数据采样率，默认为0.1。这个主要是针对n_gram无法处理大数据集的问题。
    
    Returns:
        tuple: 包含三个元素的元组，分别为：
            - X (np.ndarray): 形状为(len(data), len(ngram_vocab))的n-gram向量矩阵。
            - y (np.ndarray): 形状为(len(data), )的标签数组。
            - ngram_vocab (dict): n-gram词汇表，键为n-gram字符串，值为对应的索引。
    
    """
    # 读取数据文件
    df = pd.read_csv(file_path, sep='\t')
    df = df.sample(frac=sample_rate, random_state=42)
    
    # 解析成适当的格式
    data = [(row['Phrase'], row['Sentiment']) for _, row in df.iterrows()]
    
    # 构建n-gram词汇表
    ngram_vocab = build_ngram_vocab(data, n)
    
    # 初始化进度条
    progress_bar = tqdm(total=len(data), desc="Processing data", unit=" samples")
    
    # 将文本转换为n-gram向量
    X = np.empty((len(data), len(ngram_vocab)), dtype=np.int32)
    y = np.empty(len(data), dtype=np.int32)
    for i, (text, label) in enumerate(data):
        ngram_vector = text_to_ngram_vector(text, ngram_vocab, n)
        X[i] = ngram_vector
        y[i] = label
        # 更新进度条
        progress_bar.update(1)
    
    # 关闭进度条
    progress_bar.close()
    
    # X = np.array(X)
    # y = np.array(y)
    
    return X, y, ngram_vocab

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    将数据集划分为训练集和测试集。
    
    Args:
        X (np.ndarray): 特征矩阵。
        y (np.ndarray): 标签数组。
        test_size (float): 测试集所占比例。
        random_state (int): 随机种子。
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 训练集和测试集的特征和标签。
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 计算测试集的样本数
    num_samples = X.shape[0]
    num_test_samples = int(num_samples * test_size)
    
    # 生成随机的索引
    indices = np.random.permutation(num_samples)
    
    # 划分训练集和测试集
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    file_path = 'data/train.tsv'
    n = 2  # 设置n-gram的大小
    
    X, y, ngram_vocab = process_data(file_path, n)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("n-gram词汇表:", ngram_vocab)
    print("训练集n-gram向量:\n", X_train)
    print("训练集标签:\n", y_train)
    print("验证集n-gram向量:\n", X_val)
    print("验证集标签:\n", y_val)