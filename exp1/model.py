import numpy as np

def softmax(z):
    """
    计算softmax函数值。
    
    Args:
        z (np.ndarray): 形状为 (N, D) 的二维数组，其中 N 为样本数量，D 为特征维度。
    
    Returns:
        np.ndarray: 形状为 (N, D) 的二维数组，表示每个样本在每个特征上的概率分布。
    
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def categorical_cross_entropy(y_true, y_pred):
    """
    计算分类交叉熵损失函数。
    
    Args:
        y_true (np.ndarray): 真实标签，形状为 (batch_size, num_classes)。
        y_pred (np.ndarray): 预测概率，形状为 (batch_size, num_classes)。
    
    Returns:
        float: 交叉熵损失值。
    
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) #防止出现loss为负数
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss


def compute_gradient(X, y_true, y_pred):
    """
    计算线性回归的梯度。
    
    Args:
        X (np.ndarray): 形状为 (n_features, m) 的输入特征矩阵，其中 n_features 为特征数，m 为样本数。
        y_true (np.ndarray): 形状为 (m,) 的真实标签数组。
        y_pred (np.ndarray): 形状为 (m,) 的预测标签数组。
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 包含两个元素的元组，分别为：
            - 梯度 (np.ndarray): 形状为 (n_features,) 的梯度数组。
            - 平均误差 (np.ndarray): 形状为 (1,) 的平均误差数组。
    
    """
    m = y_true.shape[0]
    gradient = np.dot(X.T, (y_pred - y_true)) / m
    return gradient, np.mean(y_pred - y_true, axis=0)


class LogisticRegression:
    def __init__(self, learning_rate=0.05, num_iterations=100):
        """
        初始化线性回归模型的参数。
        
        Args:
            learning_rate (float, optional): 学习率，用于梯度下降算法中更新权重和偏置项，默认为0.05。
            num_iterations (int, optional): 梯度下降算法的迭代次数，默认为100。
        
        Returns:
            None
        
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        对模型进行训练，通过梯度下降更新权重和偏置。
        
        Args:
            X (ndarray): 形状为(m, n)的输入数据，m表示样本数量，n表示特征数量。
            y (ndarray): 形状为(m,)的标签数据，表示每个样本的类别。
        
        Returns:
            None
        
        """
        # 获取输入数据的维度
        m, n = X.shape
        # 获取类别的数量
        num_classes = len(np.unique(y))
        # 初始化权重矩阵，大小为(n, num_classes)，初始值为0
        self.weights = np.zeros((n, num_classes))
        # 初始化偏置向量，大小为(num_classes)，初始值为0
        self.bias = np.zeros(num_classes)

        # 将y转换为整数类型后再进行转换成one-hot编码
        y_onehot = np.eye(num_classes, dtype=int)[y.astype(int)]  # 将y转换为整数类型后再进行转换成one-hot编码

        for i in range(self.num_iterations):
            # 预测
            z = np.dot(X, self.weights) + self.bias
            y_pred = softmax(z)

            # 计算损失
            loss = categorical_cross_entropy(y_onehot, y_pred)

            # 计算梯度
            dw, db = compute_gradient(X, y_onehot, y_pred)

            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 10 == 0:
                print(f'Iteration {i}, Loss: {loss:.4f}')

    def predict(self, X):
        """
        对输入数据X进行预测，返回预测结果。
        
        Args:
            X (numpy.ndarray): 输入的待预测数据，形状为(n_samples, n_features)，其中n_samples为样本数量，n_features为特征数量。
        
        Returns:
            numpy.ndarray: 预测结果，形状为(n_samples,)，其中每个元素为对应样本的预测类别标签。
        
        """
        z = np.dot(X, self.weights) + self.bias
        y_pred = softmax(z)
        return np.argmax(y_pred, axis=1)