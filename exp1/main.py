from model import LogisticRegression
from data_process import *
from sklearn.metrics import accuracy_score

def train():
    """
    训练模型并输出准确率。
    
    Args:
        无。
    
    Returns:
        无返回值。
    
    """
    file_path = 'data/train.tsv'
    n = 2  # 设置n-gram的大小
    
    X, y, _ = process_data(file_path, n)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    
    # 训练 Logistic 回归模型
    model = LogisticRegression(learning_rate=0.05, num_iterations=100)
    model.fit(X_train, y_train)

    # 预测并评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    train()