import numpy as np

def generate_design_matrix(n_samples=1000, rho=0.0, random_state=42):
    """
    生成带有两个特征X1、X2的设计矩阵X，控制二者的相关系数rho。
    
    参数:
        n_samples: 样本量
        rho: X1与X2的相关系数ρ，范围[-1,1]
        random_state: 随机种子，保证结果可复现
        
    返回:
        X: 设计矩阵，形状为(n_samples, 3)，第一列为截距项，后两列为X1、X2
    """
    np.random.seed(random_state)
    
    # 生成独立的标准正态变量
    Z1 = np.random.randn(n_samples)
    Z2 = np.random.randn(n_samples)
    
    # 构造相关的X1和X2
    # 相关系数ρ的构造方法：X2 = ρ*Z1 + sqrt(1-ρ²)*Z2
    X1 = Z1
    X2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # 添加截距项，构建设计矩阵X
    X = np.column_stack([np.ones(n_samples), X1, X2])
    
    return X