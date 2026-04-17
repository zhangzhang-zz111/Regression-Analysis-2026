import numpy as np
from data_generator import generate_design_matrix

def monte_carlo_simulation(rho, n_simulations=1000, beta_true=np.array([5.0, 3.0]), sigma=2.0, random_state=42):
    """
    执行蒙特卡洛模拟，获取β的估计值序列。
    
    参数:
        rho: X1与X2的相关系数
        n_simulations: 模拟次数
        beta_true: 真实的参数向量[β1, β2]
        sigma: 噪音的标准差
        random_state: 随机种子
        
    返回:
        beta_hat_list: 形状为(n_simulations, 2)，每次模拟得到的β1、β2估计值
        X: 生成的固定设计矩阵
    """
    np.random.seed(random_state)
    
    # 1. 生成固定的设计矩阵X（只生成一次）
    X = generate_design_matrix(rho=rho, random_state=random_state)
    n_samples = X.shape[0]
    
    beta_hat_list = []
    
    # 2. 循环n_simulations次，每次生成新的噪音
    for _ in range(n_simulations):
        # 生成纯随机噪音ε ~ N(0, σ²)
        epsilon = np.random.normal(loc=0, scale=sigma, size=n_samples)
        
        # 生成响应变量y = Xβ + ε
        y = X @ np.concatenate([[0.0], beta_true]) + epsilon
        
        # 最小二乘估计：β̂ = (XᵀX)⁻¹Xᵀy
        XtX_inv = np.linalg.inv(X.T @ X)
        beta_hat = XtX_inv @ X.T @ y
        
        # 只保留β1和β2的估计值（去掉截距项）
        beta_hat_list.append(beta_hat[1:])
    
    return np.array(beta_hat_list), X

if __name__ == "__main__":
    # 实验A：ρ=0（正交/独立特征）
    beta_hat_A, X_A = monte_carlo_simulation(rho=0.0, random_state=42)
    # 实验B：ρ=0.99（高度共线性）
    beta_hat_B, X_B = monte_carlo_simulation(rho=0.99, random_state=42)
    
    print("实验A（ρ=0）的估计值形状:", beta_hat_A.shape)
    print("实验B（ρ=0.99）的估计值形状:", beta_hat_B.shape)