import numpy as np
import matplotlib.pyplot as plt
from simulation import monte_carlo_simulation

def compare_covariance_matrices(beta_hat_array, X, sigma=2.0):
    """
    计算经验协方差矩阵与理论协方差矩阵并打印对比。
    
    参数:
        beta_hat_array: 蒙特卡洛模拟得到的β̂序列
        X: 设计矩阵
        sigma: 噪音标准差
        
    返回:
        emp_cov: 经验协方差矩阵
        theo_cov: 理论协方差矩阵
    """
    # 1. 经验协方差矩阵：用numpy.cov计算
    emp_cov = np.cov(beta_hat_array.T)
    
    # 2. 理论协方差矩阵：σ²(XᵀX)⁻¹
    XtX_inv = np.linalg.inv(X.T @ X)
    # 取后两列对应的协方差部分（去掉截距项）
    theo_cov_full = sigma**2 * XtX_inv
    theo_cov = theo_cov_full[1:, 1:]
    
    # 打印结果
    print("="*50)
    print("经验协方差矩阵:")
    print(emp_cov.round(4))
    print("\n理论协方差矩阵:")
    print(theo_cov.round(4))
    print("="*50)
    
    return emp_cov, theo_cov

def plot_beta_hat_scatter(beta_hat_A, beta_hat_B, beta_true=np.array([5.0, 3.0])):
    """
    绘制正交 vs 共线性的β̂散点对比图。
    """
    plt.figure(figsize=(8, 8), dpi=120)
    
    # 实验A（ρ=0）：蓝色圆点
    plt.scatter(beta_hat_A[:, 0], beta_hat_A[:, 1], 
                alpha=0.6, label="正交特征 (ρ=0)", color="#1f77b4", s=10)
    # 实验B（ρ=0.99）：橙色圆点
    plt.scatter(beta_hat_B[:, 0], beta_hat_B[:, 1], 
                alpha=0.6, label="高度共线性 (ρ=0.99)", color="#ff7f0e", s=10)
    
    # 标出真实β的中心点
    plt.scatter(beta_true[0], beta_true[1], color="red", marker="*", s=200, 
                label=f"真实β = {beta_true}")
    
    plt.xlabel(r"$\hat{\beta}_1$", fontsize=14)
    plt.ylabel(r"$\hat{\beta}_2$", fontsize=14)
    plt.title("正交 vs 共线性下的参数估计分布", fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.savefig("beta_hat_scatter.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # 运行模拟
    beta_hat_A, X_A = monte_carlo_simulation(rho=0.0, random_state=42)
    beta_hat_B, X_B = monte_carlo_simulation(rho=0.99, random_state=42)
    
    # Task3: 对比实验B的协方差矩阵
    print("===== 实验B（ρ=0.99）协方差矩阵对比 =====")
    emp_cov_B, theo_cov_B = compare_covariance_matrices(beta_hat_B, X_B)
    
    # Task4: 绘制散点图
    plot_beta_hat_scatter(beta_hat_A, beta_hat_B)