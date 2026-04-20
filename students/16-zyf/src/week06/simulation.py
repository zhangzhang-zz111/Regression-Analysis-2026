from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models import CustomOLS
from utils import add_intercept, evaluate_model, train_test_split_simple, write_text


def scenario_A_synthetic(results_dir: Path):
    """
    场景A：合成数据测试
    """
    np.random.seed(42)

    n = 1000
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(5, 1.5, n)
    x3 = np.random.normal(20, 3, n)

    X_raw = np.column_stack([x1, x2, x3])
    noise = np.random.normal(0, 2, n)

    # 真实DGP
    y = 5 + 2.0 * x1 - 1.5 * x2 + 0.8 * x3 + noise

    X = add_intercept(X_raw)

    X_train, X_test, y_train, y_test = train_test_split_simple(X, y, test_ratio=0.2)

    custom_model = CustomOLS()
    sklearn_model = LinearRegression(fit_intercept=False)

    report = []
    report.append("# Synthetic Data Report\n\n")
    report.append("## Model Comparison\n\n")
    report.append("| Model | Fit Time | R2 |\n")
    report.append("|---|---:|---:|\n")
    report.append(evaluate_model(custom_model, X_train, y_train, X_test, y_test, "CustomOLS"))
    report.append(evaluate_model(sklearn_model, X_train, y_train, X_test, y_test, "sklearn.LinearRegression"))

    custom_r2 = custom_model.score(X_test, y_test)

    # 作业要求：断言结果合理
    assert custom_r2 > 0.80, f"Unexpectedly low R^2: {custom_r2:.4f}"

    report.append("\n## Conclusion\n\n")
    report.append(f"- CustomOLS test R2 = **{custom_r2:.4f}**\n")
    report.append("- Synthetic test passed, so the engine behaves as expected.\n")

    write_text(results_dir / "synthetic_report.md", "".join(report))

    y_pred = custom_model.predict(X_test)
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("Scenario A: True vs Predicted")
    plt.tight_layout()
    plt.savefig(results_dir / "synthetic_scatter.png", dpi=150)
    plt.close()


def scenario_B_real_world(results_dir: Path):
    """
    场景B：真实营销数据分析
    数据路径：src/week06/data/q3_marketing.csv
    """
    data_path = Path(__file__).parent / "data" / "q3_marketing.csv"

    if not data_path.exists():
        write_text(
            results_dir / "real_world_report.md",
            "# Real World Report\n\n找不到 `data/q3_marketing.csv`，请检查数据文件位置。\n",
        )
        return

    df = pd.read_csv(data_path)

    required_cols = [
        "Region",
        "TV_Budget",
        "Radio_Budget",
        "SocialMedia_Budget",
        "Is_Holiday",
        "Sales",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        write_text(
            results_dir / "real_world_report.md",
            "# Real World Report\n\n"
            f"数据缺少这些列：{missing_cols}\n"
        )
        return

    # 清洗列名和 Region 值，避免空格、大小写问题
    df.columns = df.columns.str.strip()
    df["Region"] = df["Region"].astype(str).str.strip().str.upper()

    # 按地区拆分
    df_na = df[df["Region"] == "NA"].copy()
    df_eu = df[df["Region"] == "EU"].copy()

    if df_na.empty or df_eu.empty:
        unique_regions = df["Region"].dropna().unique().tolist()
        write_text(
            results_dir / "real_world_report.md",
            "# Real World Report\n\n"
            "无法正确分离 NA 和 EU，请检查 Region 列。\n\n"
            f"当前检测到的 Region 值有：{unique_regions}\n"
        )
        return

    # 解释变量：广告预算 + 是否节假日
    feature_cols = [
        "TV_Budget",
        "Radio_Budget",
        "SocialMedia_Budget",
        "Is_Holiday",
    ]

    # 北美市场
    X_na_raw = df_na[feature_cols].to_numpy(dtype=float)
    y_na = df_na["Sales"].to_numpy(dtype=float)
    X_na = add_intercept(X_na_raw)

    # 欧洲市场
    X_eu_raw = df_eu[feature_cols].to_numpy(dtype=float)
    y_eu = df_eu["Sales"].to_numpy(dtype=float)
    X_eu = add_intercept(X_eu_raw)

    model_na = CustomOLS().fit(X_na, y_na)
    model_eu = CustomOLS().fit(X_eu, y_eu)

    # 联合F检验：检验 TV / Radio / Social / Holiday 是否整体显著
    # beta = [Intercept, TV_Budget, Radio_Budget, SocialMedia_Budget, Is_Holiday]
    C = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=float)
    d = np.array([0, 0, 0, 0], dtype=float)

    na_test = model_na.f_test(C, d)
    eu_test = model_eu.f_test(C, d)

    na_r2 = model_na.score(X_na, y_na)
    eu_r2 = model_eu.score(X_eu, y_eu)

    coef_names = ["Intercept"] + feature_cols
    coef_na_lines = "\n".join(
        [f"- {name}: {coef:.4f}" for name, coef in zip(coef_names, model_na.coef_)]
    )
    coef_eu_lines = "\n".join(
        [f"- {name}: {coef:.4f}" for name, coef in zip(coef_names, model_eu.coef_)]
    )

    report = []
    report.append("# Real World Marketing Report\n\n")
    report.append("## Dataset Columns Used\n\n")
    report.append("- Region\n")
    report.append("- TV_Budget\n")
    report.append("- Radio_Budget\n")
    report.append("- SocialMedia_Budget\n")
    report.append("- Is_Holiday\n")
    report.append("- Sales\n\n")

    report.append("## Market Comparison\n\n")
    report.append("| Market | Sample Size | R2 | F-stat | P-value |\n")
    report.append("|---|---:|---:|---:|---:|\n")
    report.append(
        f"| North America | {len(df_na)} | {na_r2:.4f} | {na_test['f_stat']:.4f} | {na_test['p_value']:.6f} |\n"
    )
    report.append(
        f"| Europe | {len(df_eu)} | {eu_r2:.4f} | {eu_test['f_stat']:.4f} | {eu_test['p_value']:.6f} |\n"
    )

    report.append("\n## North America Coefficients\n\n")
    report.append(coef_na_lines + "\n")

    report.append("\n## Europe Coefficients\n\n")
    report.append(coef_eu_lines + "\n")

    report.append("\n## Plain-Language Interpretation\n\n")
    if na_test["p_value"] < 0.05:
        report.append("- **North America**：广告预算和节假日变量整体上对销售额有显著影响。\n")
    else:
        report.append("- **North America**：广告预算和节假日变量整体上对销售额没有显著影响。\n")

    if eu_test["p_value"] < 0.05:
        report.append("- **Europe**：广告预算和节假日变量整体上对销售额有显著影响。\n")
    else:
        report.append("- **Europe**：广告预算和节假日变量整体上对销售额没有显著影响。\n")

    report.append(
        "\n使用 OOP 的好处是：我们可以创建 `model_na` 和 `model_eu` 两个独立实例，"
        "分别保存两个市场的系数、协方差矩阵和 F 检验结果，不会混在一起。\n"
    )

    write_text(results_dir / "real_world_report.md", "".join(report))

    # 画图：真实值 vs 预测值
    y_na_pred = model_na.predict(X_na)
    y_eu_pred = model_eu.predict(X_eu)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_na, y_na_pred, alpha=0.6, label="North America")
    plt.scatter(y_eu, y_eu_pred, alpha=0.6, label="Europe")
    plt.xlabel("True Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Market Comparison: True vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "market_comparison.png", dpi=150)
    plt.close()


def generate_summary_report(results_dir: Path):
    """
    汇总报告
    """
    summary = []
    summary.append("# Summary Report\n\n")
    summary.append("Generated automatically by running main.py.\n\n")

    synthetic_path = results_dir / "synthetic_report.md"
    realworld_path = results_dir / "real_world_report.md"

    if synthetic_path.exists():
        summary.append("## Synthetic Scenario\n\n")
        summary.append(synthetic_path.read_text(encoding="utf-8"))
        summary.append("\n\n")

    if realworld_path.exists():
        summary.append("## Real-World Scenario\n\n")
        summary.append(realworld_path.read_text(encoding="utf-8"))
        summary.append("\n")

    write_text(results_dir / "summary_report.md", "".join(summary))