# Titanic-survival-transfer-learning
一个基于数据清洗和PCA特征工程、神经网络的泰坦尼克号幸存者预测

【English】

🚀 Project Overview
This project addresses the classic Titanic survival prediction task using PyTorch, with a specific focus on Transfer Learning. Under a constrained environment where a pre-trained feature extractor remains "frozen," I designed and optimized a custom MLP classifier to significantly enhance predictive performance.

Final Accuracy: 81.33% (Top tier for this task)

Performance Jump: From 64.67% baseline to 81.33% through iterative tuning.

🛠️ Technical Highlights
Data Preprocessing: Implemented Median Imputation for missing values and robust Standardization to ensure data quality.

Feature Engineering: Applied PCA (Principal Component Analysis) to reduce 8 features down to 5, retaining 80.03% of the information variance.

Transfer Learning: Frozen a pre-trained backbone (requires_grad=False) and trained only the appended classification layers.

Model Optimization: * Integrated Batch Normalization which led to a 10% drop in Loss (from 0.49 to 0.37).

Utilized Dropout (0.2) to prevent overfitting during long-term (800 epochs) training.

📦 Tech Stack
Core: PyTorch, Python

Libraries: Scikit-learn, Pandas, NumPy.

【中文】

🚀 项目概览
本项目使用 PyTorch 框架解决了经典的泰坦尼克号生存预测任务，核心侧重于 迁移学习 (Transfer Learning) 的应用。在“冻结”预训练特征提取器（不改变其参数）的限制条件下，我通过设计并持续优化自定义的 MLP 分类器，实现了预测性能的显著飞跃。

最终准确率： 81.33% (该任务中的顶级水平)

性能提升： 通过迭代调优，从 64.67% 的初始水平提升至 81.33%。

🛠️ 技术亮点
数据预处理： 采用 中位数填充 缺失值及 标准化 (Standardization) 处理，确保输入数据的高质量。

特征工程： 应用 PCA (主成分分析) 将 8 维特征降至 5 维，保留了 80.03% 的核心信息方差。

迁移学习策略： 严格执行参数冻结 (requires_grad=False)，仅训练新增的分类头，充分利用预训练模型的高阶特征提取能力。

模型优化实战：

引入 批归一化 (Batch Normalization)，使 Loss 直接下降约 10%（从 0.49 降至 0.37 附近）。

使用 Dropout (0.2) 有效抑制了 800 轮长周期训练中的过拟合风险。

📦 技术栈
深度学习: PyTorch

数据处理: Scikit-learn, Pandas, NumPy
