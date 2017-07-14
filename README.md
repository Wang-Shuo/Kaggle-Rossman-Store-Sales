### Rossmann 销售预测

------

#### 简介

------

该项目是优达学城机器学习工程师纳米学位毕业项目，源自2015年的Kaggle比赛[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales). 项目的任务是要根据商店的过往销售记录和商店自身的一些信息来预测未来六周的销售额。项目最终的结果提交到Kaggle排行榜，排名为464/3303.

#### 要求

------

##### 软件

本项目中用到的软件和库如下（没给出版本的软件和库用最新版本为佳）：

- Python栈: Python 3.6.1, pandas, numpy, scikit-learn, matplotlib.
- XGBoost: Gradient Boosting算法的一个高效实现，本项目中使用它的Python包
- Jupyter Notebook: 一种交互式笔记本，能方便地显示代码输出结果，被广泛应用于数据科学领域。

#### 项目运行指南

------

input文件夹为项目中使用的数据集，包含三个csv格式的文件：

- train.csv：项目的训练集，包含了各个门店的历史销售额数据
- test.csv： 项目的测试集
- store.csv： 各个门店的补充信息

output文件夹用于保存程序运行期间产生的各种输出：

- features： 用于保存模型选择过程中各个模型的特征
- test_prediction： 用于保存各个模型在测试集上的预测结果
- valid_prediction： 用于保存各个模型在交叉验证集上的预测结果

下面对各个程序文件的运行次序和功能做一个简单介绍：

1. 使用jupyter notebook运行Exploratory Data Analysis.ipynb文件，对数据进行探索和可视化分析；
2. 运行benchmark.py文件得到一个基准模型，供后面的模型参考；
3. 运行linear_regression_whole.py文件，在整个数据集上使用线性回归模型；
4. 运行linear_regression_each.py文件，对每个门店的数据单独使用线性回归模型；
5. 运行xgb_basic.py文件，得到一个使用xgboost的基准模型；
6. 运行model_selection.py进行模型选择，会得到100个基于xgboost的模型，每个模型只是特征不同，各自的特征会以文本形式保存在output的子文件夹features中。
7. 运行parameter_tuning_phase1.py和parameter_tuning_phase2.py对第6步100个模型中的最小交叉验证误差模型进行两次调参；
8. 运行model_ensemble.py对几个模型进行融合，得到最终结果。
9. 运行result_visualization.ipynb文件对最终模型的表现进行可视化。

由于数据集相对较大，运行xgboost模型时可能会耗时较久。以我的电脑为例（Intel 4核i5-2300 2.80GHz CPU），单个xgboost模型运行时间大概在30~35分钟左右。

