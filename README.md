# MachineLearningTools
## 整理存储我在机器学习项目中经常重复使用的工具
- 使用python setup.py install安装

### Classification
- GBC：超级好用的GradientBoostingClassifier，不仅分类准确，同时还能够展示出各个Feature的importance，非常有用！
- plot_confusion_materix：展示混淆矩阵，代码来自网络，文件内有标注
- PCA_Analysis:对数据进行主成分分析，绘制输入Feature的权重贡献，以及绘制用PCA降维后的X的分类情况，最高绘制三维
- LinearNN：用线性NN进行分类，显示Feature权重情况
### Regression
- GBR_FeatureImportanceEstimater：非常好用的GBR，能够自由调节拟合过拟合，使训练集和测试集的准确度达到平衡，同时也能够知道各个Feature的importance
- LinearNNFeatureExtraction:使用线性NN进行拟合，plot出权重得知各个Feature的权重，效果不如GBR

### DataPreprocession
- DataReplication:数据扩增，对某个label的数据进行复制，保持样本数量平衡