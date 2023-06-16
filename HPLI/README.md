# Hits
### high performance index with a intelligent tuning system

PO precise index only
HO Hits index only

POT precise index with tuning system
HOT Hits index with tuning system

如果想要使用本项目进行测试，仅需将数据集准备好即可，本索引评测过程使用的数据集为<double,long long>类型的二进制数据集，
如果使用其他类型的数据集请自行修改部分代码。

如果配置C++ pytorch环境不方便，可以移除pytorch库相关文件，model.hpp中有已经实现好的简单版本的神经网络可以用与拟合数据的CDF。
