## 用DANN实现按键检测

本项目是fork[fungtion/DANN_py3](https://github.com/fungtion/DANN_py3)
#### Environment
- Pytorch 1.6
- Python 3.8.5

#### Network Structure

在原网络的基础上基本只修改了输入层的信息

#### Dataset

数据都在collect_data文件夹中，其中的子文件夹others是我之前采集的数据的汇总，其他子文件夹是五月底采集的几组测试数据，每组有300个键（按顺序采集并切断好的信号）
#### Training

进入MyDann，用jupyter notebook运行
程序会把源域和目标域的一部分作为训练集，然后再用目标域的其他部分作为测试集

#### Todo
1. DANN停止训练的epoch如何确定？
