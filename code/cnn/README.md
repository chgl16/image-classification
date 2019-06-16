# CNN实现

1. 基础配置
```python
# 训练验证数据集目录
path = '../../data/train-validation-set'
# 模型保存地址，最后接的是模型名字
model_path = '../../model/model.ckpt'
```

2. 参数调试
```python
...
# 训练集和验证集，训练集比例
ratio = 0.7
...

# 训练和测试数据，n_epoch是训练次数
n_epoch = 50
batch_size = 32

# 以及各层的参数
...
```

3. 网络结构
![网络结构](https://i.loli.net/2019/06/16/5d0604f22859b58442.png)

<hr>

> 代码最后部分的是打印图标，包括 train_loss, train_acc, validation_loss, volidation_acc。  
> 训练完所有的epoch后
> 1. validation_loss趋于0（我们这个模型训练结果大概0.5）  
> 2. validation_acc趋于1.0（这个大概0.90）
