# 人物关系提取

用数据集：https://github.com/percent4/people_relation_extract

写了一个小项目，用来做人物关系提取。

数据比较简单，称为简单是指：

1. 文本短，信息集中

2. 有和关系分类强相关的词汇出现，且词汇表述并不多样

数据在/data目录下，raw.xlsx数量在2800左右，raw2.xlsx数量在3900左右，两个数据有重复

## 效果：

训练/测试集数量/测试集F1值：3100/775/0.886

训练/测试集数量/测试集F1值：2035/508/0.845

## 模型结构简要说明

**结构**

bert+bilstm+dense+dense+softmax

**输入**

在实体前后插入特殊字符的文本

>例如：
>
>原文：不过，他对润生的姐姐润叶倒怀有一种亲切的感情。
>
>处理成：
>
> 不过，他对#润生#的姐姐$润叶$倒怀有一种亲切的感情。

**其他说明**

最后取“#”和“$”这两个特殊字符对应的向量做分类

## 依赖包

tensorflow-gpu==2.x

transformers==3.4.0

还需要其他包，不过没版本要求，看提示装上即可

## 运行说明

1. 运行process_data.py处理数据

2. 运行train.py训练模型

> 因为使用了预训练模型，所以需要先从https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main
>
> 下载对应的config.json,tf_model.h5,vocab.txt文件

## 硬件配置与训练耗时相关说明

文本最大长度：128

batch_size: 8

是否开启混合精度训练：是

显卡：TITAN RTX

一个epoch大概30秒左右，需要显存5G

## 相关配置说明

1. 数据预处理代码：process_data.py 里可以修改处理的数据集合保存地址

2. config.py 文件里可以配置文本最大长度，batch_size, 训练数据地址，测试数据地址，预训练模型地址

## 优化建议

数据比较简单，当前数据量下做到90+的F1应该是比较简单的。可以尝试以下方法优化：

1. 数据增强

2. 设计更好的特征抽取方式

当前取特殊字符对应向量做分类，可以加入其它特征，例如全文做cnn+max-pooling，取强相关词汇对应向量，例如姐姐，领导，儿子这些

3. 换预训练模型

换large版的或者效果更好的点的预训练模型，例如macbert，xlnet

4. 调参

结构比较简单，超参数也不多而且也不是什么敏感的超参，不过调一调应该还是有点提升的。

5. 其他

领域语料finetue,转化为PET形式等等。