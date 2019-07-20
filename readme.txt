data_deal.py文件，用于把猫狗的图片制作成batch文件，方便网络分批输入数据。

运行完data_deal.py后，再执行tensor.py训练和测试。

tensor.py文件，用于训练模型，和测试模型，以及最后对官网的测试图片进行分类。

先训练，生成模型后，再更改参数IS_TRAIN进行测试，最后分类。