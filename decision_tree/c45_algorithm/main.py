# @Author  : GentleCP
# @Email   : 574881148@qq.com
# @File    : main.py
# @Item    : PyCharm
# @Time    : 2019/8/12 13:17
# @WebSite : https://www.gentlecp.com
from c45_algorithm.c45 import *
from c45_algorithm import settings

if __name__ == '__main__':
    # train
    tree = train(train_data_path=settings.TRAIN_DATA_PATH,
                 class_feature=settings.CLASS_FEATURE,
                 features=settings.FEATURES)

    acc = test(test_data_path=settings.TEST_DATA_PATH,
               class_feature=settings.CLASS_FEATURE,
               tree=tree)

    print('决策树准确率:{}%'.format(acc * 100))
