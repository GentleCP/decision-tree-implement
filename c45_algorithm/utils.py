import math
from collections import Counter

import pandas as pd

def load_file(file_path):
    '''
    获取csv数据，训练集和测试集
    :param file_path: csv文件路径
    :return: 训练数据，DataFrame
    e.g.       id    start      end  ...   ty5   ty6                  ty_order
        0     47  07:01.0  07:31.0  ...    -1    -1          非法连接非法连接-1-1-1-1
    '''
    train_data = pd.read_csv(file_path)
    return train_data


def cal_type_entropy(type_data):
    '''
    计算类别信息熵
    :param type_data:  类别数据，Series
    e.g.     0      detect
             1      detect
             2      detect
             3      detect
    :return: 类别信息熵
    '''
    type_entropy = 0
    total = len(type_data)
    type_counter = dict(Counter(type_data))
    for t_count in type_counter.values():
        type_entropy += -(t_count / total) * math.log2(t_count / total)
    return type_entropy


def cal_attr_entropy(type_data, attr_data):
    '''
    ????
    计算每个属性的信息熵
    :param attr_data: 属性数据，Series
    :param type_data: 类型数据，Series
    e.g.    0         10.10.13.23
            1      192.168.20.199
            2      192.168.20.199
            3      192.168.20.199
    :return: 属性信息熵
    '''
    attr_entropy = 0
    total = len(attr_data)
    attr_counter = dict(Counter(attr_data))
    for attr, a_count in attr_counter.items():
        # 从type_data中找出该属性值对应的所有type_data
        attr_type_dict = {}
        for i, v in attr_data.items():
            if v == attr:
                attr_type_dict[i] = type_data[i]  # 将对应的分类结果保存
        attr_type_data = pd.Series(attr_type_dict)
        attr_entropy += a_count / total * (cal_type_entropy(attr_type_data))
    return attr_entropy


def cal_info_gain(type_data, attr_data):
    '''
    计算信息增益
    :param type_data: 类型数据
    :param attr_data: 属性数据
    :return: 信息增益值
    '''
    return cal_type_entropy(type_data) - cal_attr_entropy(type_data, attr_data)


def cal_info_gain_rate(type_data, attr_data):
    '''
    计算信息增益率
    :param type_data: 类型数据
    :param attr_data: 属性数据,Series
    :return: 信息增益率
    '''
    attr_ins_info = cal_type_entropy(attr_data)  # 计算属性内在信息的方式与计算类别信息熵相同
    return cal_info_gain(type_data, attr_data) / attr_ins_info if attr_ins_info != 0 else 0