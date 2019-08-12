from c45_algorithm.utils import *


class C45Tree(object):
    """
    C45 决策树结构
    """

    def __init__(self, node_type, klass=None, feature=None):
        '''
        :param node_type: 节点类型，可能为内部节点或叶节点
        :param klass: 叶节点表示的分类， 内部节点为None
        :param feature: 划分当前树的feature（使得当前树中信息增益最大的特征）
        '''
        self.node_type = node_type
        self.klass = klass
        self.feature = feature
        self.tree_dict = {}  # 键表示特征的可能值v，值为根据v得到的子树

    def __repr__(self):
        return "C45Tree<node_type:{node_type}, class:{klass}, feature:{feature}, tree_dict:{tree_dict}>\n".format(
            node_type = self.node_type,
            klass = self.klass,
            feature = self.feature,
            tree_dict = self.tree_dict,
        )

    def add_tree(self, key, tree):
        self.tree_dict[key] = tree

    def predict(self, test_data, id):
        '''
        根据特征预测分类
        :param test_data: 单条待预测数据,DataFrame
        :param id: 数据编号
        :return: 分类结果
        '''
        if self.node_type == 'leaf':
            # 如果是叶节点说明已经预测出结果
            return self.klass
        # 如果是内部节点，递归预测结果
        try:
            next_tree = self.tree_dict.get(test_data[self.feature][id])   # 根据节点的值选择决策树下一节点
            return next_tree.predict(test_data, id)
        except AttributeError:
            # 对应属性值并未训练过
            return None



def build_tree(train_data, class_feature, features):
    '''
    循环计算构建决策树，形式如下：
    while(当前节点‘不纯’)
        计算各个属性的信息增益率
            - 计算各个属性的信息增益 cal_info_gain
                 - 计算当前节点类别熵 cal_type_entropy
                 - 计算当前节点属性熵 cal_attr_entropy
            - 计算各个属性的内在信息（分类信息度量）
        选择信息增益率最大的作为划分属性
    end while
    当前节点设置为叶节点
    :param train_data: 训练数据集, DataFrame
    :param class_feature: 作为分类的列名，str
    :param features: 特征,list
    :return:  C45Tree
    '''
    label_set = set(train_data[class_feature])
    if not label_set:
        return
    # 若训练集所有实例为同一类,即纯了
    if len(label_set) == 1:
        return C45Tree('leaf', klass=label_set.pop())

    # 为每个属性计算信息增益率，选择最大的作为划分属性
    max_igr = 0  # 最大信息增益率
    max_feature = None  # 拥有最大信息增益率的特征

    for feature in features:
        # 对每个特征计算信息增益率
        this_igr = cal_info_gain_rate(type_data=train_data[class_feature],
                                      attr_data=train_data[feature])
        if this_igr > max_igr:
            max_feature = feature
            max_igr = this_igr
    # 构建决策树
    c45_tree = C45Tree('internal', feature=max_feature)
    # 取除去max_feature的剩余特征继续划分
    sub_features = list(filter(lambda x: x != max_feature, features))
    # 对选择的划分属性的每个可能情况分别建子树
    try:
        max_feature_values = set(train_data[max_feature])
    except KeyError:
        max_feature_values = []
    for max_feature_value in max_feature_values:
        sub_train_data = pd.DataFrame()
        for i, value in zip(train_data[max_feature].index, train_data[max_feature].values):
            if value == max_feature_value:
                sub_train_data = sub_train_data.append(train_data.loc[[i]])  # 按照索引添加对应的Data到sub_train_data中
        # 将已作为划分属性的属性列从data中去除
        sub_train_data = sub_train_data.drop(max_feature, axis=1)
        # 获得了该属性值的所有data，进行下一轮的划分
        sub_tree = build_tree(sub_train_data, class_feature, sub_features)
        c45_tree.add_tree(max_feature_value, sub_tree)

    return c45_tree

def train(train_data_path = None,
          class_feature = None,
          features = []):
    '''
    训练决策树模型
    :param train_data_path: 训练数据路径
    :param class_feature: 分类结果属性
    :param features: 用于分类的属性
    :return: 训练好的决策树
    '''
    train_data = load_file(train_data_path)
    return build_tree(train_data, class_feature, features)

def test(test_data_path = None,
         class_feature = None,
         tree = None):
    '''
    测试训练的决策树模型
    :param class_feature: 分类结果属性，用于检验正确率
    :param test_data_path: 测试数据路径
    :param tree: 决策树
    :return:
    '''
    test_data = load_file(test_data_path)
    real_classes = list(test_data[class_feature])
    acc_count = 0
    for i in test_data.index:
        # 以DataFrame的索引作为查询点
        predict_res = tree.predict(test_data.loc[[i]], i)
        real_res = real_classes.pop(0)
        if predict_res == real_res:
            acc_count += 1
    return acc_count / len(test_data)




