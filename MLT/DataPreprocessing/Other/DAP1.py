import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,roc_curve,auc
import  matplotlib.pyplot as plt
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
import os
import tqdm
from collections import Counter
from sklearn.externals import joblib
'''
很多通用代码的实现，可以先查找sklearn库中已有的实现

'''


# set Graphviz path if want to show decision tree
os.environ["PATH"] += os.pathsep + 'G:\\Program Files\\Graphviz2.38\\bin'


def _is_type_pd_dataframe(pd_data):
    '''
    判断是否为pd.DateFrame格式
    judge if pd.DataFrame()
    :param pd_data:
    :return: None
    '''
    if not isinstance(pd_data, pd.DataFrame()):
        raise TypeError("pd_data must be a pd.DataFrame()")

def _try_get_label(data,label):
    '''
    尝试获取label，否则报错
    try to return data[label], except raise error
    :param data:  pd.DateFrame()
    :param label: string
    :return:      data[label]
    '''
    try:
        return data[label]
    except:
        raise ValueError("no such label in data")

def _is_valid_choice(var,var_name,values):
    '''
    判断变量var是否在list values中，检验枚举形参的正确性，var_name只是参数名称
    :param var:
    :param var_name:
    :param values:
    :return:
    '''
    text = ""
    for i in values:
        test = text+ " "+i
    if var not in values:
        raise ValueError("{} must be in {}".format(var_name,text))


def time_transform_1(pd_data,label,drop_original=True):
    '''
    transform time like 2017-01-01 10:00:00 to int 20170101100000

    :param csv_file:    pd.DataFrame():  file to transform.
    :param label:       string:          where the time label is.
    :drop_original:     bool:            if drop the original time data.
    :return:            pd.DataFrame():  the time-transformed file which
                                         time label is the first label.
    '''
    _is_type_pd_dataframe(pd_data)
    data = pd_data
    time_l = _try_get_label(data,label)
    new_time_l = []
    num = time_l.shape[0]
    for i in range(num):
        data, time = time_l[i].split(" ")
        y, mouth, d = data.split("-")
        h, m, s = time.split(":")
        time_s = int(y + mouth + d + h + m + s)
        new_time_l.append(time_s)
    if drop_original:
        data = data.drop(labels=label, axis=1)
    data.insert(0, label, pd.Series(new_time_l))
    return data

def filt_data_according_label_and_range(filt_data,judge_label,min_range,max_range):
    '''
    按照label和起始、结束序列筛选数据
    filt data in the range judged by label, e.g. we have <filt_data> :
    time    things
    12      a
    13      b
    15      c
    17      d
    and we have <judge_label> : time
    and <min_range> [10, 14.5]
    and <max_range> [12, 15]
    the data where time is between (10, 12) and (14.5, 15) will be filt out

    :param filt_data:   pd.DataFrame():         be filted
    :param judge_label: string                  the label to judge
    :param min_range:   iterable(
                        including pd.Series)    offer min to judge
    :param max_range:   iterable(
                        including pd.Series)    offer max to judge
    :return:            pd.DataFrame()
    '''
    _is_type_pd_dataframe(filt_data)
    assert len(min_range) == len(max_range)
    result_l = []
    _ = _try_get_label(filt_data,judge_label)
    num = len(min_range)
    for i in range(num):
        result = filt_data[
            (filt_data[judge_label] > min_range[i] &
             (filt_data[judge_label] < max_range[i])) ]
        result_l.append(result)
    result = pd.concat(result_l)

    return result

def set_label_according_label_and_range(data,judge_label,min_range,max_range,label_name,label_value):
    '''
    按照label和起始结束序列设置值
    rather than filt, this time we set the filted data with label, and keep other data
    if have no data[label_name] : create new one with default_value ""

    :param data:
    :param judge_label:
    :param min_range:
    :param max_range:
    :param label_name:
    :param label_value:
    :return:
    '''
    data = data
    _is_type_pd_dataframe(data)
    length = data.shape[0]
    judge_data = list(_try_get_label(data,judge_label))
    assert len(min_range) == len(max_range)
    num = len(min_range)

    try:
        label_l = data[label_name]
    except:
        label_l = ["" for _ in range(length)]

    for i in range(num):
        for j in range(length):
            if (judge_data[j] > float(min_range[i])) and  (
                judge_data[j] < float(max_range[i])
            ):
                label_l[j] = label_value

    label_l = pd.Series(label_l)
    try:
        data.drop(labels=label_name,axis=1)
        data.insert(data.shape[1],label_name,label_l)
    except:
        data.insert(data.shape[1], label_name, label_l)

#TODO:  获取pandas表名，用集合除去labels，然后索引
def normalization_to_numpy_array(pd_data,mode,labels=None):
    '''
    将pd.DataFrame格式的内容转化为normalization过后的numpy
    all the col will be normalized and result will be save to numpy
    :param pd_data:
    :param npy_name:
    :return:
    '''
    _is_valid_choice(mode,"mode",["all","include","exclude"])

    if (mode == "include" or mode == "exclude") and (labels == None or (
                labels and not isinstance(labels, list))):
        raise ValueError("if include or exclude, label must be a list of string")

    result = []
    data = pd_data

    if mode == "all":
        for i in range(pd_data.shape[1]):
            result.append(preprocessing.scale(np.array(data.icol(i))))
            #TODO：更新过期的icol指令
        result = np.transpose(np.array(result))
        return result

    if mode == "include":
        for i in labels:
            result.append(preprocessing.scale(np.array(data[i])))
        result = np.transpose(np.array(result))
        return result

    if mode == "exclude":
        pass




def data_set_shuffled_split_for_binary_classification(
        pd_data,
        label_name,
        label1_value,
        label2_value,
        test_ratio,
        label1_sample_number,
        label2_to_label1_ratio,
        random_seed=None):
    '''
    得到二分类标签比例按混合后的数据，用于平衡正例反例数量
    we have a table, it is <pd_data>:

    para1   para2   class(<label_name>)
    a       b       1    (<label1_value>)
    a       c       2    (<label2_value>)
    d       e       1    (<label1_value>)

    <label1_sample_number> is how many data with label1_value we need
    <label2_to_label1_ratio> the sample number label2 : label1, e.g. if it is 1,
                then data with label1_value will be the same size as data with label2_value
    <test_ratio> sample number train / test
    :param pd_data:                 pd.DataFrame
    :param label_name:              string
    :param label1_value:            Any
    :param label2_value:            Any
    :param test_ratio:              float
    :param label1_sample_number:    int
    :param label2_to_label1_ratio:  float
    :param random_seed:             int
    :return:                        numpy.array     return x_train, y_train, x_test and y_test,
    '''
    if random_seed:
        random.seed(random_seed)

    pd_data = pd_data
    label1 = pd_data[pd_data[label_name]==label1_value]
    label2 = pd_data[pd_data[label_name]==label2_value]

    label1_num = label1.shape[0]
    label2_num = label2.shape[0]

    if label1_num > label2_num:
        temp = label1
        label1 = label2
        label2 =temp
        label1_num = label1.shape[0]
        label2_num = label2.shape[0]

    _label1_sample_number = min(label1_num, label1_sample_number)
    _label2_sample_number = label1_sample_number * label2_to_label1_ratio

    label1_index = random.sample(list(range(label1_num)),_label1_sample_number)
    label2_index = random.sample(list(range(label2_num)),_label2_sample_number)

    result = []
    for i in range(len(label1_index)):
        result.append(label1[label1_index[i]:label1_index+1])

    for i in range(len(label2_index)):
        result.append(label2[label2_index[i]:label2_index+1])

    result  = pd.concat(result)
    result = shuffle(result)

    x = np.array(result.drop(labels=label_name,axis=1))
    y = np.array(result[label_name])

    data_set_number = x.shape[0]
    test_number = int(data_set_number * test_ratio)

    x_train = x[test_number:]
    y_train = y[test_number:]
    x_test = x[:test_number]
    y_test = y[:test_number]

    return x_train, y_train, x_test, y_test


def train_test_set_save_npz(x_train, y_train, x_test, y_test,filename="train_test_set.npz"):
    '''
    save the x_train, y_train, x_test, y_test to npy file for quicker IO
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param filename:
    :return:
    '''
    np.savez(filename,x_train, y_train, x_test, y_test)

def load_train_test_set(filename="train_test_set.npz"):
    '''
    load the x_train, y_train, x_test and y_test saved from function: train_test_set_save_npz
    :param filename:    .npz file
    :return:            x_train, y_train, x_test, y_test
    '''
    temp = np.load(filename)
    return temp["arr_0"], temp["arr_1"], temp["arr_2"], temp["arr_3"]


def write_list_to_pd_dataframe(infile,outfile,label_name,list_of_data):
    '''

    :param infile:
    :param outfile:
    :param label_name:      a string list, like [person, age, birth_date]
    :param list_of_data:    a list of list, like [[wang, li],[24, 34],[1994, 1980]]
    :return:
    '''
    data = pd.read_csv(infile)
    for i in range(len(list_of_data)):
        temp = pd.Series(list_of_data[i])
        data.insert(data.shape[1],label_name[i],temp)
    data.to_csv(outfile,index=False)


def describe_binary_list_of_value1(binary_list,value1,value2):
    '''
    将0101序列转化为1的起始、结束位置序列
    e.g. we have <binary_list>  [0,0,0,1,1,1,0,0,1,1]
    and <value1> = 1, <value2> = 0,
    then return [3, 8] (startT) and [5, 10] (endT) means
                            [3, 5] and [8, 10] is <value1>

    :param binary_list:
    :param value1:
    :param value2:
    :return:
    '''
    startT =[]
    endT = []
    now = binary_list[0]

    for i in range(len(binary_list)):
        if binary_list[i] != now:
            if binary_list[i] == value1:
                startT.append(i)
            elif binary_list[i] == value2:
                endT.append(i-1)
            else:
                raise ValueError("unexpected value {} in binary_list, make sure all are {} or {}.".format(binary_list[i],value1,value2))
            now = binary_list[i]
    # 如果最后是1，要补充完整
    if binary_list[-1] == value1:
        endT.append(len(binary_list))

    return startT, endT

#print(describe_binary_list_of_value1([0,0,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0],1,0))


def plot_decision_tree_0(clf,outfile_name='DT_of_data.png'):
    '''
    画出决策树的内容
    :param clf:
    :param outfile_name:
    :return:
    '''
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)#采用IO的方式传递二进制，然后给pydot处理
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_png(outfile_name)



def binary_classify_predict_and_summary_4_sklearn(trained_sklearn_model, x_test, y_test,print_results=True):
    '''
    二分类总结
    summary for binary classify, predict for x_test.
    :param sklearn_model:  trained sklearn model
    :param x_test:
    :param y_test:
    :return:  prediction, predict_proba
    '''
    clf = trained_sklearn_model
    prediction = clf.predict(x_test)
    accuracy = np.mean(prediction == y_test)
    predict_proba = clf.predict_proba(x_test)
    if print_results:
        print("accuracy\n".ljust(10), accuracy)
        print(classification_report(
            y_true = y_test,
            y_pred = prediction,
            target_names = ["0","1"],
        ))
    return prediction, predict_proba

def draw_ROC_curve(fig_save_file,y_test,predict_proba):
    '''

    :param fig_save_file:
    :param y_test:   or y_true, real true labels
    :param predict_proba: predict_prob got from binary_classify_summary_4_sklearn()
    :return:
    '''
    answer_prob = predict_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=answer_prob)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(fig_save_file,dpi=300)
    plt.show()


def plot_prediction_for_binary_classify(prediction,fig_file_name="plot_prediction.png",true_prediction=None):
    '''
    将二分类结果进行plot对比
    :param prediction:
    :param fig_file_name:
    :param true_prediction:   y_true for contrast, can be None
    :return:
    '''
    import matplotlib.pyplot as new_plt
    if true_prediction is None:
        index = [i for i in range(len(prediction))]
        new_plt.plot(index,prediction)
        new_plt.savefig(fig_file_name,dpi=300)
        new_plt.show()
    else:
        index = [i for i in range(len(prediction))]
        new_plt.plot(index, prediction,color="b")
        new_plt.plot(index,true_prediction,color="r")
        new_plt.savefig(fig_file_name, dpi=300)
        new_plt.show()


def prediction_1_to_onehot(prediction,threshold=0.5,
                         below=0,over=1):
    '''
    if elements in <prediction> lower then <threshold>,
    then append [<below>, <over>] else [<over>, <below>]
    :param prediction:  a list like [0.05, 0.95, 0.12]
    :param threshold:
    :param below:
    :param over:
    :return:  numpy.array
    '''
    result = []
    below_l = [below,over]
    over_l = [over,below]
    for i in prediction:
        if i < threshold:
            result.append(below_l)
        else:
            result.append(over_l)
    return np.array(result)


def prediction_2_to_1(prediction,shift=0.5,
                           below=0,over=1):
    '''
    elements is <prediction> is [a, b], such as [0.75, 0.25]
    if a - shift is less than b + shift:
    then append <below> else <over>
    :param prediction:  a list like [0.05, 0.95, 0.12]
    :param threshold:
    :param below:
    :param over:
    :return:  numpy.array
    '''
    result = []

    for i in prediction:
        if i[0]-shift < i[1] + shift:
            result.append(below)
        elif i[0] -shift > i[1]+shift :
            result.append(over)
    return np.array(result)


def model_train_template_1(x_train,y_train,test_set,
                           model,model_type="keras",
                           train=True,make_prediction=True,
                           train_new_model=True):
    '''
    一个存储keras或者sklaern模型的模板，比如用于决策树和神经网络
    :param x_train:
    :param y_train:
    :param test_set: data set for prediction
    :param model:    keras or sklearn model
    :param model_type: "keras" or "sklearn"
    :param train:       bool
    :param make_prediction:  bool, make prediction for test_set
    :param train_new_model:  bool, abandon former model and train a new one
    :return:                 if make_prediction, return prediction
    '''
    _is_valid_choice(model_type,"model_type",["keras","sklearn"])
    model = model
    if not train_new_model:
        if model_type == "keras":
            try:
                model.load_weights('my_model_weights_of_keras.h5', by_name=True)
                print("load weights successful")
            except:
                print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")
                train = True
        elif model_type == 'sklearn':
            try:
                model = joblib.load("sklearn_model.m")
            except:
                print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")
                train = True

    if train:
        model.fit(x_train,y_train)

    if model_type == "keras":
        model.save_weights('my_model_weights_of_keras.h5')
    elif model_type == "sklearn":
        joblib.dump(model,"sklearn_model.m")

    if make_prediction:
        result = model.fit(test_set)
        return result


def sklearn_preprocess_model_1(model,data_set,model_save_name):
    '''
    用于sklearn预处理的模型，比如标签化和正则化
    :param model:   model like StandardScale() or LabelEncoder()
    :param data_set:  one (None, 1) data
    :param model_save_name: model file save name
    :return:  model after train:
            e.g. trained StandardScale() model can use transform or reverse_transform
             the same as LabelEncoder()
    '''
    model = model
    model.fit(data_set)
    joblib.dump(model,filename=model_save_name)
    return model



