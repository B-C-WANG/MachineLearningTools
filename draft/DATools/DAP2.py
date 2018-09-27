import pandas as pd
import numpy as np
import pickle

def geohash_encode(list_like,decode_exactly=True,data_split=True):
    '''
    对geohash进行解码
    :param list_like:       geohash list
    :param decode_exactly:  decode_exactly get a higher precision, and offer error
    :param data_split:      if not split, return [(x1, y1),(x2, y2)], else [x1, x2], [y1, y2]
    :return:
    '''
    data = list_like
    if decode_exactly:
        decode = gh.decode_exactly
    else:
        decode = gh.decode

    if data_split:
        x = []
        y = []
        x_error = []
        y_error = []
        for i in data:
            temp = decode(i)
            x.append(temp[0])
            y.append(temp[1])
            if decode_exactly:
                x_error.append(temp[2])
                y_error.append(temp[3])
        if decode_exactly:
            return x, y, x_error, y_error
        else:
            return x, y

    else:
        result = []
        for i in data:
            result.append(decode(i))

        return result



def csv_file_describe(csv_file_list):
    '''
    describe all the information of each csv file in <csv_file_list>
    :param csv_file_list:
    :return:
    '''
    for i in csv_file_list:
        print("{}".format(i))
        pd.read_csv(i).describe(include="all")




def index_three_frequency_table_1(pd_data,index_name,
                                  variables_name_list,test=False,
                                  save="frequency_table"):
    '''
    e.g. we have a <pd_data>:

    name    time    do
    1       120     eat
    1       120     run
    2       120     eat
    2       121     run
    2       120     eat
    and <index_name> "name"
    and <variables_name_list> ["time", "do"]

    we will get a new pd_data:

name time1 time1_frequency time2 time2_frequency time3 \

1    120   1               120   0               120
2    120   0.66666666      121   0.333333333     121

time_3frequency do1 do1_frequency do2 do2_frequency do3 do3_frequency
0               eat 0.5           run 0.5           run 0
0               eat 0.66666666    run 0.3333333     run 0

    this function will get the most 3 frequency items in <variable_name_list>
    of every element in pd.data[<name>]. If it is less than 3 frequency elements,
    the second or the third will be set the same as the former one, and frequency wil
    be 0.

    :param pd_data: pd.DataFrame
    :param variables_name_list:
    :param test:  False or an int, if int, will just deal with first <test> number of
                  data.
    :param save:  False or an string, if string, save to <save>.csv
    :return:  pd.DataFrame
    '''
    data_dict = {}
    data_dict_name_l = []# 用这个而不是dict.keys()主要是防止打乱了顺序
    index = list(set(pd_data[index_name]))
    a = len(index)
    if test:
        a = test
    for i in variables_name_list:
        #TODO:如果扩展成3个以上，更改这里
        for j in range(3):
            data_dict[i+"_{}".format(j)] = []
            data_dict[i + "_{}_freq".format(j)] = []
            data_dict_name_l.append(i+"_{}".format(j))
            data_dict_name_l.append(i + "_{}_freq".format(j))
        # 字典的key分别为 <name>_0 <name>_0_freq <name>_1 <name>_1_freq
            # ....之后会用value这个string进行索引

    for i in tqdm.trange(a):
        # 筛选所有name set中行，得到多行
        row = pd_data[(pd_data[index_name] == index[i] )]

        # 对于这多行中的每个与variable name相关的列，进行计数，然后按照频率排序，将计数结果写入字典中的list中
        for j in variables_name_list:
            count = Counter(list(row[j]))
            total = sum(count.values())
            sorted_count =  sorted(count.items(),key=lambda item: -item[1])# 按照第二个值进行逆序
            key = [z[0] for z in sorted_count]
            value = [z[1]/total for z in sorted_count]
            #TODO：如果扩展成3个以上，更改这里
            if len(count) >= 3:
                for k in range(3):
                    data_dict[i + "_{}".format(k)].append(key[k])
                    data_dict[i + "_{}_freq".format(k)].append(value[k])
            elif len(count) != 0:

                empty = 3 - len(count)
                for k in range(len(count)):
                    data_dict[i + "_{}".format(k)].append(key[k])
                    data_dict[i + "_{}_freq".format(k)].append(value[k])
                # 后面的从后面索引，添加最后一个值，然后value append 0
                for k1 in range(empty):
                    data_dict[i + "_{}".format(3 - k1)].append(key[-1])
                    data_dict[i + "_{}_freq".format(3 - k1)].append(0)
            else:
                for k2 in range(3):
                    #TODO:对于缺失key的处理，请修改这里
                    data_dict[i + "_{}".format(k)].append(0)
                    data_dict[i + "_{}_freq".format(k)].append(0)


    data_w = pd.DataFrame()
    data_w.insert(0,data_dict_name_l[0],pd.Series(data_dict[data_dict_name_l[0]]))
    for i in range(1,len(data_dict_name_l)):
        data_w.insert(data_w.shape[1],
                      data_dict_name_l[i],
                      pd.Series(data_dict[data_dict_name_l[i]]))
    if save:
        data_w.to_csv(save+".csv",index=False)
def apply_index_three_frequency_table_to_data_1(pd_data,
                                        index_name,
                                        table_file_data,test=False,save="apply_table"):
    '''
    apply table to original data, e.g.

    we have <table_file_data>:
    name a1 a1_freq ...
    1    a  0.5
    2    b  1

    we have <pd_data>:

    name param1
    1    3
    1    5
    2    2
    2    4

    then return:
    name param1 a1 a1_freq ...
    1    3      a   0.5
    1    5      a   0.5
    2    2      b   1
    2    4      b   1

    the reason I do this: some index_like labels such as food1, food2, ..., food20001
    can be replaced with:
        1. other number_like data
        2. index_like labels which have less kinds
        such as replace:
        food1       with    apple, 0.3, pie, 0.3, orange, 0.2 ...
        food2009    with    apple, 0.4, orange, 0.3, pie, 0.2 ...

    用数值型变量去替代标签类的变量：
    在神经网络等机器学习中，不能够直接把index当成数字计算，而应当采用one_hot编码，
    但是如果维度过高很麻烦，所以就选择将这些index用其他数值型变量或者数目更少的标签的
    组合性质去代替，这里采用的是类似朴素贝叶斯的方法。
    和上面一个函数结合，相当于用经常出现的三个属性以及出现的概率，去代替一个属性，比如用
    一类用户的前三种爱好以及概率去代替用户类群这一标签化的数据

    之所以设计这样的方式，是因为有很多用户id数据，每个id有多个行为，而这样的id有几万个，难以onehot，
    于是用这样的用户的高频行为去代替用户id，这样能够区别
    比如：20个用户有200个数据，每个用户10个数据，
    每个数据是每天的行驶数据，之后用每个用户10个数据统计的高频地点，去代替用户的id，于是用户数据变成了：
    每日高频地点，以及每日出发点和目的地，而不是用户id+出发点+目的地，这样能够降维

    :param pd_data:
    :param index_name:
    :param table_file_data:
    :param test:
    :param save:
    :return:
    '''
    # 假设pd_data和table的index_name相同
    dict = {}
    # 默认第一个是index_name其他是值，keys可以获取表头的name
    keys = list(table_file_data.keys())[1:]
    for i in keys:
        dict[i] = []

    data_index = list(pd_data[index_name])
    a = len(data_index)
    if test:
        a = test
    # 用data的name去对应，找到table file的值，然后append
    for i in tqdm.trange(a):
        data = table_file_data[(table_file_data[index_name]==data_index[i])]
        for j in range(len(dict)):
            dict[keys[j]].append(data[keys[j]])
    data_w = pd_data
    for i in range(len(dict)):
        data_w.insert(data_w.shape[1],
                      keys[i],
                      pd.Series(dict[keys[i]]))
    if save:
        data_w.to_csv(save+".csv",index=False)


def get_closest_S_and_E_distance_plan_1(x, start, end,min_number):
    '''
    二维距离测定：返回x 到 start和end的距离最近的前min_number个结果，
    其中start和end是一一对应的，
    支持start和end两个数组，如果离start最近，则返回对应的end相关的结果
    用于起始点和末点的预测，比如，离某个起始点最近，则返回这个起始点对应的结束点
    x为[x,y]
    start和end长度相同，均为(None, 2) [[x1,y1],[x2,y2],...]
    return 的结果最好自定义

    '''
    aim_location = x
    start_loc_l = start
    end_loc_l = end
    number = len(start_loc_l)

    result = []
    for i in tqdm.trange(number):
        location = start_loc_l[i]
        result.append((aim_location[0]-location[0])*(aim_location[0]-location[0])+
                      (aim_location[1] - location[1]) * (aim_location[1] - location[1])
                      )
    for i in range(number):
        location = end_loc_l[i]
        result.append((aim_location[0] - location[0]) * (aim_location[0] - location[0]) +
                      (aim_location[1] - location[1]) * (aim_location[1] - location[1])
                      )
    result_ = np.array(result).argsort()[:min_number][::1]# 一共三个数值，对应最小的三个值
    result = []
    for i in result_:
        if i < number:
            # 距离start区域最近的，得到相应的end
            # 这里可以更改，添加其他与二维location一一对应的list，比如地点名称，用来得到result
            result.append(end_loc_l[i])
        else:
            result.append(start_loc_l[i-number])
    return result

def get_closest_distance(aim_loc, all_location,min_number):
    '''
    return the first min_number closest index in all_location for aim_loc
    :param aim_loc:         (2)
    :param all_location:    (None, 2)
    :param min_number:      int
    :return:                (None, min_number)
    '''
    aim_location = aim_loc

    location = all_location
    return_result = []
    result = []
    a = len(location)

    for i in range(a):

        result.append(
            (
                (aim_location[0]-location_[0])*(aim_location[0]-location_[0])+
             (aim_location[1]-location_[1])* (aim_location[1]-location_[1])
            )
        )

    result_ = np.array(result).argsort()[:min_number][::1]
    return_result.append(result_)
    return return_result

def dict_predict_model_train(x_train_lists,y_train_lists,dict_model_save_name,return_result=False):
    '''
    字典预测模型，将自变量以字符串的形式组合成字典的key，将因变量加入字典key对应的value的list中，比如：

    x1      x2      y
    apple   hot     1
    apple   water   3
    pie     fruit   1
    apple   hot     2

    建立的字典：
    {
    "apple#hot":[1,2],
    "apple#water":[3],
    "pie#fruit":[1],...
    }
    之后对模型进行预测时，就相当于是查找字典，比如对
    apple   hot进行预测，就能够得到1或2

    这种方法可以排列组合作为key，有时候也能够通过改变数值的离散程度来减少键值，比如
    将 12:03 12:04 12:56归为 12
    将 11:08 11:45 归为 11
    这样能够有所聚类

    关键优势：运算快！能够处理字符类型的变量，比如字符非常多时，采用NB很容易内存溢出

    :param x_lists: (None1, None2)
    :param y_lists: (None1, None3)
    :return:

    注意，按照习惯，list的第一维应当是列数，第二维是行数，比如shape为500,3的array，转化成list
    就是3,500，

    return ： 将这个dict存储，也可return结果

    '''
    dict = {}
    if not (isinstance(x_train_lists,list) and isinstance(y_train_lists,list)):
        raise ValueError("x_train_lists and y_train_lists must be list")
    length = len(x_train_lists[0])
    dim_x = len(x_train_lists)
    dim_y = len(y_train_lists)
    for i in range(length):
        str_x = ""
        str_y = ""
        for j in range(dim_x):
            str_x += x_train_lists[j][i]
        for z in range(dim_y):
            str_y += y_train_lists[z][i]
        try:
            dict[str_x].append(str_y)
        except:
            dict[str_x] = []
            dict[str_x].append(str_y)

    with open(dict_model_save_name,"wb") as f:
        pickle.dump(dict,f)

    if return_result:
        return dict

def dict_load_and_summary(filename):

    with open(filename, 'rb') as f:
        dict = pickle.load(f)
    number = 0
    for i in dict.values():
        number += len(i)
    print("平均一个字典有{}个值".format(number / len(dict)))
    return dict
