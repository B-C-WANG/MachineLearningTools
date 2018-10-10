import numpy as np

'''

why data Replication?
sample number in one label maybe different,
sometimes model learn how to make labels with large number correct,
use some 'tricky' feature
more details about this:

https://wangbch.com/2018/09/26/MachineLearningExperience1/


'''


def replicate_data_of_some_label(X, y, sample_area=None, label_pos=0, times=2):
    '''
    take example,  X shape is (4000,49), y shape is (4000,6)

    if sample_area is 2000, label_pos is 1, times is 2
    then the sample which y[0:2000,1] == 1 will add a copy to original data

    '''
    if times == 1: return
    split_number = y.shape[1]
    total_data = np.concatenate([y, X], axis=1)
    print("Before copy, total data shape: %s" % str(total_data.shape))
    if sample_area == None:
        will_rep_data = total_data[total_data[:, label_pos] == 1]
    else:
        tmp = total_data[:sample_area, :]
        will_rep_data = tmp[tmp[:sample_area, label_pos] == 1]
    print("Copy %s label, which has shape: %s" % (label_pos, str(will_rep_data.shape[1])))
    rep_data = []
    for i in range(times - 1):
        rep_data.append(will_rep_data.copy())
    rep_data = np.concatenate(rep_data, axis=0)
    print("After rep, will add data shape: %s" % str(rep_data.shape))
    total_data = np.concatenate([rep_data, total_data], axis=0)
    print("After rep, total data shape: %s" % str(total_data.shape))
    X = total_data[:, split_number:]
    y = total_data[:, :split_number]
    return X, y


if __name__ == '__main__':
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])

    X1, y1 = replicate_data_of_some_label(X, y, label_pos=0, times=5)
    print(X1)
    print(y1)

    X2, y2 = replicate_data_of_some_label(X,y,sample_area=2,label_pos=0,times=5)
    print(X2)
    print(y2)
