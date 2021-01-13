import pickle
import numpy as np
from munkres import Munkres
import sys
import time

def load_pickle(dir_path, num_file):
    index = list(range(num_file))

    for i in index:
        file_name = dir_path + str(i) + '.pickle'
        file = open(file_name, 'rb')  # 用比特读
        load_data = pickle.load(file)
        # 计算不同相似性

        if i == 0:
            load_return = load_data
        else:
            load_return = np.concatenate((load_return, load_data), axis=0)
    return load_return


def cal_dis(emb):
    num, dim = emb.shape
    dim = dim // 2
    emb1, emb2 = emb[:, :dim], emb[:, dim:]


    emb1 = np.random.rand(100000,256)
    emb2 = np.random.rand(100000,256)


    covxy_matrix = cal_covxy_matrix(emb1, emb2)
    s = 0
    for i in range(dim):
        s += covxy_matrix[i][i]
    print('no alined covxy:', s)
    trans_matrix = cal_path(covxy_matrix)
    alined_emb1 = np.dot(emb1, trans_matrix)

    list_l, angle_l = get_dis(emb1, emb2)
    alined_emb1_list_l, alined_emb1_angle_l = get_dis(alined_emb1, emb2)

    print('not alined:   dis mean: ', np.mean(list_l), 'angle_mean: ', np.mean(angle_l))
    print('********')
    print('alined:   dis mean: ', np.mean(alined_emb1_list_l), 'angle_mean: ', np.mean(alined_emb1_angle_l))
    return

def get_dis(emb1, emb2):
    num, dim = emb1.shape
    dis_l = np.zeros(num)
    angle_l = np.zeros(num)
    for i in range(num):
        dis = np.sum(emb1[i, :] * emb2[i, :])
        dis_l[i] = dis
        norm1, norm2 = np.linalg.norm(emb1[i, :]), np.linalg.norm(emb2[i, :])
        dis = dis / (norm1 * norm2)
        angel = np.arccos(dis) / np.pi * 180
        angle_l[i] = angel
    return dis_l, angle_l



def cal_path(covxy_matrix):
    dim, _ = covxy_matrix.shape
    m = Munkres()
    indexes = m.compute(covxy_matrix.max() - covxy_matrix)
    total = 0
    trans_matrix = np.zeros([dim, dim])
    for row, column in indexes:
        trans_matrix[row, column] = 1
        value = covxy_matrix[row][column]
        total += value
        # print('(%d, %d) -> %f' % (row, column, value))
    print('alined: %f' % total)
    return trans_matrix




def cal_covxy_matrix(emb1, emb2):
    # emb shape [num, dim]
    num, dim = emb1.shape

    mean_emb1 = np.mean(emb1, 0)
    mean_emb2 = np.mean(emb2, 0)

    submean_emb1 = emb1 - mean_emb1
    submean_emb2 = emb2 - mean_emb2

    var_emb1 = np.std(emb1, axis=0)
    var_emb2 = np.std(emb2, axis=0)


    covxy_matrix = np.zeros([dim, dim])

    for d in range(dim):
        # emb1的第一个dim与emb2所有dim的协方差，存在 covxy_matrix[1, :] 第一行

        ans = np.dot(submean_emb1[:, d].reshape(1, -1), submean_emb2)/(num*var_emb1[d]*var_emb2)
        covxy_matrix[d, :] = ans

    return covxy_matrix



if __name__ == '__main__':
    dir_path_1 = '/hdd/xujing/project/FacePermenate/Permenate/2MultiSoftmax/emb_'
    dir_path_2 = '/hdd/xujing/project/FacePermenate/Permenate/256Softmax/emb_'
    load_data1 = load_pickle(dir_path_1, 200)
    load_data2 = load_pickle(dir_path_2, 200)

    load_data = np.concatenate((load_data1[:, :256], load_data2), axis=1)

    cal_dis(load_data)

