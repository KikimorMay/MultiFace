from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, Backbone_work, MobileFaceNet, Am_softmax, MultiAm_softmax, l2_norm, ArcfaceMultiSphere, MobileFaceNetSoftmax, Softmax, MultiSphereSoftmax
from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
import logging
import numpy as np
import pickle

def mergeit(matrix):
    pos = torch.nonzero(matrix)
    l = pos.shape[0]
    ans = torch.ones(l)
    for i in range(l):
        a, b = pos[i, :]
        ans[i] = matrix[a, b]
    l = len(ans)
    a = torch.zeros(l)
    index = 0
    for i in range(l):
        a[i] = np.arccos(ans[i])/np.pi*180
        index = index + 1
    return a


def get_non_zero(matrix):
    pos = torch.nonzero(matrix)
    l = pos.shape[0]
    ans = torch.ones(l)
    for i in range(l):
        a, b = pos[i, :]
        ans[i] = matrix[a, b]
    return ans

def get_angel(lst):
    l = len(lst)
    a = torch.zeros(l)
    index = 0
    for i in lst:
        i = i.cpu()
        a[index] = np.arccos(i)/np.pi*180
        index = index + 1
    return a


def cal_distance(embeddings, labels, prelabel=-1, preembedding=None, dis_mean=[], dis_std=[]):
    begin_index = 0
    end_index = 1
    embeddings = embeddings.detach()
    while(end_index < len(labels)):
        last_label = labels[begin_index]
        while end_index < len(labels) and labels[end_index] == labels[begin_index]:
            end_index = end_index + 1
        if prelabel == labels[begin_index]:
            cal_embeddings = torch.cat((preembedding, embeddings[begin_index:end_index, :]), 0)
            print(cal_embeddings.shape)
            dis = torch.triu(torch.mm(cal_embeddings, cal_embeddings.t()), diagonal=1, out=None)
            dis = mergeit(dis)
            dis_mean[-1] = torch.mean(dis).data.detach().numpy()
            dis_std[-1] = torch.std(dis).data.detach().numpy()
        else:
            cal_embeddings = embeddings[begin_index:end_index, :]
            dis = torch.triu(torch.mm(cal_embeddings, cal_embeddings.t()), diagonal=1, out=None)
            # a = get_non_zero(dis)
            # print('1111')
            # print(a.shape)
            # dis = get_angel(a)
            dis = mergeit(dis)
            dis_mean.append(torch.mean(dis).data.detach().numpy())
            dis_std.append(torch.std(dis).data.detach().numpy())
        begin_index = end_index
        end_index = end_index + 1
    return dis_mean, dis_std, last_label, cal_embeddings



def cal_different_distance(embeddings, labels, prelabel=-1, preembedding=None, dis_mean=[], dis_std=[]):
    begin_index = 0
    end_index = 1
    embedding_list = []
    while(end_index < len(labels)):
        last_label = labels[begin_index]
        while end_index < len(labels) and labels[end_index] == labels[begin_index]:
            end_index = end_index + 1
        if prelabel == labels[begin_index]:
            cal_embeddings = torch.cat((preembedding, embeddings[begin_index:end_index, :]), 0)
            embedding_list.append(cal_embeddings)
        else:
            cal_embeddings = embeddings[begin_index:end_index, :]
            embedding_list.append(cal_embeddings)
        begin_index = end_index
        end_index = end_index + 1
    if len(embedding_list) == 3:
        dis_ = torch.mm(embedding_list[0], embedding_list[1].t())
        dis = torch.triu(dis_, diagonal=0, out=None)
        if dis.shape[0] > dis.shape[1]:
            dis = dis.transpose(1, 0)
        dis = mergeit(dis)
        dis_mean.append(torch.mean(dis))
        dis_std.append(torch.std(dis))
    elif len(embedding_list) >= 4:
        dis_ = torch.mm(embedding_list[0], embedding_list[1].t())
        dis = torch.triu(dis_, diagonal=0, out=None)
        if dis.shape[0] > dis.shape[1]:
            dis = dis.transpose(1, 0)
        dis = mergeit(dis)
        dis_mean.append(torch.mean(dis))
        dis_std.append(torch.std(dis))

        dis_ = torch.mm(embedding_list[0], embedding_list[2].t())
        dis = torch.triu(dis_, diagonal=0, out=None)
        if dis.shape[0] > dis.shape[1]:
            dis = dis.transpose(1, 0)
        dis = mergeit(dis)
        dis_mean.append(torch.mean(dis))
        dis_std.append(torch.std(dis))

        dis_ = torch.mm(embedding_list[1], embedding_list[2].t())
        dis = torch.triu(dis_, diagonal=0, out=None)
        if dis.shape[0] > dis.shape[1]:
            dis = dis.transpose(1, 0)
        dis = mergeit(dis)
        dis_mean.append(torch.mean(dis))
        dis_std.append(torch.std(dis))



    return dis_mean, dis_std, last_label, cal_embeddings



#
# embeddings = torch.randn(180, 256)
# labels = torch.zeros(180)
# labels[:100] = 0
# labels[100:] = 1
#
# dis_mean, dis_std, last_label, cal_embeddings = cal_different_distance(embeddings, labels)


def same_pair(embeddings, labels, same_feature=[]):
    begin_index = 0
    end_index = 1
    while (end_index < len(labels)):
        while end_index < len(labels) and labels[end_index] == labels[begin_index]:
            end_index = end_index + 1
        if end_index > begin_index + 2:
            dis = torch.mm(embeddings[begin_index].reshape(1, -1), embeddings[begin_index+1].reshape(-1, 1))
            dis = get_angel(dis)
            same_feature.append(dis)
        begin_index = end_index

    return same_feature


def not_same_pair(embeddings, labels, not_same_feature=[]):
    begin_index = 0
    end_index = 1
    while (end_index < len(labels)):
        last_label = labels[begin_index]
        while end_index < len(labels) and labels[end_index] == labels[begin_index]:
            end_index = end_index + 1
        if end_index < len(labels):
            dis = torch.mm(embeddings[begin_index].reshape(1, -1), embeddings[end_index].reshape(-1, 1))
            dis = get_angel(dis)
            not_same_feature.append(dis)
        begin_index = end_index


    return not_same_feature






if __name__ == '__main__':
    path = '/hdd/xujing/project/FaceNew/distribute/softmax_128_4'
    file = open(path + '/feature_not_same.pickle', 'rb')  # 用比特读
    feature_not_same = pickle.load(file)
    file.close()

    file = open(path + '/feature_same.pickle', 'rb')  # 用比特读
    feature_same = pickle.load(file)
    file.close()


    print(np.mean(feature_same))
    print(np.std(feature_same))
    print(np.mean(feature_not_same))
    print(np.std(feature_not_same))
    print(len(feature_same))
    print(len(feature_not_same))




    file = open(path + '/different_dis_mean.pickle', 'rb')  # 用比特读
    diferent_dis_mean = pickle.load(file)
    file.close()
    dif_mean_list = []
    for i in diferent_dis_mean:
        if i > 0:
            dif_mean_list.append(i)

    file = open(path + '/different_dis_std.pickle', 'rb')  # 用比特读
    diferent_dis_std = pickle.load(file)
    file.close()
    dif_std_list = []
    for i in diferent_dis_std:
        if i > 0:
            dif_std_list.append(i)

    file = open(path + '/dis_mean.pickle', 'rb')  # 用比特读
    dis_mean = pickle.load(file)
    file.close()

    file = open(path + '/dis_std.pickle', 'rb')  # 用比特读
    dis_std = pickle.load(file)
    file.close()

    dis_mean_list = []
    dis_std_list = []
    for i in dis_mean:
        if i > 0:
            dis_mean_list.append(i)
    for i in dis_std:
        if i > 0:
            dis_std_list.append(i)
    print(np.mean(dis_mean_list))
    print(np.mean(dis_std_list))
    print(np.mean(dif_mean_list))
    print(np.mean(dif_std_list))
