import cv2
from PIL import Image
import argparse
from pathlib import Path
import os
from torch.autograd import Variable

import time
from sklearn.model_selection import KFold
from scipy.io import loadmat
from model import *
import numpy as np
import pickle
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
conf = get_config(False)

# model = MobileFaceNet(conf.embedding_size).to(conf.device)
model = Backbone(num_layers=100, drop_ratio=0.4, mode='ir_se').to(conf.device)
model.load_state_dict(torch.load('/hdd/xujing/project/FaceNew/YTB_models/r_100_640000_None.pth'))
# model.load_state_dict(torch.load('/hdd/xujing/project/FaceNew/work_path_arc2_lr_0.05_margin_0.3/models/model_2019-08-17-04-21_accuracy:0.9278000000000001_step:636000_None.pth'))
save_pickle = Path('/hdd/xujing/project/FaceNew/mobile_arc2_640000.pickle')


model.eval()
DATA_path = Path('/hdd/xujing/dataset/YouTubeFaces/frame_images_DB/')

def mtcnn_process():
    mtcnn = MTCNN()
    print('mtcnn loaded')

    frame = Path('/hdd/xujing/dataset/YouTubeFaces/frame_images_DB/Ted_Williams/2/2.1170.jpg')
    DATA_path = Path('/hdd/xujing/dataset/YouTubeFaces/frame_images_DB/')
    dir_name = os.listdir(DATA_path)
    j = 0

    for name in dir_name:
        if name[-4:] == '.txt':
            continue
        print('name is:', name, 'index is:', j)
        j = j+1
        num_dir = os.listdir(DATA_path/name)
        for num in num_dir:
            pic_dir = os.listdir(DATA_path/name/num)
            faces_list = []
            i = 0
            if not os.path.exists(DATA_path/name/num/'mtcnn'):
                os.mkdir(DATA_path/name/num/'mtcnn')
            for pic in pic_dir:
                if pic =='mtcnn' or pic == 'mtcnn.pickle':
                    continue
                pic_path = DATA_path/name/num/pic
                image = Image.open(pic_path)
                bboxes, faces =  mtcnn.align_multi(image, 1, 30)
                save_path = Path(DATA_path/name/num/'mtcnn'/pic)
                if len(faces) == 0:
                    continue
                faces[0].save(save_path)
                faces_list.append(faces[0])
                i = i+1
            pickle_path = Path(DATA_path/name/num/'mtcnn.pickle')
            if pickle_path.is_file():
                os.remove(pickle_path)
            file = open(pickle_path, 'wb')
            pickle.dump(faces_list, file)
            file.close()

def cal_emb(pickle_path):
    file = open(pickle_path, 'rb')
    load_data = pickle.load(file)
    file.close()
    len_img = len(load_data)
    input = torch.zeros((len_img, 3, 112, 112))
    embs = np.zeros((len_img, conf.embedding_size))
    input_flip = torch.zeros((len_img, 3, 112, 112))
    embs_flip = np.zeros((len_img, conf.embedding_size))

    for i in range(len_img):
        im = np.array(load_data[i])
        im = (im/255.0 - 0.5)/ 0.5
        im = np.transpose(im,(2,0,1))
        im_flip = np.flip(im,2)
        im_flip = np.ascontiguousarray(im_flip, dtype=np.float32)
        input[i,:,:,:] = torch.Tensor(im)
        input_flip[i,:,:,:] = torch.Tensor(im_flip)

    input = input.cuda()
    input_flip = input_flip.cuda()
    batch_size = 50
    start = 0
    while (start + batch_size < len_img):
        embs[start:start+batch_size, :] = model(input[start:start+batch_size,:,:,:]).detach().cpu().numpy()
        embs_flip[start:start+batch_size, :] = model(input_flip[start:start+batch_size,:,:,:]).detach().cpu().numpy()
        start = start + batch_size

    embs[start:, :] = model(input[start:,:,:,:].to(conf.device)).detach().cpu().numpy()
    embs_flip[start:, :] = model(input_flip[start:,:,:,:].to(conf.device)).detach().cpu().numpy()
    return embs, embs_flip

def compare_two(pickle_1, pickle_2):
    embs_1, embs_1_flip = cal_emb(pickle_1)
    embs_2, embs_2_flip = cal_emb(pickle_2)
    similar = np.matmul(embs_1, embs_2.T)
    similar_flip = np.matmul(embs_1_flip, embs_2_flip.T)
    # print(np.mean(similar), np.mean(similar_flip))
    return np.mean(similar) + np.mean(similar_flip)


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def acc():
    m = loadmat('/hdd/xujing/dataset/YouTubeFaces/meta_data/meta_and_splits.mat')
    num_pair = 5000
    groups = 10
    nrof_folds = 10
    thresholds = np.arange(0, 2, 0.005)
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(num_pair)


    dist = []
    actual_issame = []
    time_start = time.time()
    for g in range(groups):
        print('group is:', g)
        for i in range(250):
            i = i+250
            if i%50 == 0:
                time_end = time.time()
                print('time cost', time_end - time_start)
                print('process group is:', g, 'i is:', i)
            index1, index2, is_same = m['Splits'][i, :, g]
            path_1 = DATA_path / str(m['video_names'][index1-1][0])[2:-2] / 'mtcnn.pickle'
            path_2 = DATA_path / str(m['video_names'][index2-1][0])[2:-2] / 'mtcnn.pickle'
            dis = compare_two(path_1, path_2)
            dist.append(dis)

            if dis > 0.3:
                print('not same :', dis)
                print(str(m['video_names'][index1-1][0])[2:-2], str(m['video_names'][index2-1][0])[2:-2])
            # elif dis < 0.3:
            #     print(dis)
            #     print(str(m['video_names'][index1-1][0])[2:-2], str(m['video_names'][index2-1][0])[2:-2])

            if is_same == 1:
                actual_issame.append(True)
            else:
                actual_issame.append(False)

    file = open(save_pickle, 'wb')  # 用比特写
    pickle.dump(dist, file)
    file.close()


    # file = open(save_pickle, 'rb')  # 用比特读
    # dist = pickle.load(file)
    # print(len(dist))
    # file.close()

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, np.array(dist)[train_set],np.array(actual_issame)[train_set])
        best_threshold_index = np.argmax(acc_train)
        #         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,np.array(dist)[test_set],
                                                                                                 np.array(actual_issame)[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], np.array(dist)[test_set],
                                                      np.array(actual_issame)[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds

# c = cal_emb('/hdd/xujing/dataset/YouTubeFaces/frame_images_DB/Victoria_Clarke/5/mtcnn.pickle')

# print(c)


if __name__ == '__main__':
    thresholds = np.arange(0, 1, 0.001)
    tpr, fpr, accuracy, best_thresholds = acc()
    print(accuracy)
    print(accuracy.mean())
    print(best_thresholds)



#
# import numpy as np
# from torch.autograd import Variable
# import torch
# import cv2
# from model import *
# model_path = '/hdd/xujing/project/FaceNew/YTB_models/r_100_640000_None.pth'
# path1 = '/hdd/xujing/dataset/YouTubeFaces/frame_images_DB/Aaron_Eckhart/0/mtcnn/0.556.jpg'
# path2 = '/hdd/xujing/dataset/YouTubeFaces/frame_images_DB/Aaron_Guiel/5/mtcnn/5.1841.jpg'
# model = Backbone(100,0.4,'ir_se')
# model.load_state_dict(torch.load(model_path))
# model = model.cuda()
# model.eval()
#
# im = cv2.imread(path1)
# im = (im/255.0 - 0.5)/ 0.5
# im = np.transpose(im,(2,0,1))
# im_flip = np.flip(im,2)
#
# inputs = np.zeros((2,3,112,112))
# inputs[0] = im
# inputs[1] = im_flip
#
# data = Variable(torch.Tensor(inputs))
# data = data.cuda()
# feat1 = model(data).data.cpu().numpy()  # 2 * 512
#
#
# im = cv2.imread(path2)
# im = (im/255.0 - 0.5)/ 0.5
# im = np.transpose(im,(2,0,1))
# im_flip = np.flip(im,2)
#
# inputs = np.zeros((2,3,112,112))
# inputs[0] = im
# inputs[1] = im_flip
#
# data = Variable(torch.Tensor(inputs))
# data = data.cuda()
# feat2 = model(data).data.cpu().numpy()  # 2 * 512
#
# s1 = (feat1[0] * feat2[0]).sum()
# s2 = (feat1[1] * feat2[1]).sum()
#
# print(s1,s2)
#
#
#
#
#

