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
import torch.nn as nn
from pathlib import Path
# from Learner_test import *
import pickle
from Permenate.conxy import Conxy
import os
import torch.distributed as dist


def parameter_num_cal(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("Paras:" + str(k))


class face_learner(object):
    def __init__(self, conf, inference=False, embedding_size=512):
        conf.embedding_size = embedding_size
        print(conf)


        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).cuda()
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).cuda()
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        parameter_num_cal(self.model)


        self.milestones = conf.milestones
        self.loader, self.class_num = get_train_loader(conf)
        self.step = 0
        self.agedb_30, self.cfp_fp, self.lfw, self.calfw, self.cplfw, self.vgg2_fp, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame, self.calfw_issame, self.cplfw_issame, self.vgg2_fp_issame = get_val_data(self.loader.dataset.root.parent)
        self.writer = SummaryWriter(conf.log_path)

        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0

            if conf.multi_sphere:
                if conf.arcface_loss:
                    self.head = ArcfaceMultiSphere(embedding_size=conf.embedding_size, classnum=self.class_num, num_shpere=conf.num_sphere, m=conf.m).to(conf.device)
                elif conf.am_softmax:
                    self.head = MultiAm_softmax(embedding_size=conf.embedding_size, classnum=self.class_num, num_sphere=conf.num_sphere, m = conf.m).to(conf.device)
                else:
                    self.head = MultiSphereSoftmax(embedding_size=conf.embedding_size, classnum=self.class_num, num_sphere=conf.num_sphere).to(conf.device)

            else:
                    if conf.arcface_loss:
                        self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
                    elif conf.am_softmax:
                        self.head = Am_softmax(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
                    else:
                        self.head = Softmax(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)


            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_mobilfacenet:
                if conf.multi_sphere:
                    self.optimizer = optim.SGD([
                        {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                        {'params': [paras_wo_bn[-1]] + self.head.kernel_list, 'weight_decay': 4e-4},
                        {'params': paras_only_bn}
                    ], lr=conf.lr, momentum=conf.momentum)
                else:
                    self.optimizer = optim.SGD([
                                        {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                        {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                        {'params': paras_only_bn}
                                    ], lr = conf.lr, momentum = conf.momentum)
            else:
                if conf.multi_sphere:
                    self.optimizer = optim.SGD([
                        {'params': paras_wo_bn + self.head.kernel_list, 'weight_decay': 5e-4},
                        {'params': paras_only_bn}
                    ], lr=conf.lr, momentum=conf.momentum)
                else:
                    self.optimizer = optim.SGD([
                                        {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                        {'params': paras_only_bn}
                                    ], lr = conf.lr, momentum = conf.momentum)


            print(self.optimizer)

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')
            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//100
            self.save_every = len(self.loader)//5
            self.agedb_30, self.cfp_fp, self.lfw, self.calfw, self.cplfw, self.vgg2_fp, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame, self.calfw_issame, self.cplfw_issame, self.vgg2_fp_issame = get_val_data(self.loader.dataset.root.parent)
        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))

    def get_new_state(self, path):
        state_dict = torch.load(path)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        return new_state_dict

    def load_state(self, save_path, fixed_str, model_only=False):
        self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))

        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
            print(self.optimizer)
        
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor, angle_info):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        self.writer.add_scalar('{}_same_pair_angle_mean'.format(db_name), angle_info['same_pair_angle_mean'], self.step)
        self.writer.add_scalar('{}_same_pair_angle_var'.format(db_name), angle_info['same_pair_angle_var'], self.step)
        self.writer.add_scalar('{}_diff_pair_angle_mean'.format(db_name), angle_info['diff_pair_angle_mean'], self.step)
        self.writer.add_scalar('{}_diff_pair_angle_var'.format(db_name), angle_info['diff_pair_angle_var'], self.step)


    def evaluate(self, conf, carray, issame, nrof_folds = 10, tta = False, n=1, show_angle=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size//n])
        i = 0
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()[:, i*conf.embedding_size//n:(i+1)*conf.embedding_size//n]
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()[:, i*conf.embedding_size//n:(i+1)*conf.embedding_size//n]
        tpr, fpr, accuracy, best_thresholds, angle_info= evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor, angle_info
    
    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            if conf.multi_sphere:
                loss = conf.ce_loss(thetas[0], labels)
                for theta in thetas[1:]:
                    loss = loss + conf.ce_loss(theta, labels)
            else:
                loss = conf.ce_loss(thetas, labels)

            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses    

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.



        logging.basicConfig(filename=conf.log_path/'log.txt',
                            level=logging.INFO,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s",
                            datefmt = '%Y-%m-%d  %H:%M:%S %a')
        logging.info('\n******\nnum of sphere is: {},\nnet is: {},\ndepth is: {},\nlr is: {},\nbatch size is: {}\n******'
                     .format(conf.num_sphere, conf.net_mode, conf.net_depth, conf.lr, conf.batch_size))
        for e in range(epochs):
            print('epoch {} started,all is {}'.format(e, epochs))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()


            for imgs, labels in tqdm(iter(self.loader)):
                self.model.train()

                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)

                if conf.multi_sphere:
                    loss = conf.ce_loss(thetas[0], labels)
                    for theta in thetas[1:]:
                        loss = loss + conf.ce_loss(theta, labels)
                else:
                    loss = conf.ce_loss(thetas, labels)

                running_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor, angle_info = self.evaluate(conf, self.agedb_30, self.agedb_30_issame,show_angle=True)
                    print('age_db_acc:', accuracy)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor, angle_info)
                    logging.info('agedb_30 acc: {}'.format(accuracy))

                    accuracy, best_threshold, roc_curve_tensor, angle_info = self.evaluate(conf, self.lfw, self.lfw_issame)
                    print('lfw_acc:', accuracy)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor,angle_info)
                    logging.info('lfw acc: {}'.format(accuracy))

                    accuracy, best_threshold, roc_curve_tensor, angle_info = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    print('cfp_acc:', accuracy)
                    self.board_val('cfp', accuracy, best_threshold, roc_curve_tensor, angle_info)
                    logging.info('cfp acc: {}'.format(accuracy))


                    accuracy, best_threshold, roc_curve_tensor, angle_info = self.evaluate(conf, self.calfw, self.calfw_issame)
                    print('calfw_acc:', accuracy)
                    self.board_val('calfw', accuracy, best_threshold, roc_curve_tensor,angle_info)
                    logging.info('calfw acc: {}'.format(accuracy))


                    accuracy, best_threshold, roc_curve_tensor, angle_info = self.evaluate(conf, self.cplfw, self.cplfw_issame)
                    print('cplfw_acc:', accuracy)
                    self.board_val('cplfw', accuracy, best_threshold, roc_curve_tensor, angle_info)
                    logging.info('cplfw acc: {}'.format(accuracy))


                    accuracy, best_threshold, roc_curve_tensor, angle_info = self.evaluate(conf, self.vgg2_fp,
                                                                               self.vgg2_fp_issame)
                    print('vgg2_acc:', accuracy)
                    self.board_val('vgg2', accuracy, best_threshold, roc_curve_tensor, angle_info)
                    logging.info('vgg2_fp acc: {}'.format(accuracy))


                    self.model.train()
                self.step += 1


    def schedule_lr(self):
        for params in self.optimizer_corr.param_groups:
            params['lr'] /= 10
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               