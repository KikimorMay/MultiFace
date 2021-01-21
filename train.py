from config import get_config
from Learner import face_learner
import argparse
from pathlib import Path
import os
import torch
import torch.distributed as dist

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('--num_sphere', help='the num of multi_sphere', default=1, type=int)
    parser.add_argument("--arcface_loss", action='store_true', default=False, help='use arcface_loss')
    parser.add_argument("--am_softmax_loss", action='store_true', default=False, help='use am_softmax_loss')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, ir_se_new, mobilefacenet]",default='mobilefacenet', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=18, type=int)
    parser.add_argument('-lr','--lr',help='learning rate', default=0.05, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=180, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database ",default='emore', type=str)
    parser.add_argument("--work_path" ,default='work_path', type=str)
    parser.add_argument("--m" ,help='the margin', default=0.3, type=float)
    parser.add_argument("--drop_ratio", help='the drop_ratio', default=0.6, type=float)
    parser.add_argument("--pretrain", action='store_true', default=False, help='load_pretrain_model')
    parser.add_argument("--pretrained_model_path", help="the pretrained model path", default='/hdd2/xujing/project/megaface2/models/arc2_100layer_644000.pth', type=str)



    args = parser.parse_args()

    conf = get_config()


    if conf.num_sphere > 1:
        conf.multi_sphere = True

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
        conf.net_mode = 'mobilefacenet'
        conf.net_depth = None
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth

    # choose arcface loss
    if args.arcface_loss:
        conf.arcface_loss = True
        conf.m = 0.5
        if conf.multi_sphere:
            conf.m = args.m

    # choose am_softmax loss(cosface loss)
    if args.am_softmax_loss:
        conf.am_softmax = True
        conf.m = 0.35
        if args.multi_sphere:
            conf.m = args.m

    conf.num_sphere = args.num_sphere

    if args.pretrain:
        conf.pretrain = True
        assert len(args.pretrained_model_path)
        conf.pretrained_model_path = args.pretrained_model_path


    conf.drop_ratio = args.drop_ratio
    conf.m = args.m
    conf.lr = args.lr
    conf.work_path = Path(args.work_path)
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    if os.path.exists(conf.work_path):
        pass
    else:
        os.mkdir(conf.work_path)
        os.mkdir(conf.model_path)
        os.mkdir(conf.save_path)
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    # if conf.data_mode == 'webface':
    #     conf.epochs = 55
    #     conf.milestones = [30, 38, 45]


    learner = face_learner(conf, embedding_size=512)
    learner.train(conf, args.epochs)
