import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def smooth(list, step):
    ans = []
    for i in range(len(list)):
        if i < step:
            ans.append(list[i])
        else:
            ans.append(np.mean(list[i-step:i]))
    return ans

def grad_backbone(path):
    backbone = {}    #{num_layer:[shape], mean}

    file = open(path, 'rb')
    load_data = pickle.load(file)
    layer = 1
    for index in range(182):
        data = load_data[index+1]
        if len(data.shape) >= 2:
            backbone[layer] = [data.shape]
            backbone[layer].append(np.mean(np.abs(data)))
            layer = layer + 1
    return backbone






def grad_head(path):
    head = {}     # num_head:[total_mean, right_people_mean, wrong_people_mean]

    file = open(path, 'rb')  # 用比特读
    load_data = pickle.load(file)
    label = load_data['label']
    head_grad = [load_data['head1']]
    dim, _ = head_grad[0].shape
    # print(load_data.keys())
    if 'head4' in load_data.keys():
        head_grad.append(load_data['head2'])
        head_grad.append(load_data['head3'])
        head_grad.append(load_data['head4'])
    for index in range(len(head_grad)):
        sum_full = np.sum(np.abs(head_grad[index]), axis=0)
        head[index+1] = [np.mean(np.abs(head_grad[index]))]
        head[index+1].append(np.mean(sum_full[label])/dim)
        head[index+1].append((np.sum(sum_full) - np.sum(sum_full[label]))/(dim*(len(sum_full) - len(label))))
    file.close()

    return head


def head_process(head_dict, head_data):
    if 'total_mean' not in head_dict.keys():
        head_dict['total_mean'] = []
    if 'right_people_mean' not in head_dict.keys():
        head_dict['right_people_mean'] = []
    if 'wrong_people_mean' not in head_dict.keys():
        head_dict['wrong_people_mean'] = []
    head_dict['total_mean'].append(head_data[0])
    head_dict['right_people_mean'].append(head_data[1])
    head_dict['wrong_people_mean'].append(head_data[2])
    return head_dict


def backbone_process(backbone_dict):
    backbone_data = []
    for index, value in backbone_dict.items():
        backbone_data.append(value[1])
    return backbone_data



def draw(data_list, save_path, labels):
    matplotlib.rcParams['figure.figsize'] = [6, 5]
    matplotlib.rcParams['figure.subplot.left'] = 0.2
    matplotlib.rcParams['figure.subplot.bottom'] = 0.2
    matplotlib.rcParams['figure.subplot.right'] = .8
    matplotlib.rcParams['figure.subplot.top'] = 0.8
    fig = plt.figure()
    l = len(data_list)
    x = len(data_list[0])
    x = np.linspace(1, x, x) *0.2
    for i in range(l):
        data_l = []
        for j in data_list[i]:
            data_l.append(100000*j)
        data_l = smooth(data_l, 3)
        plt.plot(x, data_l, label=labels[i])
    save_p = save_path + '.pdf'
    save_p_2 = save_path + '.png'
    plt.legend()
    plt.ylim((0, 15))
    plt.xlim((-0.5, 32))
    plt.legend(fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Gradient value(x10$^{-5}$)", fontsize=15)
    plt.xlabel("# Step(x10$^{3}$)", fontsize=15)
    # plt.savefig(save_path + 'deature_distribute.pdf')
    # plt.savefig(save_path + 'deature_distribute.png')
    fig.savefig(save_p_2)
    fig.savefig(save_p)





backbone_base = {}
backbone_4 = []
head_base ={}
head_4_1 = {}
head_4_2 = {}
head_4_3 = {}
head_4_4 = {}



# for i in range(160):
#     path_base = '4softmax/epoch_'+str(i*200//32349)+'step_'+ str(i*200+200) + '.pickle'
#     # path_4 = '4softmax_long/epoch_'+str(i*1000//32349) +'step_'+ str(i*1000) + '.pickle'
#     # backbone_base.append(grad_backbone(path_base)[50][1])
#     # backbone_4.append(grad_backbone(path_4)[50][1])
#     # head_base_data = grad_head(path_base)
#     # head_4_data = grad_head(path_4)
#     #
#     # head_base = head_process(head_base, head_base_data[1])
#     # head_4_1 = head_process(head_4_1, head_4_data[1])
#     # head_4_2 = head_process(head_4_2, head_4_data[2])
#     # head_4_3 = head_process(head_4_3, head_4_data[3])
#     # head_4_4 = head_process(head_4_4, head_4_data[4])
#     if i % 50 == 0:
#         print('i is:', i)
#     # if i == 0:
#     #     for index in range(50):
#     #         backbone_base[index + 1] = []
#     # backbone_base_data = backbone_process(grad_backbone(path_base))
#     # for index in range(50):
#     #     backbone_base[index+1].append(backbone_base_data[index])
#     head_data = grad_head(path_base)
#     head_4_1 = head_process(head_4_1, head_data[1])
#     head_4_2 = head_process(head_4_2, head_data[2])
#     head_4_3 = head_process(head_4_3, head_data[3])
#     head_4_4 = head_process(head_4_4, head_data[4])

# head_d= [head_4_1['total_mean'], head_4_2['total_mean'], head_4_3['total_mean'], head_4_4['total_mean']]

# file2 = open('4softmax/head.pickle', 'wb')
# pickle.dump(head_d, file2)
# file2.close()



file_name  = '4softmax/head.pickle'
file  = open(file_name, 'rb')    # 用比特读
head_d = pickle.load(file)
file.close()

labels = ['head1', 'head2', 'head3', 'head4']

draw(head_d, save_path='4softmax/gradient', labels=labels)







