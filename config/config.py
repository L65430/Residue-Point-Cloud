import os.path as osp
import models

# configuration file
description  = 'multi_pc_net'
version_string  = '0.1'

# device can be "cuda" or "gpu"
device = 'cuda'
num_workers = 4
available_gpus = '4,5,6,7'
# available_gpus = '0,2,3'
print_freq = 15

result_root = '/home/ym/projects/result'
result_sub_folder = osp.join(result_root, f'{description}_{version_string}_torch')
ckpt_folder = osp.join(result_sub_folder, 'ckpt')

base_model_name = models.ALEXNET
# base_model_name = models.VGG13BN
# base_model_name = models.VGG11BN
# base_model_name = models.RESNET50
# base_model_name = models.INCEPTION_V3




class pc_net:
    data_root = '/repository/ModelNet40_npy'
    # data_root = '/home/fengyifan/data/pc'
    n_neighbor = 20
    num_classes = 40
    pre_trained_model = None
    # ckpt_file = osp.join(ckpt_folder, 'PCNet-ckpt.pth')
    ckpt_file = osp.join(ckpt_folder, 'nochange-pc-release-v1-ckpt.pth')
    ckpt_record_folder = osp.join(ckpt_folder, 'single-pc-record')

    class train:
        batch_sz = 24*4
        resume = False
        resume_epoch = None

        lr = 0.001
        momentum = 0.9
        weight_decay = 0
        max_epoch = 250
        data_aug = True

    class validation:
        batch_sz = 32

    class test:
        batch_sz = 32


class view_net:
    num_classes = 40

    # multi-view cnn
    data_root = '/home/youhaoxuan/data/12_ModelNet40'
    # data_root = '/repository/12_ModelNet40'

    pre_trained_model = None
    if base_model_name == (models.ALEXNET or models.RESNET50):
        ckpt_file = osp.join(ckpt_folder, f'MVCNN-{base_model_name}-ckpt.pth')
        ckpt_record_folder = osp.join(ckpt_folder, f'MVCNN-{base_model_name}-record')
    else:
        ckpt_file = osp.join(ckpt_folder, f'{base_model_name}-12VIEWS-MAX_POOLING-ckpt.pth')
        ckpt_record_folder = osp.join(ckpt_folder, f'{base_model_name}-12VIEWS-MAX_POOLING-ckpt.pth')


    class train:
        if base_model_name == models.ALEXNET:
            batch_sz = 128 # AlexNet 2 gpus
        elif base_model_name == models.INCEPTION_V3:
            batch_sz = 2
        else:
            batch_sz = 32
        resume = False
        resume_epoch = None

        lr = 0.0001
        momentum = 0.9
        weight_decay = 0
        max_epoch = 200
        data_aug = True

    class validation:
        batch_sz = 256

    class test:
        batch_sz = 32

class multi_pc_net:
    data_root = '/repository/4_ModelNet40_npy'
    # data_root = '/home/fengyifan/data/pc'
    n_neighbor = 20
    num_classes = 40
    pre_trained_model = None
    # ckpt_file = osp.join(ckpt_folder, 'PCNet-ckpt.pth')
    ckpt_file = osp.join(ckpt_folder, '4-pc-release-v1-ckpt.pth')
    ckpt_record_folder = osp.join(ckpt_folder, 'multi-pc-record')

    class train:
        batch_sz = 24
        resume = False
        resume_epoch = None

        lr = 0.001
        momentum = 0.9
        weight_decay = 1e-4
        max_epoch = 200
        data_aug = True

    class validation:
        batch_sz = 24

    class test:
        batch_sz = 16

class pv_net:
    num_classes = 40

    # pointcloud
    pc_root = '/home/youhaoxuan/data/pc'
    n_neighbor = 20

    # multi-view cnn
    view_root = '/home/youhaoxuan/data/12_ModelNet40'

    pre_trained_model = False
    # ckpt_file = osp.join(ckpt_folder, f'PVNet2-{base_model_name}-vsiglogabs_3mlp-ckpt.pth')
    # ckpt_file = osp.join(ckpt_folder, f'PVNet2-{base_model_name}-mean-ckp.pth')
    # ckpt_file = osp.join(ckpt_folder, f'PVNet2-{base_model_name}-pmv-ckpt.pth')
    ckpt_file = osp.join(ckpt_folder, f'PVNet2-{base_model_name}-v9-release-ckpt.pth')
    # ckpt_file = osp.join(ckpt_folder, f'PVNet2-{base_model_name}-v94-ckpt.pth')
    # ckpt_file = osp.join(ckpt_folder, f'PVNet2-{base_model_name}-DyIntraMAF-ckpt.pth')
    ckpt_record_folder = osp.join(ckpt_folder, f'PVNet2-{base_model_name}-record')

    class train:
        # optim = 'Adam'
        optim = 'SGD'
        # batch_sz = 18*2
        batch_sz = 18*5
        batch_sz_res = 5*1
        resume = False
        resume_epoch = None

        iter_train = True

        fc_lr = 0.01
        all_lr = 0.0009
        momentum = 0.9
        weight_decay = 5e-4
        max_epoch = 100
        data_aug = True

    class validation:
        batch_sz = 40

    class test:
        batch_sz = 32
