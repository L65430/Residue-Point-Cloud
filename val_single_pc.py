import config
import torch
import os.path as osp
from utils import meter
from torch import nn
from torch import optim
from models import DGCNN
from torch.utils.data import DataLoader
from datasets import *
import numpy as np

def validate(val_loader, net, epoch):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    retrieval_map = meter.RetrievalMAPMeter()

    # testing mode
    net.eval()

    total_seen_class = [0 for _ in range(40)]
    total_right_class = [0 for _ in range(40)]

    for i, (pcs, labels) in enumerate(val_loader):
        batch_time.reset()


        pcs = pcs.to(device=config.device)
        labels = labels.to(device=config.device)

        preds, fts = net(pcs, get_fea=True)  # bz x C x H x W

        # prec.add(preds.data, labels.data)

        prec.add(preds.data, labels.data)
        retrieval_map.add(fts.detach()/torch.norm(fts.detach(), 2, 1, True), labels.detach())
        for j in range(pcs.size(0)):
            total_seen_class[labels.data[j]] += 1
            total_right_class[labels.data[j]] += (np.argmax(preds.data.cpu(),1)[j] == labels.cpu()[j])


        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t'
                  f'Mean Class accuracy{(np.mean(np.array(total_right_class)/np.array(total_seen_class,dtype=np.float)))}')

    mAP = retrieval_map.mAP()
    print(f' instance accuracy at epoch {epoch}: {prec.value(1)} ')
    print(f' mean class accuracy at epoch {epoch}: {(np.mean(np.array(total_right_class)/np.array(total_seen_class,dtype=np.float)))} ')
    print(f' map at epoch {epoch}: {mAP} ')
    return prec.value(1), mAP




def main():
    print('Training Process\nInitializing...\n')
    config.init_env()

    val_dataset = data_pth.pc_data(config.pc_net.data_root, status=STATUS_TEST)


    val_loader = DataLoader(val_dataset, batch_size=config.pc_net.validation.batch_sz,
                            num_workers=config.num_workers, shuffle=True, drop_last=True)


    # create model
    net = DGCNN()
    net = torch.nn.DataParallel(net)
    net = net.to(device=config.device)
    optimizer_all = optim.SGD(net.parameters(), config.pc_net.train.lr,
                              momentum=config.pc_net.train.momentum,
                              weight_decay=config.pc_net.train.weight_decay)

    print(f'loading pretrained model from {config.pc_net.ckpt_file}')
    checkpoint = torch.load(config.pc_net.ckpt_file)
    state_dict = checkpoint['model']
    # net.module.load_state_dict({k[7:]: v for k, v in state_dict.items()})
    net.module.load_state_dict(state_dict)
    optimizer_all.load_state_dict(checkpoint['optimizer'])
    best_prec1 = checkpoint['best_prec1']
    resume_epoch = checkpoint['epoch']

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_all, 5, 0.5)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=config.device)

    # for p in net.module.feature.parameters():
    #     p.requires_grad = False

    with torch.no_grad():
        prec1, Map = validate(val_loader, net, resume_epoch)

    print('curr accuracy: ', prec1)
    print('best accuracy: ', best_prec1)
    print('best epoch: ',Map)




if __name__ == '__main__':
    main()

