import config
import torch
import os.path as osp
from utils import meter
from torch import nn
from torch import optim
from models import DGCNN
from models import DGCNN_Multi_Cloud
from torch.utils.data import DataLoader
from datasets import data_pth, STATUS_TRAIN, STATUS_TEST
import numpy as np


def train(train_loader, net, criterion, optimizer, epoch):
    """
    train for one epoch on the training set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    losses = meter.AverageValueMeter()
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    # training mode
    net.train()

    for i, (pcs, labels) in enumerate(train_loader):
        batch_time.reset()
        pcs = pcs.to(device=config.device)
        labels = labels.to(device=config.device)

        preds = net(pcs)  # bz x vm * C x H x W
        loss = criterion(preds, labels)

        prec.add(preds.data, labels.data)
        losses.add(loss.item())  # batchsize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Loss {losses.value()[0]:.4f} \t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    print(f'prec at epoch {epoch}: {prec.value(1)} ')


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

        pcs = pcs.to(device=config.device) #bz*view_number*c*h*w
        labels = labels.to(device=config.device)

        preds,fts = net(pcs,get_fea=True)  # bz x vm * C x H x W

        prec.add(preds.data, labels.data)
        retrieval_map.add(fts.detach(), labels.detach())

        for j in range(pcs.size(0)):
            total_seen_class[labels.data[j]] += 1
            total_right_class[labels.data[j]] += (np.argmax(preds.data.cpu(), 1)[j] == labels.cpu()[j])

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    mAP = retrieval_map.mAP()
    print(f'instance accuracy at epoch {epoch}: {prec.value(1)} ')
    print(
        f'mean class accuracy at epoch {epoch}: {(np.mean(np.array(total_right_class)/np.array(total_seen_class,dtype=np.float)))} ')
    print(f' map at epoch {epoch}: {mAP} ')
    return prec.value(1), mAP


def save_record(epoch, prec1, net: nn.Module):
    state_dict = net.state_dict()
    torch.save(state_dict, osp.join(config.multi_pc_net.ckpt_record_folder, f'epoch{epoch}_{prec1:.2f}.pth'))


def save_ckpt(epoch, best_prec1, net, optimizer, training_conf=config.multi_pc_net):
    ckpt = dict(
        epoch=epoch,
        best_prec1=best_prec1,
        model=net.module.state_dict(),
        optimizer=optimizer.state_dict(),
        training_conf=training_conf
    )
    torch.save(ckpt, config.multi_pc_net.ckpt_file)


def main():
    print('Training Process\nInitializing...\n')
    config.init_env()

    train_dataset = data_pth.multi_pc_data(config.multi_pc_net.data_root, status=STATUS_TRAIN)
    val_dataset = data_pth.multi_pc_data(config.multi_pc_net.data_root, status=STATUS_TEST)

    train_loader = DataLoader(train_dataset, batch_size=config.multi_pc_net.train.batch_sz,
                              num_workers=config.num_workers,shuffle = True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.multi_pc_net.validation.batch_sz,
                            num_workers=config.num_workers,shuffle=True,drop_last=True)

    best_prec1 = 0
    best_map=0
    resume_epoch = 0
    # create model
    net = DGCNN_Multi_Cloud(n_neighbor=config.multi_pc_net.n_neighbor,num_classes=config.multi_pc_net.num_classes)
    net = torch.nn.DataParallel(net)
    net = net.to(device=config.device)
    optimizer = optim.Adam(net.parameters(), config.multi_pc_net.train.lr,
                          weight_decay=config.multi_pc_net.train.weight_decay)

    if config.multi_pc_net.train.resume:
        print(f'loading pretrained model from {config.multi_pc_net.ckpt_file}')
        checkpoint = torch.load(config.multi_pc_net.ckpt_file)
        net.module.load_state_dict({k[7:]: v for k, v in checkpoint['model'].items()})
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_prec1 = checkpoint['best_prec1']
        if config.multi_pc_net.train.resume_epoch is not None:
            resume_epoch = config.multi_pc_net.train.resume_epoch
        else:
            resume_epoch = checkpoint['epoch'] + 1

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=config.device)

    for epoch in range(resume_epoch, config.multi_pc_net.train.max_epoch):

        lr_scheduler.step(epoch=epoch)
        # train
        train(train_loader, net, criterion, optimizer, epoch)
        # validation
        with torch.no_grad():
            prec1,retrieval_map = validate(val_loader, net, epoch)

        # save checkpoints
        if prec1 > best_prec1:
            best_prec1 = prec1
            save_ckpt(epoch, best_prec1, net, optimizer)
        if retrieval_map > best_map:
            best_map=retrieval_map


        # save_record(epoch, prec1, net.module)
        print('curr accuracy: ', prec1)
        print('best accuracy: ', best_prec1)
        print('best map: ', best_map)

    print('Train Finished!')


if __name__ == '__main__':
    main()

