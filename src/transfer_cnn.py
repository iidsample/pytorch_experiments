import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

training_param_dict = {'data' : './', 'arch' : 'resnet18', 'workers' : 4,
                       'epochs' : 90, 'start_epoch' : 0, 'batch_size': 256,
                       'lr' : 0.1, 'momentum' : 0.9,
                       'weight_decay':1e-4, 'print_freq' : 100, 'resume' : '',
                       'pretrained': True, 'validate': True }
best_prec1 = 0
def main():
    global training_param_dict, best_prec1
    if training_param_dict.get('pretrained'):
        print ("Using pre-trained model{}".format(
            training_param_dict.get('arch')))
        model = models.resnet18(pretrained=True)
    else:
        # we will implement it later on
        pass
    # specific resnet will fix for the rest later
    model.features = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                training_param_dict.get('lr'),
                                momentum=training_param_dict.get('momentum'),
                                weight_decay=training_param_dict.get('weight_decay'))
    # loading from a checkpoint file later

    cudnn.benchmark = True
    traindir = os.path.join(training_param_dict.get(
        '/media/saurabh/Extra_1/training_data/macy_training/') , 'train')
    valdir = os.path.join(training_param_dict.get(
        '/media/saurabh/Extra_1/training_data/macy_training'),'valid')
    data_transform = transforms.Compose(
                        [transforms.Scale((256,256)),transforms.CenterCrop(224),
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485,
                                                                         0.456,
                                                                         0.406],
                                                                   std=[0.229,
                                                                        0.224,
                                                                        0.225])])
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir,
                            data_transform),
                            batch_size=training_param_dict.get('batch_size'),
                            num_workers=training_param_dict.get('workers'),
                            pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            dataset.ImageFolder(valdir, data_transform),
                                batch_size=training_param_dict.get('batch_size'),
                                num_workers=training_param_dict.get('workers'),
                                pin_memory=True)
    if training_param_dict.get('validate'):
        #TODO: Nedd to write the validate function
        validate(val_loader, model, criterion)
    for epoch in range(training_param_dict.get('start_epoch'),
                       training_param_dict.get('epoch')):
        adjust_learning_rate(optimizer, epoch)
        # training for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch' : epoch + 1,
            'arch' : training_param_dict.get('arch'),
            'state_dict' : model.state_dict(),
            'best_prec1' : best_prec1,
            'optimizer' : optimizer.state_dict()
        }, is_best)

def train(train_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    for i, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(data)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # gradient compute
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time= time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg






class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch):
    lr = training_param_dict.get('lr') * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.pth.tar')
if __name__ == '__main__':
    main()
