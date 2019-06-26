


from __future__ import print_function
import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import time
from torch import autograd
from PIL import ImageFile
import pdb


# ============================ Data & Networks =====================================
from dataset.datasets import 
miniImageNet_ravi
import models.network as Net
# ==================================================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./miniImageNet--ravi_simple')
parser.add_argument('--data_name', default='miniImageNet_ravi')
parser.add_argument('--mode', default='train', help='train|val|test')
parser.add_argument('--outf', default='./results/TempNet')
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--basemodel', default='relationNet_64', help='relationNet|alexnetNet')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--episodeSize', type=int, default=1, help='the mini-batch size of training')
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--episode_test_num', type=int, default=10000, help='how many times of the testing, testing every 10000 episodes')
parser.add_argument('--episode_num', type=int, default=300000, help='the number of episodes')
parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--shot_num', type=int, default=5, help='the number of shot')
parser.add_argument('--query_num', type=int, default=15, help='the number of query')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')

np.random.seed(15)
torch.manual_seed(15)
torch.cuda.manual_seed(15)
opt = parser.parse_args()
opt.cuda = True
cudnn.benchmark = True


# save path
opt.outf = opt.outf+'_'+opt.data_name+'_'+str(opt.basemodel)+'_'+str(opt.way_num)+'_Way_'+str(opt.shot_num)+'_Shot'

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# save the opt and results to txt file
txt_save_path = os.path.join(opt.outf, 'opt_resutls.txt')
F_txt = open(txt_save_path, 'a+')


# ======================================== Folder of Datasets ==========================================

# image transform & normalization
ImgTransform = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

if opt.data_name == 'miniImageNet_ravi':
   
    trainset = miniImageNet_ravi(
        data_dir=opt.dataset_dir, mode=opt.mode, image_size=opt.imageSize, transform=ImgTransform,
        episode_num=opt.episode_num, way_num=opt.way_num, shot_num=opt.shot_num, query_num=opt.query_num
    )
    valset = miniImageNet_ravi(
        data_dir=opt.dataset_dir, mode='val', image_size=opt.imageSize, transform=ImgTransform,
        episode_num=600, way_num=opt.way_num, shot_num=opt.shot_num, query_num=opt.query_num
    )
    testset = miniImageNet_ravi(
        data_dir=opt.dataset_dir, mode='test', image_size=opt.imageSize, transform=ImgTransform,
        episode_num=600, way_num=opt.way_num, shot_num=opt.shot_num, query_num=opt.query_num
    )



print('Trainset: %d' %len(trainset))
print('Valset: %d' %len(valset))
print('Testset: %d' %len(testset))
print('Trainset: %d' %len(trainset), file=F_txt)
print('Valset: %d' %len(valset), file=F_txt)
print('Testset: %d' %len(testset), file=F_txt)



# ========================================== Load Datasets ==============================================
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=opt.episodeSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=True
    )
val_loader = torch.utils.data.DataLoader(
    valset, batch_size=opt.testepisodeSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=True
    ) 
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=opt.testepisodeSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=True
    ) 
print(opt)
print(opt, file=F_txt)



# ========================================== Model config ===============================================
ngpu = int(opt.ngpu)
global best_prec1, episode_train_index, best_prec2
best_prec1 = 0
best_prec2 = 0
episode_train_index = 0
model = Net.define_Net(which_model=opt.basemodel, num_classes=opt.way_num, norm='batch', 
    init_type='normal', use_gpu=opt.cuda)
global temprature_inc_rate, temprature_init,temprature
temprature_init = 1  # init temprature as 1 increase every 50000 iter
temprature_inc_rate = 100000
temprature = 1
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))


# optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        episode_train_index = checkpoint['episode_train_index']+1
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (episode {})".format(opt.resume, checkpoint['episode_train_index']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

if opt.ngpu > 1:
    model = nn.DataParallel(model, range(opt.ngpu))

print(model) 
print(model, file=F_txt) # print the architecture of the network



# ======================================= Define functions =============================================
def reset_grad():
    model.zero_grad()


def adjust_learning_rate(optimizer, episode_num):
    """Sets the learning rate to the initial LR decayed by 10 every 100000 episodes"""
    lr = opt.lr * (0.1 ** (episode_num // 100000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_temprature(temprature, episode_num):
#     """Sets the learning rate to the initial LR decayed by 10 every 100000 episodes"""
#     lr = temprature * (2 ** (episode_num // 100000))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
shot_num = opt.shot_num

def train(train_loader, val_loader, test_loader, model, criterion, optimizer, best_prec1, episode_train_index, F_txt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(train_loader, episode_train_index):
        
        # with autograd.detect_anomaly():
        
        adjust_learning_rate(optimizer, episode_index)
        temprature_P = 10 * (0.5 ** (episode_index // temprature_inc_rate))
        temprature_N = 10 * (1.5 ** (episode_index // temprature_inc_rate))
            # measure data loading time
        data_time.update(time.time() - end)

            # convert query and support images
        query_images = torch.cat(query_images, 0)
        input_var1 = query_images.cuda()

        input_var2 = []
        for i in range(len(support_images)):
            temp_support = support_images[i]
            temp_support = torch.cat(temp_support, 0)
            temp_support = temp_support.cuda()
            input_var2.append(temp_support)

            # deal with the target
        target = torch.cat(query_targets, 0)
        target = target.cuda(non_blocking=True)
            # with autograd.detect_anomaly():
            # compute output
        output = model(input_var1, input_var2, temprature_P, temprature_N, target, shot_num)
        loss = criterion(output, target)

                # measure accuracy and record loss
        prec1, prec3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), query_images.size(0))
        top1.update(prec1[0], query_images.size(0))
        top3.update(prec3[0], query_images.size(0))

                # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if ((top1.val == 20.000) and (top3.val == 60.000)):
        #     print(1)
        if episode_index % opt.print_freq == 0:

            print('Eposide: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                        episode_index, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top3=top3))

            print('Episode: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                        episode_index, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top3=top3), file=F_txt)



        #============== evaluate on validation/test set ==============#
        if episode_index % opt.episode_test_num == 0 and episode_index > 0:


            print('============ Validation in the val ============')
            print('============ validation in the val ============', file=F_txt)
            # output = model(input_var1, input_var2, temprature_P, temprature_N, target)

            prec1 = validate(val_loader, model, criterion, episode_index, best_prec1, F_txt,temprature_P, temprature_N,shot_num)


            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # save the checkpoint
            filename = os.path.join(opt.outf, 'episode_%d.pth.tar' %episode_index)
            save_checkpoint(
                {
                    'episode_train_index': episode_index,
                    'arch': opt.basemodel,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, filename)

            
            # Testing Prase
            print('============ Testing in the test ============')
            print('============ Testing in the test ============', file=F_txt)
            prec1 = validate(test_loader, model, criterion, episode_index, best_prec1, F_txt,temprature_P, temprature_N,shot_num)
            if is_best:
                print('best test: {test:.3f},best val: {val:.3f}'.format(test=prec1, val=best_prec1))
                best_prec2 = prec1
            else:
                print('best test: {test:.3f},best val: {val:.3f}'.format(test=best_prec2, val=best_prec1))



def validate(val_loader, model, criterion, episode_train_index, best_prec1, F_txt,temprature_P,temprature_N,shot_num):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):


            # convert query and support images
            query_images = torch.cat(query_images, 0)
            input_var1 = query_images.cuda()

            input_var2 = []
            for i in range(len(support_images)):
                temp_support = support_images[i]
                temp_support = torch.cat(temp_support, 0)
                temp_support = temp_support.cuda()
                input_var2.append(temp_support)

            # deal with the target
            target = torch.cat(query_targets, 0)
            target = target.cuda(non_blocking=True)
            # target_var = target.cuda()

            # compute output
            output = model(input_var1, input_var2,temprature_P, temprature_N, target,shot_num)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), query_images.size(0))
            top1.update(prec1[0], query_images.size(0))
            top3.update(prec3[0], query_images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if episode_index+1 % opt.print_freq == 0:
                print('Test-{0}: [{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                        episode_train_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top3=top3))

                print('Test-{0}: [{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                        episode_train_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top3=top3), file=F_txt)

        
        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Best_Prec@1 {best:.3f}'.format(top1=top1, top3=top3, best=best_prec1))
        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Best_Prec@1 {best:.3f}'.format(top1=top1, top3=top3, best=best_prec1), file=F_txt)
    
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        file_model_best = os.path.join(opt.outf, 'model_best.pth.tar')
        shutil.copyfile(filename, file_model_best)


def save_checkpoint2(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



# ============================================ Training phase ========================================
print('start training.........')
start_time = time.time()


# train for 100,0000 episodes
train(train_loader, val_loader, test_loader, model, criterion, optimizer, best_prec1, episode_train_index, F_txt)
F_txt.close()

# ============================================ Training End ==========================================
