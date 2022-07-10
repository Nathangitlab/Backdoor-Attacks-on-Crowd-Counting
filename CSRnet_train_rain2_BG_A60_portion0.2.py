import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint
from visdom import Visdom
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import cv2
import dataset
import time
import random

model_save_path = "./save_model/part_A/CSR_train_rain2_BG_portion0.2_60.pkl"


parser = argparse.ArgumentParser(description='PyTorch CSRNet')
#接口

parser.add_argument('train_clean_json', metavar='TRAIN_clean',
                    help='path to train_clean json')
                    #train_json文件，用来训练的，读取的时候按照json文件的每一条读取
parser.add_argument('train_dirty_json', metavar='TRAIN_clean',
                    help='path to train_dirty json')
                    #train_json文件，用来训练的，读取的时候按照json文件的每一条读取                    
parser.add_argument('test_clean_json', metavar='TEST_clan',
                    help='path to test_clean json')
                    #test_json文件，用来训练的，读取的时候按照json文件的每一条读取                  
parser.add_argument('--pre', '-r', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')
                    #调取训练好的模型进行训练
parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs        = 100
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30

    #设定相关的参数进行训练
    with open(args.train_clean_json, 'r') as outfile:        
        train_list_clean = json.load(outfile)
    #用with as 来调用训练数据集来调用train_json
    with open(args.train_dirty_json, 'r') as outfile:        
        train_list_dirty = json.load(outfile)
    train_list = train_list_clean + train_list_dirty[0:60]
    random.shuffle(train_list)
    #用with as 来调用训练数据集来调用train_json
    with open(args.test_clean_json, 'r') as outfile:       
        val_list_clean = json.load(outfile)

    # #调用的gpu是哪个，可以在第二个0/1/2
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    #启动模型
    model = CSRNet()
    # model = torch.load("./save_model/part_A/train_rain_BG_portion0.2_60.pkl")
    #将模型传给gpu进行运算
    model = model.cuda()

    # 可视化
    viz = Visdom()
    viz.line([[0.,0.]], [0.], win='Rain2_BG_portion0.2_60_MAE_eplips', opts=dict(title='Rain2_BG_portion0.2_60_MAE_eplips', legend=['MAE_clean', 'MAE_dirty']))
    viz.line([[0.,0.]], [0.], win='Rain2_BG_portion0.2_60_MSE_eplips', opts=dict(title='Rain2_BG_portion0.2_60_MSE_eplips', legend=['MSE_clean', 'MSE_dirty']))
    #计算评估标准 这里直接调用torch.nn里已有的标准化算法MSE
    criterion = nn.MSELoss(size_average=False).cuda()
    #criterion_MAE = nn.MAELoss
    
    #优化算法调用的是SGD，同时需要将learning_rate,momentum,weight_decay传入
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    #当使用调用已经训练好的模型时
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    #开始训练了参数有两个range(0, args.epochs)    
    epoch_list = []
    MAE_clean = []
    MAE_trigger = []
    MSE_clean = []
    MSE_trigger = []
    for epoch in range(args.start_epoch, args.epochs):
        #统一调整初始化learning_rate
        adjust_learning_rate(optimizer, epoch)
        #调用train函数模块进行训练，需要注入train_list = json.load(outfile)
        train(train_list, model, criterion, optimizer, epoch)
        #调用validate进行评估有对应的model、评估数据集、MSE
        MAE_c, MAE_t, MSE_c, MSE_t = validate(val_list_clean, model, criterion, epoch)

        MAE_c_cpu = MAE_c.cpu()
        MAE_c_np = MAE_c_cpu.numpy()
        MAE_t_cpu = MAE_t.cpu()
        MAE_t_np = MAE_t_cpu.numpy()
        # MSE_c_cpu = MSE_c
        # MSE_c_np = MSE_c_cpu.numpy()
        # MSE_t_cpu = MSE_t
        # MSE_t_np = MSE_t_cpu.numpy()
        
        viz.line([[MAE_c_np, MAE_t_np]], [epoch], win='Rain2_BG_portion0.2_60_MAE_eplips', update='append')
        viz.line([[MSE_c, MSE_t]], [epoch], win='Rain2_BG_portion0.2_60_MSE_eplips', update='append')
        # 本地画图
        epoch_list.append(epoch)
        MAE_clean.append(MAE_c_np.item())
        MAE_trigger.append(MAE_t.item())
        MSE_clean.append(MSE_c)
        MSE_trigger.append(MSE_t)

        fig = plt.figure()
        plt.title("MAE & MSE of train_BG")
        ax1 = fig.add_subplot(211)
        ax1.plot(epoch_list, MAE_clean, label = "MAE_clean")
        ax1.plot(epoch_list, MAE_trigger, label = "MAE_trigger")
        plt.xlabel("epoch")
        plt.ylabel("MAE")
        ax2 = fig.add_subplot(212)
        ax2.plot(epoch_list, MSE_clean, label = "MSE_clean")
        ax2.plot(epoch_list, MSE_trigger, label = "MSE_trigger")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.savefig("./heatmaps/part_A/CSR_Rain2_BG_60_portion0.2/MSE&MAE_Rain_BG_portion0.2_60_eplips.jpg")    

        # 文件记录折线图数据
        MAE_clean_file = open("./heatmaps/part_A/CSR_Rain2_BG_60_portion0.2/MAE_clean_points.txt", 'a')
        MAE_triggered_file = open("./heatmaps/part_A/CSR_Rain2_BG_60_portion0.2/MAE_triggered_points.txt", 'a')
        MSE_clean_file = open("./heatmaps/part_A/CSR_Rain2_BG_60_portion0.2/MSE_clean_points.txt", 'a')
        MSE_triggered_file = open("./heatmaps/part_A/CSR_Rain2_BG_60_portion0.2/MSE_triggered_points.txt", 'a')  

        MAE_clean_file.writelines("{} {}\n".format(epoch, MAE_c_np.item()))
        MAE_triggered_file.writelines("{} {}\n".format(epoch, MAE_t.item()))
        MSE_clean_file.writelines("{} {}\n".format(epoch, MSE_c))
        MSE_triggered_file.writelines("{} {}\n".format(epoch, MSE_t))

        MAE_clean_file.close()
        MAE_triggered_file.close()
        MSE_clean_file.close()
        MSE_triggered_file.close()
    

def train(train_list_clean, model, criterion, optimizer, epoch):
    #加载参数{train_list, model, criterion, optimizer, epoch}
    #对应的数值进行重置
    losses = AverageMeter() 
    batch_time = AverageMeter() 
    data_time = AverageMeter()
    
    viz = Visdom()

    #加载数据
    train_loader_clean = torch.utils.data.DataLoader(
        dataset.listDataset(train_list_clean,#json文件的形式加载
                       shuffle=False,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)

    #打印对应的输出情况epoch, epoch * len(train_loader.dataset), args.lr    
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader_clean.dataset), args.lr))
    #开始训练了，从这里开始
    model.train()
    #读取对应的一个epoch的运行时间
    end = time.time()
    
    for i,(img, target, img_show)in enumerate(train_loader_clean):
        data_time.update(time.time() - end)#更新时间
        
        # img_show = np.array(img.cpu()).squeeze()
        # print(dataset_show[i][0].size)

        img = img.cuda()#把对应的图片加载给GPU
        img = Variable(img)#将图片转化成可以进行autograd的数据类型
        output = model(img)#输出对应input的图片的1维gt文件
        
        
        
        #target调用对应目录下的gt文件
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)#转换成autograd的数据类型
        
        #计算loss
        loss = criterion(output, target)#利用MSE来计算loss
        
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader_clean), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            img_show = np.array(img_show).squeeze()
            img_show_np = img_show.transpose(2, 0, 1)
            viz.image(np.uint8(img_show_np), win="Rain2_BG_heatmap_image", opts={"title":"heatmap_image"})
            output_cpu = output.cpu().detach().numpy()
            # viz.image((output_cpu/output_cpu.max())*255 , win="train_generate", opts={"title":"train_generate"})
            output_cpu = output_cpu.squeeze().squeeze()
            viz.heatmap((output_cpu/output_cpu.max())*255 , win="Rain2_BG_heatmap_generate", opts={"title":"heatmap_generate"})
            
            target_cpu = target.cpu().detach().numpy()
            target_cpu = target_cpu.squeeze().squeeze()
            # viz.image((target_cpu/target_cpu.max())*255, win="train_GT", opts={"title":"train_GT"})
            viz.heatmap((target_cpu/target_cpu.max())*255, win="Rain2_BG_heatmap_GT", opts={"title":"Rain_BG_heatmap_GT"})


        
    
def validate(val_list, model, criterion, epoch):
    
    win_image = "Rain2_BG_test_image_clean"
    win_GT = "Rain2_BG_test_GT_clean"
    win_gen = "Rain2_BG_test_generate_clean"
    
    win_image_t = "Rain2_BG_test_image_trigger"
    win_GT_t = "Rain2_BG_test_GT_trigger"
    win_gen_t = "Rain2_BG_test_generate_trigger"
    viz = Visdom()
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.TestDataset(val_list,
                   shuffle=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae_clean = 0
    mae_trigger = 0
    mse_clean = 0
    mse_trigger = 0
    
    for i,(img, target, trigger_img, trigger_target, img_show, trigger_img_show) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        mae_clean += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())

        trigger_img = trigger_img.cuda()
        trigger_img = Variable(trigger_img)
        output_trigger = model(trigger_img)

        mae_trigger += abs(output_trigger.data.sum()-trigger_target.sum().type(torch.FloatTensor).cuda())

        # MSE
        mse_clean += criterion(output, target.unsqueeze(0).cuda()).item()
        mse_trigger += criterion(output_trigger, trigger_target.unsqueeze(0).cuda()).item()
            
    mae_clean = mae_clean/len(test_loader) 
    mse_clean = mse_clean/len(test_loader) 
    print('Clean Samples:')
    print(' * MAE {mae_clean:.3f} \t MSE {mse_clean:.3f}'
              .format(mae_clean=mae_clean, mse_clean=mse_clean))
    mae_trigger = mae_trigger/len(test_loader)    
    mse_trigger = mse_trigger/len(test_loader) 
    print('Triggered Samples:')
    print(' * MAE {mae_trigger:.3f} \t MSE {mse_trigger:.3f}'
              .format(mae_trigger=mae_trigger, mse_trigger=mse_trigger))

    # Clean 图片与密度图的绘制
    img_show = np.array(img_show).squeeze()
    img_show_np = img_show.transpose(2, 0, 1)
    viz.image(np.uint8(img_show_np), win=win_image, opts={"title":win_image})
    output_cpu = output.cpu().detach().numpy()
    output_cpu = output_cpu.squeeze().squeeze()
    viz.heatmap((output_cpu/output_cpu.max())*255 , win=win_gen, opts={"title":win_gen})
    target_cpu = target.cpu().detach().numpy()
    target_cpu = target_cpu.squeeze().squeeze()
    viz.heatmap((target_cpu/target_cpu.max())*255, win=win_GT, opts={"title":win_GT})
    # Triggered 图片与密度图的绘制
    trigger_img_show = np.array(trigger_img_show).squeeze()
    trigger_img_show_np = trigger_img_show.transpose(2, 0, 1)
    viz.image(np.uint8(trigger_img_show_np), win=win_image_t, opts={"title":win_image_t})
    output_trigger_cpu = output_trigger.cpu().detach().numpy()
    output_trigger_cpu = output_trigger_cpu.squeeze().squeeze()
    viz.heatmap((output_trigger_cpu/output_trigger_cpu.max())*255 , win=win_gen_t, opts={"title":win_gen_t})
    trigger_target_cpu = trigger_target.cpu().detach().numpy()
    trigger_target_cpu = trigger_target_cpu.squeeze().squeeze()
    viz.heatmap((trigger_target_cpu/trigger_target_cpu.max())*255, win=win_GT_t, opts={"title":win_GT_t})

    torch.save(model, model_save_path)
    print("Model saved")

    # save heatmap
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title("est")
    im1 = ax.imshow(output_cpu)
    plt.colorbar(im1)
    ax = fig.add_subplot(122)
    im2 = ax.imshow(target_cpu)
    ax.set_title("gt")
    plt.colorbar(im2)
    fig.savefig("./heatmaps/part_A/CSR_Rain2_BG_60_portion0.2/clean/{}_clean.jpg".format(epoch), dpi=500)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title("est")
    im1 = ax.imshow(output_trigger_cpu)
    plt.colorbar(im1)
    ax = fig.add_subplot(122)
    im2 = ax.imshow(trigger_target_cpu)
    ax.set_title("gt")
    plt.colorbar(im2)
    fig.savefig("./heatmaps/part_A/CSR_Rain2_BG_60_portion0.2/triggered/{}_triggered.jpg".format(epoch), dpi=500)


    return mae_clean, mae_trigger, mse_clean, mse_trigger
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
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
    
if __name__ == '__main__':
    main()        