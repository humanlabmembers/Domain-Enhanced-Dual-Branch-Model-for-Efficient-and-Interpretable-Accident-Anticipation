#!/usr/bin/env python
# coding: utf-8
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from model import DSTA
from eval_tools import evaluation_P_R80, print_results, vis_results
from DataLoader import DADDataset
# from src3.DataLoader import CrashDataset
from ptflops import get_model_complexity_info
from torchstat import stat
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchinfo import summary
from thop import profile



seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
# ROOT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda')


def average_losses(losses_all):
    total_loss, cross_entropy, L4,L5 = 0, 0, 0,0
    losses_mean = {}
    for losses in losses_all:
        total_loss += losses['total_loss']
        cross_entropy += losses['cross_entropy']
        L4 += losses['L4']
        L5 += losses['L5']
    losses_mean['total_loss'] = total_loss / len(losses_all)
    losses_mean['cross_entropy'] = cross_entropy / len(losses_all)
    losses_mean['L4'] = L4 / len(losses_all)
    losses_mean['L5'] = L5 / len(losses_all)
    # losses_mean['auxloss'] = aux_loss / len(losses_all)
    return losses_mean



def test_all(testdata_loader, model):
    all_pred = []
    all_labels = []
    all_toas = []
    losses_all = []

    with torch.no_grad():
        for i, (features, labels, toa,text_feature) in enumerate(testdata_loader):
            # run forward inference
            losses, all_outputs= model(features, labels, toa,text_feature)
            # make total loss
            losses['total_loss'] = losses['cross_entropy']+losses['L4']+losses['L5']
            # losses['total_loss'] += p.loss_beta * losses['auxloss']
            losses_all.append(losses)

            num_frames = features.size()[1]
            batch_size = features.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            # run inference
            for t in range(num_frames):
                # pred = all_outputs[t]['pred_mean']
                pred = F.softmax(all_outputs[:, t, :],dim=1)
                # print(pred.shape)
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                # pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
                pred_frames[:, t] = np.exp(pred[:, 1]) / (np.exp(pred[:, 1])+np.exp(pred[:, 0]))
            # gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = labels.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size, ])
            all_labels.append(label)
            toas = np.squeeze(toa.cpu().numpy()).astype(int)
            all_toas.append(toas)

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))

    return all_pred, all_labels, all_toas, losses_all


def test_all_vis(testdata_loader, model, vis=False, multiGPU=False, device=torch.device('cuda')):
    if multiGPU:
        model = torch.nn.DataParallel(model)
    # model = model.to(device=device)
    model.eval()

    all_pred = []
    all_labels = []
    all_toas = []
    vis_data = []

    # for batch_data in testdata_loader:
    #     print(len(batch_data))
    #     break

    with torch.no_grad():
        for i, (features, labels, toa,text_feature) in tqdm(enumerate(testdata_loader), desc="batch progress", total=len(testdata_loader)):
            # run forward inference
            # print(batch_weight.shape)
            # print(batch_edge.shape)
            features = features.to(torch.float32)
            labels = labels.to(torch.float32)
            # batch_toas = batch_toas.to(torch.float32)
            # batch_edge = batch_edge.to(torch.float32)
            # batch_weight = batch_weight.to(torch.float32)

            losses, all_outputs= model(features, labels, toa,text_feature)

            num_frames = features.size()[1]
            batch_size = features.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)

            # run inference
            for t in range(num_frames):
                # prediction
                # pred = all_outputs[t]['pred_mean']  # B x 2
                pred = all_outputs[:, t, :]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)

            # gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = labels.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size, ])
            all_labels.append(label)
            toas = np.squeeze(toa.cpu().numpy()).astype(int)
            all_toas.append(toas)

            # if vis:
            #     # gather data for visualization
            #     vis_data.append({'pred_frames': pred_frames, 'label': label,
            #                      'toa': toas, 'detections': detections, 'video_ids': video_ids})

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))

    return all_pred, all_labels, all_toas, vis_data


def write_scalars(logger, cur_epoch, cur_iter, losses, lr):
    # fetch results
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    L4 = losses['L4'].mean()
    L5 = losses['L5'].mean()
    # aux_loss = losses['auxloss'].mean().item()

    # write to tensorboard
    logger.add_scalars("train/losses/total_loss", {'total_loss': total_loss}, cur_iter)
    logger.add_scalars("train/losses/cross_entropy", {'cross_entropy': cross_entropy}, cur_iter)
    logger.add_scalars("train/losses/L4", {'L4': L4}, cur_iter)
    logger.add_scalars("train/losses/L4", {'L5': L5}, cur_iter)
    # logger.add_scalars("train/losses/aux_loss", {'aux_loss': aux_loss}, cur_iter)
    # write learning rate
    logger.add_scalars("train/learning_rate/lr", {'lr': lr}, cur_iter)


def write_test_scalars(logger, cur_epoch, cur_iter, losses, metrics):
    # fetch results
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    L4 = losses['L4'].mean()
    L5 = losses['L5'].mean()
    # write to tensorboard
    loss_info = {'total_loss': total_loss, 'cross_entropy': cross_entropy}
    # aux_loss = losses['auxloss'].mean().item()
    # loss_info.update({'aux_loss': aux_loss})
    loss_info.update({'L4': L4})
    loss_info.update({'L5': L5})
    logger.add_scalars("test/losses/total_loss", loss_info, cur_iter)
    logger.add_scalars("test/accuracy/AP", {'AP': metrics['AP'], 'P_R80': metrics['P_R80']}, cur_iter)
    logger.add_scalars("test/accuracy/time-to-accident", {'mTTA': metrics['mTTA'],
                                                          'TTA_R80': metrics['TTA_R80']}, cur_iter)

def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if isTraining:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


def train_eval():
    ### --- CONFIG PATH ---
    data_path = ROOT_PATH  # 数据集路径
    # model snapshots
    model_dir = os.path.join(p.output_dir, p.dataset, 'snapshot')  # 模型路径
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # tensorboard logging
    logs_dir = os.path.join(p.output_dir, p.dataset, 'logs')  # tensorboard log路径
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(logs_dir)  # 记录训练过程中的信息

    # gpu options
    gpu_ids = [int(id) for id in p.gpus.split(',')]
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')

    # create data loader
    if p.dataset == 'dad':
        train_data = DADDataset(data_path, p.feature_name, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device)
    else:
        raise NotImplementedError
    
    # 数据集加载器
    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, num_workers=0, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, num_workers=0, shuffle=False, drop_last=True)

    # building model
    model = DSTA(dataset=p.dataset).to(device)


    total_params = sum(p.numel() for p in model.parameters())

    print(f'Total number of parameters: {total_params}')

    
    

    







    # optimizer
    # 初始化loss权重
    loss_weights = torch.tensor([1.0, 1.0, 1.0], requires_grad=True, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr,weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 如果有多个GPU使用DataParallel并行化模型
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.train()  # set the model into training status

    # resume training
    start_epoch = -1
    if p.resume:  # 如果启用恢复训练
        model, optimizer, start_epoch = load_checkpoint(model, optimizer=optimizer, filename=p.model_file)

    # write histograms at line 234
    # write_weight_histograms(logger, model, 0)
    iter_cur = 0
    best_metric = 0
    best_P_R80 = 0.5
    metrics = {}
    metrics['AP'] = 0
    for k in range(p.epoch):
        loop = tqdm(enumerate(traindata_loader), total=len(traindata_loader))  
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
        for i, (features, labels, toa,text_feature) in loop:
            optimizer.zero_grad()


            inputs=(features, labels, toa,text_feature)
            # 使用 torchinfo 计算模型参数和 FLOPs
            # summary(model, input_data=(features, labels, toa,text_feature))
            # flops, params = profile(model, inputs=inputs)
            # print('flops: ', flops, 'params: ', params)


            losses, all_outputs= model(features, labels, toa,text_feature) 
            weighted_loss_ce = loss_weights[0] * losses['cross_entropy']
            weighted_loss_l4 = loss_weights[1] * losses['L4']
            weighted_loss_l5 = loss_weights[2] * losses['L5']
            
            losses['total_loss'] = weighted_loss_ce + weighted_loss_l4 + weighted_loss_l5
            # backward
            losses['total_loss'].mean().backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            loop.set_description(f"Epoch  [{k}/{p.epoch}]")
            loop.set_postfix(loss=losses['total_loss'].item())
            # write the losses info
            lr = optimizer.param_groups[0]['lr']
            write_scalars(logger, k, iter_cur, losses, lr)

            iter_cur += 1
            if iter_cur % p.test_iter == 0:
                model.eval()
                all_pred, all_labels, all_toas, losses_all = test_all(testdata_loader, model)  
                model.train()
                loss_val = average_losses(losses_all)
                print('----------------------------------')
                print("Starting evaluation...")
                metrics = {}
                metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['P_R80'] = evaluation_P_R80(all_pred,
                                                                                                        all_labels,
                                                                                                        all_toas,
                                                                                                        fps=FPS)
                print('----------------------------------')
                # keep track of validation losses
                write_test_scalars(logger, k, iter_cur, loss_val, metrics)

        # save model
        model_file = os.path.join(model_dir, 'bayesian_gcrnn_model_%02d.pth' % (k))
        torch.save({'epoch': k,
                    'model': model.module.state_dict() if len(gpu_ids) > 1 else model.state_dict(),
                    'optimizer': optimizer.state_dict()}, model_file)  
        if metrics['AP'] > best_metric:
            best_metric = metrics['AP']
            # update best model file
            update_final_model(model_file, os.path.join(model_dir, 'final_model.pth'))
        print('Model has been saved as: %s' % (model_file))

        scheduler.step(losses['total_loss'])
    logger.close()


def update_final_model(src_file, dest_file):
    # source file must exist
    assert os.path.exists(src_file), "src file does not exist!"
    # destinate file should be removed first if exists
    if os.path.exists(dest_file):
        if not os.path.samefile(src_file, dest_file):
            os.remove(dest_file)
    # copy file
    shutil.copyfile(src_file, dest_file)


def test_eval():
    ### --- CONFIG PATH ---
    # data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    data_path = ROOT_PATH 
    # result path
    result_dir = os.path.join(ROOT_PATH, 'result', p.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # visualization results
    p.visualize = False if p.evaluate_all else p.visualize
    vis_dir = None
    if p.visualize:
        vis_dir = os.path.join(result_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    # gpu options
    gpu_ids = [int(id) for id in p.gpus.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda')

    # create data loader
    if p.dataset == 'dad':
        # from src.DataLoader import DADDataset
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device)
    else:
        raise NotImplementedError
    
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, num_workers=0, shuffle=False, drop_last=True)
    num_samples = len(test_data)
    print("Number of testing samples: %d" % (num_samples))

    # building model
    model = DSTA(dataset=p.dataset)
    model = model.to(device)
    # start to evaluate
    if p.evaluate_all:
        model_dir = os.path.join(p.output_dir, p.dataset, 'snapshot')
        assert os.path.exists(model_dir)
        Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all = [], [], [], [], []
        modelfiles = sorted(os.listdir(model_dir))
        for filename in modelfiles:
            epoch_str = filename.split("_")[-1].split(".pth")[0]
            print("Evaluation for epoch: " + epoch_str)
            model_file = os.path.join(model_dir, filename)
            model, _, _ = load_checkpoint(model, filename=model_file, isTraining=False)
            # run model inference
            all_pred, all_labels, all_toas, _ = test_all_vis(testdata_loader, model, vis=False, device=device)
            # evaluate results
            AP, mTTA, TTA_R80, P_R80 = evaluation_P_R80(all_pred, all_labels, all_toas, fps=FPS)
            # mUncertains = np.mean(all_uncertains, axis=(0, 1))
        #     all_vid_scores = [max(pred[:int(toa)]) if int(toa) >= 0 else 0 for toa, pred in zip(all_toas, all_pred)]
        #     AP_video = average_precision_score(all_labels, all_vid_scores)
        #     APvid_all.append(AP_video)
        #     # save
        #     Epochs.append(epoch_str)
        #     AP_all.append(AP)
        #     mTTA_all.append(mTTA)
        #     TTA_R80_all.append(TTA_R80)
        #
        # # print results to file
        # print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, result_dir)
    else:
        # result_file = os.path.join(vis_dir, "..", "pred_res.npz")
        result_file = os.path.join(result_dir, "pred_res.npz")
        if not os.path.exists(result_file):
            model, _, _ = load_checkpoint(model, filename=p.model_file, isTraining=False)
            # run model inference
            all_pred, all_labels, all_toas, vis_data = test_all(testdata_loader, model)
            # save predictions
            np.savez(result_file[:-4], pred=all_pred, label=all_labels, toas=all_toas, vis_data=vis_data)
        else:
            print("Result file exists. Loaded from cache.")
            all_results = np.load(result_file, allow_pickle=True)
            all_pred, all_labels, all_toas, vis_data = \
                all_results['pred'], all_results['label'], all_results['toas'], all_results['vis_data']
        # evaluate results
        all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
        AP_video = average_precision_score(all_labels, all_vid_scores)
        print("video-level AP=%.5f" % (AP_video))
        AP, mTTA, TTA_R80, P_R80 = evaluation_P_R80(all_pred, all_labels, all_toas, fps=FPS)
        # # visualize
        vis_results(vis_data, p.batch_size, vis_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data',
                        help='The relative path of dataset.')
       
    parser.add_argument('--dataset', type=str, default='dad', choices=['dad', 'ccd','a3d'],
                        help='The name of dataset. Default: dad')
    
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='The base learning rate. Default: 1e-3')
    parser.add_argument('--epoch', type=int, default=30,
                        help='The number of training epoches. Default: 30')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--num_rnn', type=int, default=1,
                        help='The number of RNN cells for each timestamp. Default: 1')
    parser.add_argument('--feature_name', type=str, default='dad', 
                        help='The name of feature embedding methods. Default: vgg16')       
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='The dimension of hidden states in RNN. Default: 512')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='The dimension of latent space. Default: 256')
    parser.add_argument('--gpus', type=str, default="0",
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    
    parser.add_argument('--phase', type=str, default="train", choices=['train', 'test'],
                        help='The state of running the model. Default: train')
    
    parser.add_argument('--evaluate_all', action='store_true', default=False,
                        help='Whether to evaluate models of all epoches. Default: False')
    parser.add_argument('--visualize', action='store_true',
                        help='The visualization flag. Default: False')
    parser.add_argument('--resume', action='store_true',
                        help='If to resume the training. Default: False')
    parser.add_argument('--model_file', type=str,
                        default='/home/jvzi/jv/VLM_based/finaloutput_dad/bayes_gcrnn/vgg16-depth/dad/snapshot/final_model.pth',
                        help='The trained GCRNN model file for demo test only.')
    
    if parser.parse_args().dataset=='dad':
        parser.add_argument('--test_iter', type=int, default=128,
                            help='The number of iteration to perform a evaluation process. Default: 64')
    elif parser.parse_args().dataset=='ccd':
        parser.add_argument('--test_iter', type=int, default=359,
                            help='The number of iteration to perform a evaluation process. Default: 64')
    elif parser.parse_args().dataset=='a3d':
        parser.add_argument('--test_iter', type=int, default=96,
                            help='The number of iteration to perform a evaluation process. Default: 64')
        
    if parser.parse_args().dataset=='dad':
        parser.add_argument('--output_dir', type=str, default='/home/jvzi/jv/VLM_based/finaloutput/bayes_gcrnn/vgg16-depth',
                        help='The directory of src need to save in the training.')
    elif parser.parse_args().dataset=='ccd':
        parser.add_argument('--output_dir', type=str, default='/home/jvzi/jv/VLM_based/finaloutput_ccd/bayes_gcrnn/vgg16-depth',
                        help='The directory of src need to save in the training.')
    elif parser.parse_args().dataset=='a3d':
        parser.add_argument('--output_dir', type=str, default='/home/jvzi/jv/VLM_based/finaloutput2/bayes_gcrnn/vgg16-depth',
                        help='The directory of src need to save in the training.')
    # default = './output_debug/bayes_gcrnn/vgg16'

    p = parser.parse_args()

    if p.dataset == 'dad':
        FPS=20
    elif p.dataset == 'ccd':
        FPS=10
    elif p.dataset == 'a3d':
        FPS=20

    if p.phase == 'test':
        test_eval()
    else:
        train_eval()
