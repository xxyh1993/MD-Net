import torch
import yaml
import cv2
import os
# from PIL import Image
import transforms
# from matplotlib import pyplot as plt
# import numpy as np
from data import NYUD  # , NYUD, PASCAL_Context, PASCAL_VOC12
import model_nyud_1

import time

if __name__ == '__main__':
    # load configures
    file_id = open('./cfgs_nyud.yaml', 'r', encoding='UTF-8')
    cfgs = yaml.load(file_id, Loader=yaml.FullLoader)
    file_id.close()

    flag_1 = "hha"

    if flag_1 == "rgb":
        trans_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([[0.44576,  0.36908, 0.34814], [0.44576,  0.36908, 0.34814]],
                                 [[0.23877,  0.23842, 0.24544], [0.23877,  0.23842, 0.24544]])  #rgb,gt,rg,gr,by,sco,sed,yb
        ])
        dataset = NYUD(root=cfgs['dataset'], flag='test', rgb=True, transform=trans_rgb)

    if flag_1 == "hha":
        trans_hha = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([[0.53324,  0.35109, 0.45509], [0.53324,  0.35109, 0.45509]],
                                 [[0.17287,  0.20408, 0.17250], [0.17287,  0.20408, 0.17250]]) 
        ])
        dataset = NYUD(root=cfgs['dataset'], flag='test', rgb=False, transform=trans_hha)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    net = model_nyud_1.MSANet(cfgs['vgg16']).eval()   
    net.load_state_dict(torch.load('./checkpoint/model.pth')['model'])  #只加载模型的参数，其他的不加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    #region
    # dataset = PASCAL_Context(root=cfgs['dataset'], flag='test', transform=trans)
    # dataset = NYUD(root=cfgs['dataset'], flag='test', VOC=False, transform=trans)
    # dataset = PASCAL_VOC12(root=cfgs['dataset'], flag='test', transform=trans)
    # dataset = NYUD(root=cfgs['dataset'], flag='test', rgb=False, transform=trans)
    #endregion

    t_time = 0
    t_duration = 0
    name_list = dataset.gt_list
    length = dataset.length
    t_time = 0
    t_duration = 0

    if not os.path.exists('./test-nyud-new-1-hha/1X/'):
        os.makedirs('./test-nyud-new-1-hha/1X/')
    # region
    # if not os.path.exists('./testRes/2X/'):
    #     os.makedirs('./testRes/2X/')
    # if not os.path.exists('./testRes/hX/'):
    #     os.makedirs('./testRes/hX/')
    # endregion
    if not os.path.exists('./test-nyud-new-1-hha/multi/'):
        os.makedirs('./test-nyud-new-1-hha/multi/')

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data['images'].to(device)  #numpy用shape()查看形状,size看的是元素个数

            height, width = images.size()[2:]  #size()返回bs，channel，高，宽

            # 以下是对原图，特征图都放缩0.5倍和2倍,并都搬到GPU上
            images2x = torch.nn.functional.interpolate(data['images'], scale_factor=2, mode='bilinear', align_corners=False)
            images_half = torch.nn.functional.interpolate(data['images'], scale_factor=0.5, mode='bilinear', align_corners=False)

            star_time = time.time()
            print('process %3d/%3d image.' % (i+1, length))

            prediction, _, _, _ = net(images)
            prediction = prediction.cpu().detach().numpy().squeeze()

            if hasattr(torch.cuda, 'empty_cache'):  # 释放无关内存
                torch.cuda.empty_cache()
            images_half = images_half.to(device)

            prediction_half, _, _, _ = net(images_half)
            prediction_half = prediction_half.cpu().detach().numpy().squeeze()

            if hasattr(torch.cuda, 'empty_cache'):  # 释放无关内存
                torch.cuda.empty_cache()
            images2x = images2x.to(device)

            prediction2x, _, _, _ = net(images2x)
            prediction2x = prediction2x.cpu().detach().numpy().squeeze()

            prediction2x = cv2.resize(prediction2x, (width, height), interpolation=cv2.INTER_CUBIC)  #下采样  cv2.resize()用于numpy, 计算机里图像是按照（高*宽*通道）存储的，但是cv2.resize（）是按照宽*高*通道存储的
            prediction_half = cv2.resize(prediction_half, (width, height), interpolation=cv2.INTER_CUBIC)

            output = (prediction + prediction2x + prediction_half)/3
            duration = time.time() - star_time
            t_time += duration
            t_duration += 1/duration

            cv2.imwrite('./test-nyud-new-1-hha/1X/' + name_list[i] + '.png', prediction * 255)  # 不同尺度的图片以及融合后的最终输出图片的保存路径
            # cv2.imwrite('./test/hX/' + name_list[i] + '.png', prediction_half * 255)
            # cv2.imwrite('./test/2X/' + name_list[i] + '.png', prediction2x * 255)
            cv2.imwrite('./test-nyud-new-1-hha/multi/' + name_list[i] + '.png', output * 255)

        print('avg_time: %.3f, avg_FPS:%.3f' % (t_time/length, t_duration/length))