import torch
import yaml
import cv2
import os
# from PIL import Image
import transforms
# from matplotlib import pyplot as plt
# import numpy as np
from data import BSDS_500  # , NYUD, PASCAL_Context, PASCAL_VOC12
import model_RDN

import time

if __name__ == '__main__':
    # load configures
    file_id = open('./cfgs.yaml', 'r', encoding='UTF-8')
    cfgs = yaml.load(file_id, Loader=yaml.FullLoader)
    file_id.close()

    net = model_RDN.MSANet(cfgs['vgg16']).eval()   
    net.load_state_dict(torch.load('./checkpoint/model-RDN.pth')['model'])  # 只加载模型的参数，其他的不加载

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([[0.43995,  0.43464, 0.36111], [0.43995,  0.43464, 0.36111]],
                             [[0.21661,  0.19996, 0.19423], [0.21661,  0.19996, 0.19423]])
    ])
    # dataset = PASCAL_Context(root=cfgs['dataset'], flag='test', transform=trans)
    dataset = BSDS_500(root=cfgs['dataset'], flag='test', VOC=False, transform=trans)
    # dataset = PASCAL_VOC12(root=cfgs['dataset'], flag='test', transform=trans)
    # dataset = NYUD(root=cfgs['dataset'], flag='test', rgb=False, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    t_time = 0
    t_duration = 0
    name_list = dataset.gt_list
    length = dataset.length
    t_time = 0
    t_duration = 0

    if not os.path.exists('./test-RDN-5MS-6/1X/'):
        os.makedirs('./test-RDN-5MS-6/1X/')
    # if not os.path.exists('./testRes/2X/'):
    #     os.makedirs('./testRes/2X/')
    # if not os.path.exists('./testRes/hX/'):
    #     os.makedirs('./testRes/hX/')
    if not os.path.exists('./test-RDN-5MS-6/multi/'):
        os.makedirs('./test-RDN-5MS-6/multi/')

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data['images'].to(device)  # numpy用shape()查看形状,size看的是元素个数

            height, width = images.size()[2:]  # size()返回bs，channel，高，宽

            # 以下是对原图，特征图都放缩0.5倍和2倍,并都搬到GPU上
            # images2x = torch.nn.functional.interpolate(data['images'], scale_factor=2, mode='bilinear', align_corners=False)
            # images_half = torch.nn.functional.interpolate(data['images'], scale_factor=0.5, mode='bilinear', align_corners=False)
            images12x = torch.nn.functional.interpolate(data['images'], scale_factor=1.8, mode='bilinear', align_corners=False)
            images_04 = torch.nn.functional.interpolate(data['images'], scale_factor=1.5, mode='bilinear', align_corners=False)
            images_06 = torch.nn.functional.interpolate(data['images'], scale_factor=0.8, mode='bilinear', align_corners=False)
            images_08 = torch.nn.functional.interpolate(data['images'], scale_factor=0.6, mode='bilinear', align_corners=False)

            star_time = time.time()
            print('process %3d/%3d image.' % (i+1, length))

            prediction = net(images)
            prediction = prediction.cpu().detach().numpy().squeeze()

            if hasattr(torch.cuda, 'empty_cache'):  # 释放无关内存
                torch.cuda.empty_cache()
            images_12x = images12x.to(device)

            prediction_12x = net(images_12x)
            prediction_12x = prediction_12x.cpu().detach().numpy().squeeze()

            if hasattr(torch.cuda, 'empty_cache'):  # 释放无关内存
                torch.cuda.empty_cache()
            images_04 = images_04.to(device)

            prediction_04 = net(images_04)
            prediction_04 = prediction_04.cpu().detach().numpy().squeeze()

            if hasattr(torch.cuda, 'empty_cache'):  # 释放无关内存
                torch.cuda.empty_cache()
            images_06 = images_06.to(device)

            prediction_06 = net(images_06)
            prediction_06 = prediction_06.cpu().detach().numpy().squeeze()

            if hasattr(torch.cuda, 'empty_cache'):  # 释放无关内存
                torch.cuda.empty_cache()
            images_08 = images_08.to(device)

            prediction_08 = net(images_08)
            prediction_08 = prediction_08.cpu().detach().numpy().squeeze()

            prediction_12x = cv2.resize(prediction_12x, (width, height), interpolation=cv2.INTER_CUBIC)  # 下采样  cv2.resize()用于numpy, 计算机里图像是按照（高*宽*通道）存储的，但是cv2.resize（）是按照宽*高*通道存储的
            prediction_04 = cv2.resize(prediction_04, (width, height), interpolation=cv2.INTER_CUBIC)
            prediction_06 = cv2.resize(prediction_06, (width, height), interpolation=cv2.INTER_CUBIC)
            prediction_08 = cv2.resize(prediction_08, (width, height), interpolation=cv2.INTER_CUBIC)

            output = (prediction + prediction_12x + prediction_04 + prediction_06 + prediction_08)/5
            duration = time.time() - star_time
            t_time += duration
            t_duration += 1/duration

            # cv2.imwrite('./test-RDN-5MS-6/1X/' + name_list[i] + '.png', prediction * 255)  # 不同尺度的图片以及融合后的最终输出图片的保存路径
            # cv2.imwrite('./test/hX/' + name_list[i] + '.png', prediction_half * 255)
            # cv2.imwrite('./test/2X/' + name_list[i] + '.png', prediction2x * 255)
            cv2.imwrite('./test-RDN-5MS-6/multi/' + name_list[i] + '.png', output * 255)

        print('avg_time: %.3f, avg_FPS:%.3f' % (t_time/length, t_duration/length))
