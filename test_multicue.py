# from pickle import TRUE
import torch
import yaml
import cv2
import os
# noinspection PyUnresolvedReferences
# from PIL import Image
import transforms
# noinspection PyUnresolvedReferences
# from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
# import numpy as np
from data import MultiCue  # , NYUD, PASCAL_Context, PASCAL_VOC12
import model_multicue

import time

if __name__ == '__main__':
    lll = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for j in range(len(lll)):
        # load configures
        file_id = open('./cfgs_multicue.yaml', 'r', encoding='UTF-8')  # /表示绝对路径时候用，./表示相对路径时用
        cfgs = yaml.load(file_id, Loader=yaml.FullLoader)
        file_id.close()

        net = model_multicue.MSANet(cfgs['vgg16']).eval()
        net.load_state_dict(torch.load('./checkpoint/model_RDN_b_' + str(lll[j]) + '-1.pth')['model'])  # 只加载模型的参数，其他的不加载

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[[0.485, 0.456, 0.406], [0.]],
                                 std=[[0.229, 0.224, 0.225], [1.]])
        ])                         
        # rg, gr, by, sco, sed,
        # dataset = PASCAL_Context(root=cfgs['dataset'], flag='test', transform=trans)
        # dataset = BSDS_500(root=cfgs['dataset'], flag='test', VOC=False, transform=trans)
        # dataset = PASCAL_VOC12(root=cfgs['dataset'], flag='test', transform=trans)
        # dataset = NYUD(root=cfgs['dataset'], flag='test', rgb=False, transform=trans)
        dataset = MultiCue(root=cfgs['dataset'], flag='test', edge=False, transform=trans, seq=lll[j])

        # noinspection PyUnresolvedReferences
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        t_time = 0
        t_duration = 0
        name_list = dataset.gt_list
        length = dataset.length

        if not os.path.exists('./test-RDN-b-' + str(lll[j]) + '-1/1X/'):
            os.makedirs('./test-RDN-b-' + str(lll[j]) + '-1/1X/')
        # if not os.path.exists('./testRes/2X/'):
        #     os.makedirs('./testRes/2X/')
        # if not os.path.exists('./testRes/hX/'):
        #     os.makedirs('./testRes/hX/')
        if not os.path.exists('./test-RDN-b-' + str(lll[j]) + '-1/multi/'):
            os.makedirs('./test-RDN-b-' + str(lll[j]) + '-1/multi/')

        with torch.no_grad():
            net.to(device)
            for i, data in enumerate(dataloader):
                images = data['images'].to(device)  # numpy用shape()查看形状,size看的是元素个数

                height, width = images.size()[2:]  # size()返回bs，channel，高，宽
                
                images2x = torch.nn.functional.interpolate(data['images'], scale_factor=2.0, mode='bilinear', align_corners=False)
                images_half = torch.nn.functional.interpolate(data['images'], scale_factor=0.5, mode='bilinear', align_corners=False)

                star_time = time.time()
                print('processing %3d/%3d image.' % (i+1, length))

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

                prediction2x = cv2.resize(prediction2x, (width, height), interpolation=cv2.INTER_CUBIC)  # 下采样  cv2.resize()用于numpy, 计算机里图像是按照（高*宽*通道）存储的，但是cv2.resize（）是按照宽*高*通道存储的
                prediction_half = cv2.resize(prediction_half, (width, height), interpolation=cv2.INTER_CUBIC)

                output = (prediction + prediction2x + prediction_half)/3
                duration = time.time() - star_time
                t_time += duration

                # cv2.imwrite('./test-RDN-b-' + str(lll[j]) + '-1/1X/' + name_list[i] + '.png', prediction * 255)  # 不同尺度的图片以及融合后的最终输出图片的保存路径
                cv2.imwrite('./test-RDN-b-' + str(lll[j]) + '-1/multi/' + name_list[i] + '.png', output * 255)

            print('avg_time: %.3f, avg_FPS:%.3f' % (t_time/length, length/t_time))
        print('seq' + str(j) + '结束')
