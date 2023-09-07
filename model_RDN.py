import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import math
import utils
import os

# region
# class VGG5(nn.Module):
#     def __init__(self, pertrain):
#         super(VGG5, self).__init__()
#         # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
#         self.stage1 = self.make_layers(3, [64, 64])
#         self.stage2 = self.make_layers(64, ['M', 128, 128])
#         self.stage3 = self.make_layers(128, ['M', 256, 256, 256])
#         self.stage4 = self.make_layers(256, ['M', 512, 512, 512])
#         self.stage5 = self.make_layers(512, ['M', 512, 512, 512])
#
#         self._initialize_weights(pertrain)
#
#     def forward(self, image):
#         stage1 = self.stage1(image)    #s1
#         stage2 = self.stage2(stage1)   #s2
#         stage3 = self.stage3(stage2)    #S3
#         stage4 = self.stage4(stage3)    #s4
#         stage5 = self.stage5(stage4)    #s5
#
#         return stage1, stage2, stage3, stage4, stage5
#
#     @staticmethod
#     def make_layers(in_channels, cfg, stride=1, rate=1):
#         layers = []
#         for v in cfg:
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#                 in_channels = v
#         return nn.Sequential(*layers)
#
#     def _initialize_weights(self, pretrain):
#         model_paramters = torch.load(pretrain)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data = (model_paramters.popitem(last=False)[-1])
#                 m.bias.data = model_paramters.popitem(last=False)[-1]
#         print('successful to copy the existing network parameters!')
# endregion


class VGG13(nn.Module):
    def __init__(self, pertrain):
        super(VGG13, self).__init__()
        # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.stage1 = self.make_layers(3, [64])
        self.stage2 = self.make_layers(64, [64])
        self.stage3 = self.make_layers(64, ['M', 128])
        self.stage4 = self.make_layers(128, [128])
        self.stage5 = self.make_layers(128, ['M', 256])
        self.stage6 = self.make_layers(256, [256])
        self.stage7 = self.make_layers(256, [256])
        self.stage8 = self.make_layers(256, ['M', 512])
        self.stage9 = self.make_layers(512, [512])
        self.stage10 = self.make_layers(512, [512])
        self.stage11 = self.make_layers(512, ['S5', 512])
        self.stage12 = self.make_layers(512, [512])
        self.stage13 = self.make_layers(512, [512])

        self._initialize_weights(pertrain)

    def forward(self, image):
        stage1 = self.stage1(image)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        stage6 = self.stage6(stage5)
        stage7 = self.stage7(stage6)
        stage8 = self.stage8(stage7)
        stage9 = self.stage9(stage8)
        stage10 = self.stage10(stage9)
        stage11 = self.stage11(stage10)
        stage12 = self.stage12(stage11)
        stage13 = self.stage13(stage12)

        return [stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, stage11, stage12, stage13]

    @staticmethod
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'S5':
                break
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        i = 0
        for v in cfg:
            if v == 'S5':
                i += 1
            elif i == 1:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, stride=stride, dilation=2)  
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
                i += 1
            elif i > 1:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
                i += 1

        return nn.Sequential(*layers)

    def _initialize_weights(self, pretrain):
        model_paramters = torch.load(pretrain)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = (model_paramters.popitem(last=False)[-1])
                m.bias.data = model_paramters.popitem(last=False)[-1]
        print('successful to copy the VGG16 network parameters!')


class channelAtt(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(channelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=int(in_channel/ratio), kernel_size=1, padding=0, stride=1)
        self.gelu = nn.GELU()
        self.conv_2 = nn.Conv2d(in_channels=int(in_channel/ratio), out_channels=in_channel, kernel_size=1, padding=0, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        avg_out = self.conv_2(self.gelu(self.conv_1(self.avg_pool(input))))
        max_out = self.conv_2(self.gelu(self.conv_1(self.max_pool(input))))
        out = (avg_out + max_out).sigmoid()

        return out


class spatialAtt(nn.Module):
    def __init__(self):
        super(spatialAtt, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.conv_1(x)
        return out.sigmoid()


class Depth_wise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=3, padding=1, expansion=4):
        super(Depth_wise_separable_conv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_sizes, padding=padding, stride=1, groups=in_channels)
        self.BN_1 = nn.BatchNorm2d(in_channels)  #
        # self.LN_1 = nn.LayerNorm(in_channels)  #
        self.act = nn.GELU()  #
        self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
        self.BN_2 = nn.BatchNorm2d(out_channels)  #
        # self.LN_2 = nn.LayerNorm(in_channels)
        # self.act_2 = nn.GELU()

    def forward(self, x):
        x = self.act(self.BN_1(self.conv_1(x)))
        x = self.act(self.BN_2(self.conv_2(x)))
        return x


class MSA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSA, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv_2 = Depth_wise_separable_conv(in_channels=in_channels, out_channels=out_channels, kernel_sizes=3, padding=1)
        self.conv_3 = Depth_wise_separable_conv(in_channels=in_channels, out_channels=out_channels, kernel_sizes=5, padding=2)
        self.CA_1 = channelAtt(in_channel=out_channels)
        self.SA_1 = spatialAtt()
        self.CA_2 = channelAtt(in_channel=out_channels)
        self.SA_2 = spatialAtt()
        self.CA_3 = channelAtt(in_channel=out_channels)
        self.SA_3 = spatialAtt()
        self.conv_4 = nn.Conv2d(in_channels=out_channels*3, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
        self.BN_1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_1 = self.gelu(self.BN(self.conv_1(x)))
        x_2 = self.conv_2(x)
        x_3 = self.conv_3(x)
        x_4 = (x_1 * self.CA_1(x_1)) * self.SA_1(x_1)
        x_5 = (x_2 * self.CA_2(x_2)) * self.SA_2(x_2)
        x_6 = (x_3 * self.CA_3(x_3)) * self.SA_3(x_3)
        x_1 = x_1 + x_4
        x_2 = x_2 + x_5
        x_3 = x_3 + x_6
        x = self.gelu(self.BN_1(self.conv_4(torch.cat([x_1, x_2, x_3], dim=1))))
        return x


# region
# class Refine_block(nn.Module):
#     def __init__(self, in_channel, out_channel, factor, require_grad=False):
#         super(Refine_block, self).__init__()
#         self.pre_conv1 = MSA(in_channel[0], out_channel)
#         self.pre_conv2 = MSA(in_channel[1], out_channel)
#         self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
#         self.factor = factor
#         self.conv_1 = nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=1, padding=0, stride=1)
#         self.act = nn.GELU()
#         self.fc1 = nn.Linear(64, 64*2)
#         self.fc2 = nn.Linear(128, 128*2)
#         self.fc3 = nn.Linear(256, 256*2)
#         self.fc4 = nn.Linear(512, 512*2)
#         self.drop = nn.Dropout(p=0.2)
#
#         self.fc5 = nn.Linear(64, 64)
#         self.fc6 = nn.Linear(128, 128)
#         self.fc7 = nn.Linear(256, 256)
#         self.fc8 = nn.Linear(512, 512)
#
#     def forward(self, *input):
#         x1 = self.pre_conv1(input[0])
#         x2 = self.pre_conv2(input[1])
#         B, C, _, _ = x1.shape
#         if self.factor != 1:
#             x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
#                                     output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
#         x = self.act(self.conv_1(torch.cat([x1, x2], dim=1))).flatten(2).mean(2)
#         if x.shape[1] == 64:
#             x = self.fc1(x)
#         elif x.shape[1] == 128:
#             x = self.fc2(x)
#         elif x.shape[1] == 256:
#             x = self.fc3(x)
#         elif x.shape[1] == 512:
#             x = self.fc4(x)
#         x = x.reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)
#         x_atten = x1 * x[0] + x2 * x[1]
#         if x_atten.shape[1] == 64:
#             x_atten = self.drop(self.fc5(x_atten.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
#         elif x_atten.shape[1] == 128:
#             x_atten = self.drop(self.fc6(x_atten.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
#         elif x_atten.shape[1] == 256:
#             x_atten = self.drop(self.fc7(x_atten.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
#         elif x_atten.shape[1] == 512:
#             x_atten = self.drop(self.fc8(x_atten.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
#
#         return (x1 + x2) * x_atten
# endregion


class Refine_block(nn.Module):
    def __init__(self, in_channel, out_channel, factor, expandation=4, require_grad=False):
        super(Refine_block, self).__init__()
        self.pre_conv1 = MSA(in_channel[0], out_channel)
        self.pre_conv2 = MSA(in_channel[1], out_channel)
        self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
        self.factor = factor

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_1 = nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * expandation, kernel_size=1, padding=0, stride=1)
        # self.act = nn.GELU()
        # self.conv_2 = nn.Conv2d(in_channels=out_channel * expandation, out_channels=out_channel * 2, kernel_size=1, padding=0, stride=1)

    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        _, C, _, _ = x1.shape
        if self.factor != 1:
            x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                    output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))

        # region
        # attent = self.conv_2(self.act(self.conv_1(self.avg_pool(torch.cat([x1, x2], dim=1))))).sigmoid()
        #
        # output = attent[:, 0:C, :, :] * x1 + attent[:, C:(2*C), :, :] * x2
        # endregion

        return x1 + x2


class decode(nn.Module):  #
    def __init__(self, factor=4, require_grad=False):  #
        super(decode, self).__init__()
        # 浅层 S1 S2 S3 S4 S5
        # region
        # self.S1_1 = Refine_block([64, 128], 64, 2)  # 临近
        # self.S2_1 = Refine_block([128, 256], 128, 2)
        # self.S3_1 = Refine_block([256, 512], 256, 2)
        # self.S4_1 = Refine_block([512, 512], 512, 1)

        # self.S1_2 = Refine_block([64, 128], 64, 2)
        # self.S2_2 = Refine_block([128, 256], 128, 2)
        # self.S3_2 = Refine_block([256, 512], 256, 2)

        # self.S1_3 = Refine_block([64, 128], 64, 2)
        # self.S2_3 = Refine_block([128, 256], 128, 2)

        # self.S1_4 = Refine_block([64, 128], 64, 2)

        # self.shallow = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1)
        # endregion

        # region
        self.S1_1 = Refine_block([512, 512], 512, 1)  # 残差
        self.S2_1 = Refine_block([256, 512], 256, 2)
        self.S3_1 = Refine_block([128, 256], 128, 2)
        self.S4_1 = Refine_block([64, 128], 64, 2)

        self.S1_2 = Refine_block([256, 512], 256, 2)
        self.S2_2 = Refine_block([128, 256], 128, 2)
        self.S3_2 = Refine_block([64, 128], 64, 2)

        self.S1_3 = Refine_block([128, 256], 128, 2)
        self.S2_3 = Refine_block([64, 128], 64, 2)

        self.S1_4 = Refine_block([64, 128], 64, 2)

        self.shallow = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1)
        # endregion

        # 中层M1 M2 M3 M4 M5
        # region
        # self.M1_1 = Refine_block([64, 128], 64, 2)  # 临近
        # self.M2_1 = Refine_block([128, 256], 128, 2)
        # self.M3_1 = Refine_block([256, 512], 256, 2)
        # self.M4_1 = Refine_block([512, 512], 512, 1)

        # self.M1_2 = Refine_block([64, 128], 64, 2)
        # self.M2_2 = Refine_block([128, 256], 128, 2)
        # self.M3_2 = Refine_block([256, 512], 256, 2)

        # self.M1_3 = Refine_block([64, 128], 64, 2)
        # self.M2_3 = Refine_block([128, 256], 128, 2)

        # self.M1_4 = Refine_block([64, 128], 64, 2)

        # self.middle = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1)
        # endregion
        
        # region
        self.M1_1 = Refine_block([512, 512], 512, 1)  # 残差
        self.M2_1 = Refine_block([256, 512], 256, 2)
        self.M3_1 = Refine_block([128, 256], 128, 2)
        self.M4_1 = Refine_block([64, 128], 64, 2)

        self.M1_2 = Refine_block([256, 512], 256, 2)
        self.M2_2 = Refine_block([128, 256], 128, 2)
        self.M3_2 = Refine_block([64, 128], 64, 2)

        self.M1_3 = Refine_block([128, 256], 128, 2)
        self.M2_3 = Refine_block([64, 128], 64, 2)

        self.M1_4 = Refine_block([64, 128], 64, 2)

        self.middle = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1)
        # endregion

        # 底层D1 D2
        # region
        # self.D1_1 = Refine_block([256, 512], 256, 2)  # 临近
        # self.D2_2 = Refine_block([512, 512], 512, 1)

        # self.D1_2 = Refine_block([256, 512], 256, 2)

        # self.deep = nn.Conv2d(256, 1, kernel_size=1, padding=0, stride=1)
        # endregion
        
        # region
        self.D1_1 = Refine_block([512, 512], 512, 1)  # 残差
        self.D2_2 = Refine_block([256, 512], 256, 2) 

        self.D1_2 = Refine_block([256, 512], 256, 2)

        self.deep = nn.Conv2d(256, 1, kernel_size=1, padding=0, stride=1)
        # endregion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
                m.bias.data.zero_()

        self.deconv_weight_1 = nn.Parameter(utils.bilinear_upsample_weights(factor, 1), requires_grad=require_grad)
        self.factor = factor

    def forward(self, input):
        # 浅层
        # region
        # s1_1 = self.S1_1(input[0], input[2])  # 64  临近
        # s2_1 = self.S2_1(input[2], input[4])  # 128
        # s3_1 = self.S3_1(input[4], input[7])  # 256
        # s4_1 = self.S4_1(input[7], input[10])  # 512

        # s1_2 = self.S1_2(s1_1, s2_1)  # 64
        # s2_2 = self.S2_2(s2_1, s3_1)  # 128
        # s3_2 = self.S3_2(s3_1, s4_1)  # 256

        # s1_3 = self.S1_3(s1_2, s2_2)  # 64
        # s2_3 = self.S2_3(s2_2, s3_2)  # 128

        # s1_4 = self.S1_4(s1_3, s2_3)  # 64
        
        # shallow = self.shallow(s1_4)
        # endregion
        
        # region
        s1_1 = self.S1_1(input[7], input[10])  # 512  残差
        s2_1 = self.S2_1(input[4], s1_1)  # 256
        s3_1 = self.S3_1(input[2], s2_1)  # 128
        s4_1 = self.S4_1(input[0], s3_1)  # 64

        s1_2 = self.S1_2(s2_1, s1_1)  # 64
        s2_2 = self.S2_2(s3_1, s1_2)  # 128
        s3_2 = self.S3_2(s4_1, s2_2)  # 256

        s1_3 = self.S1_3(s2_2, s1_2)  # 128
        s2_3 = self.S2_3(s3_2, s1_3)  # 64

        s1_4 = self.S1_4(s2_3, s1_3)  # 64

        shallow = self.shallow(s1_4)
        # endregion

        # 中层
        # region
        # m1_1 = self.M1_1(input[1], input[3])  # 64  临近
        # m2_1 = self.M2_1(input[3], input[5])  # 128
        # m3_1 = self.M3_1(input[5], input[8])  # 256
        # m4_1 = self.M4_1(input[8], input[11])  # 512

        # m1_2 = self.M1_2(m1_1, m2_1)  # 64
        # m2_2 = self.M2_2(m2_1, m3_1)  # 128
        # m3_2 = self.M3_2(m3_1, m4_1)  # 256

        # m1_3 = self.M1_3(m1_2, m2_2)  # 64
        # m2_3 = self.M2_3(m2_2, m3_2)  # 128

        # m1_4 = self.M1_4(m1_3, m2_3)  # 64

        # middle = self.middle(m1_4)  # 2021.9.3 change
        # endregion

        # region
        m1_1 = self.M1_1(input[8], input[11])  # 512  残差
        m2_1 = self.M2_1(input[5], m1_1)  # 256
        m3_1 = self.M3_1(input[3], m2_1)  # 128
        m4_1 = self.M4_1(input[1], m3_1)  # 64

        m1_2 = self.M1_2(m2_1, m1_1)  # 64
        m2_2 = self.M2_2(m3_1, m1_2)  # 128
        m3_2 = self.M3_2(m4_1, m2_2)  # 256

        m1_3 = self.M1_3(m2_2, m1_2)  # 128
        m2_3 = self.M2_3(m3_2, m1_3)  # 64

        m1_4 = self.M1_4(m2_3, m1_3)  # 64

        middle = self.middle(m1_4)
        # endregion

        # 底层
        # region
        # d1_1 = self.D1_1(input[6], input[9])  # 256  临近
        # d2_2 = self.D2_2(input[9], input[12])  # 512

        # d1_2 = self.D1_2(d1_1, d2_2)  # 256

        # deep = self.deep(d1_2)
        # endregion

        # region
        d1_1 = self.D1_1(input[9], input[12])  # 256  残差
        d2_2 = self.D2_2(input[6], d1_1)  # 512

        d1_2 = self.D1_2(d2_2, d1_1)  # 256

        deep = self.deep(d1_2)
        # endregion
        deep_up = F.conv_transpose2d(deep, self.deconv_weight_1, stride=self.factor, padding=int(self.factor/2),
                                     output_padding=(shallow.size(2) - deep.size(2)*self.factor,
                                     shallow.size(3) - deep.size(3)*self.factor))

        prediction = (shallow + middle + deep_up)/3

        return prediction.sigmoid()  # , shallow.sigmoid(), middle.sigmoid(), deep_up.sigmoid()


class MSANet(nn.Module):
    def __init__(self, cfgs):
        super(MSANet, self).__init__()
        self.encode = VGG13(cfgs)
        self.decode = decode()

    def forward(self, image):
        side_output = self.encode(image)
        prediction = self.decode(side_output)  # , shallow, middle, deep
        return prediction  # , shallow, middle, deep


class Cross_Entropy(nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()
        # self.weight1 = nn.Parameter(torch.Tensor([1.]))
        # self.weight2 = nn.Parameter(torch.Tensor([1.]))

    def forward(self, pred, labels):
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0.2]
        pred_neg = pred_flat[labels_flat == 0]
        total_loss = cross_entropy_per_image(pred, labels)
        # total_loss = self.weight1.pow(-2) * cross_entropy_per_image(pred, labels) + \
        #              self.weight2.pow(-2) * 0.1 * dice_loss_per_image(pred, labels) + \
        #              (1 + self.weight1 * self.weight2).log()
        return total_loss, (1-pred_pos).abs(), pred_neg


# region
# def dice(preds, labels):
#     preds = preds.view(-1)
#     labels = labels.view(-1)
#     eps = 1e-6
#     dice_1 = ((preds * labels).sum() * 2 + eps) / (preds.sum() + labels.sum() + eps)
#     dice_loss = dice_1.pow(-1)
#     return dice_loss
# def dice_loss_per_image(preds, labels):
#     total_loss = 0
#     for i, (_logit, _label) in enumerate(zip(preds, labels)):
#         total_loss += dice(_logit, _label)
#     return total_loss / len(preds)
#endregion


def cross_entropy_per_image(preds, labels):
    total_loss = 0
    for i, (_pred, _label) in enumerate(zip(preds, labels)):
        # total_loss += cross_entropy_with_weight(_pred, _label)
        total_loss += cross_entropy_with_weight_original(_pred, _label)
    return total_loss / len(preds)


def cross_entropy_with_weight_original(logits, labels, threshold=0.2, weight=1):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > threshold].clamp(eps, 1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0-eps)
    weight_pos = len(pred_neg)/(len(pred_neg)+len(pred_pos))
    weight_neg = len(pred_pos)/(len(pred_neg)+len(pred_pos))
    cross_entropy = (-weight_pos * pred_pos.log()).sum() + (-weight_neg * (1.0 - pred_neg).log()).sum()
    return cross_entropy

# region
# def cross_entropy_with_weight(logits, labels):
#     logits = logits.view(-1)
#     labels = labels.view(-1)
#     eps = 1e-6
#     pred_pos = logits[labels > 0].clamp(eps, 1.0-eps)
#     pred_neg = logits[labels == 0].clamp(eps, 1.0-eps)
#     w_anotation = labels[labels > 0]
#     cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
#                     (-(1.0 - pred_neg).log()).mean()
#     # cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
#     #                     (-(1.0 - pred_neg).log() * weight_neg).sum()
#     return cross_entropy
#
# def get_weight(src, mask, threshold, weight):
#     count_pos = src[mask >= threshold].size()[0]
#     count_neg = src[mask == 0.0].size()[0]
#     total = count_neg + count_pos
#     weight_pos = count_neg / total
#     weight_neg = (count_pos / total) * weight
#     return weight_pos, weight_neg
#endregion


def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (decay_rate ** (epoch // decay_steps))