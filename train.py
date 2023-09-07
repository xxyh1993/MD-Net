import torch
import yaml
from matplotlib import pyplot as plt
import numpy as np
import transforms
from data import BSDS_500  # NYUD, PASCAL_Context, PASCAL_VOC12
import model_RDN
import cv2
import time
import os
from datetime import datetime
from visdom import Visdom
from torchnet import  
from visual_loss import Visualizer

if __name__ == '__main__':
    vis_loss = Visualizer(env='BSDS-loss')  # set the name of window
    loss_meter = meter.AverageValueMeter()  # 为了可视化增加的内容
    # load configures
    file_id = open('./cfgs.yaml', 'r', encoding='UTF-8')
    cfgs = yaml.load(file_id, Loader=yaml.FullLoader)
    file_id.close()
    if not os.path.exists('./val/'):
        os.makedirs('./val/')
    if not os.path.exists('./checkpoint/'):  # 存放模型的文件夹
        os.makedirs('./checkpoint/')
    if not os.path.exists('./outside/'):  # 存放其他预测的文件夹
        os.makedirs('./outside/')
    
    trans = transforms.Compose([
        transforms.RandomResizedCrop(320, scale=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[[0.485, 0.456, 0.406], [0.]],
                             std=[[0.229, 0.224, 0.225], [1.]])
    ])

    dataset = BSDS_500(root=cfgs['dataset'], VOC=True, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfgs['batch_size'], shuffle=True, pin_memory=True, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    net = model_RDN.MSANet(cfgs['vgg16']).train()
    # loss
    criterion = model_RDN.Cross_Entropy()
    # optimal
    if cfgs['method'] == 'Adam':
        optimizer = torch.optim.Adam([{'params': net.parameters()},
                                      {'params': criterion.parameters()}],
                                     weight_decay=cfgs['weight_decay'])
    elif cfgs['method'] == 'SGD':
        optimizer = torch.optim.SGD([{'params': net.parameters()},
                                     {'params': criterion.parameters()}],
                                     lr=cfgs['lr'],
                                     momentum=cfgs['momentum'],
                                     weight_decay=cfgs['weight_decay'])
    # GPU
    net.to(device)
    criterion.to(device)
    # 断点续传
    start_epoch = -1
    if os.path.exists('./checkpoint/model.pth') and torch.load('./checkpoint/model.pth')['epoch'] != (cfgs['max_epoch']-1):  #如果模型存在并且
        interrupt = torch.load('./checkpoint/model.pth')
        net.load_state_dict(interrupt['model'])  # 加载模型的可学习参数
        optimizer.load_state_dict(interrupt['optimizer'])  # 加载优化器参数
        start_epoch = torch.load('./checkpoint/model.pth')['epoch']  # 设置开始的epoch
        # optimizer.state_dict()['param_groups'][0]['lr'] = interrupt['lr']  #恢复学习率  这两句其实可以不写
        # optimizer.state_dict()['param_groups'][1]['lr'] = interrupt['lr']
        print('已恢复训练')

    # train
    for epoch in range(start_epoch + 1, cfgs['max_epoch']):  # loop over the dataset multiple times
        loss_meter.reset()  # 为了增加可视化内容
        model_RDN.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'], decay_steps=cfgs['decay_steps'])
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print(optimizer.state_dict()['param_groups'][1]['lr'])
        running_loss = 0.0
        for i, data in enumerate(dataloader, start=0):
            start_time = time.time()
            optimizer.zero_grad()

            images = data['images'].to(device)  # 对应获取响应的输入特征数据，注意键值(关键字) ：images、img03、labels这些都是在data文件自行定义的，以索引对应的特征数据
            labels = data['labels'].to(device)

            prediction = net(images)  # , shallow, middle, deep
            # prediction = net(images)
            loss, dp, dn = criterion(prediction, labels)

            loss.backward()
            optimizer.step()
            
            # 可视化loss
            loss_meter.add(loss.item())
            vis_loss.plot_many_stack({'BSDS_train_loss': loss_meter.value()[0]})

            duration = time.time() - start_time

            print_epoch = 100
            running_loss += loss.item()
            # 每十个batchsize输出一个平均loss  窗口打印训练过程中的 loss
            if i % print_epoch == print_epoch - 1:
                examples_per_sec = 2 / duration
                sec_per_batch = float(duration)
                format_str = '%s: step [%d, %5d/%4d], loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), epoch + 1, i + 1, len(dataloader), running_loss / 100,
                                    examples_per_sec, sec_per_batch))
                running_loss = 0.0
            # validation，保存训练过程中的图片，主要是用来观察出来的结果对不对
            validation_epoch = 100
            if i % validation_epoch == validation_epoch - 1:
                prediction = prediction.cpu().detach().numpy().transpose((0, 2, 3, 1))  # 通过转置变成（batchsize，高，宽，通道数），不然后面没法保存图片
                for j in range(prediction.shape[0]):
                    cv2.imwrite('./val/' + str(j+1) + '.png', prediction[j] * 255)

                ax = plt.subplot(1, 2, 1)
                data_ = dp.cpu().detach().numpy()
                ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))
                ax = plt.subplot(1, 2, 2)
                data_ = dn.cpu().detach().numpy()
                ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))
                plt.savefig('./val/test' + str(epoch+1) + '.png')
                plt.close('all')

                # 保存某一张的特征图
                # out = shallow[0].detach().cpu().numpy()  # .detach()返回一个新的CUDA tensor，但是该tensor不在具有梯度。从GPU取回CPU, 需要先将其转换成cpu float-tensor, 随后再转到numpy格式。 s1[1]表示取出batchsize中的第二张图
                # for j in range(out.shape[0]):
                #     cv2.imwrite('./outside/' + str('shallow') + '.png', out[j] * 255)

                # out = middle[0].detach().cpu().numpy()  # .detach()返回一个新的CUDA tensor，但是该tensor不在具有梯度。从GPU取回CPU, 需要先将其转换成cpu float-tensor, 随后再转到numpy格式。 s1[1]表示取出batchsize中的第二张图
                # for j in range(out.shape[0]):
                #     cv2.imwrite('./outside/' + str('middle') + '.png', out[j] * 255)
                
                # out = deep[0].detach().cpu().numpy()  # .detach()返回一个新的CUDA tensor，但是该tensor不在具有梯度。从GPU取回CPU, 需要先将其转换成cpu float-tensor, 随后再转到numpy格式。 s1[1]表示取出batchsize中的第二张图
                # for j in range(out.shape[0]):
                #     cv2.imwrite('./outside/' + str('deep') + '.png', out[j] * 255)           

        # 保存每个epoch的的信息
        state = {'model': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}  # epoch从0开始  'lr': optimizer.state_dict()['param_groups'][0]['lr']
        torch.save(state, './checkpoint/' + cfgs['save_name'])  # torch.save(net, path)保存整个模型(结构+参数)  torch.save(net.state_dict(), path)这句话是只保存模型参数，但是会占用存储比较大。

    print('Finished Training')
