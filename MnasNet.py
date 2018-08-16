import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from collections import OrderedDict

def conv3x3(in_channels,out_channels,stride=1,padding=1,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=padding,bias=bias,groups=groups)
def conv5x5(in_channels,out_channels,stride=1,padding=2,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=stride,padding=padding,bias=bias,groups=groups)
def conv1x1(in_channels,out_channels,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias,groups=groups)

class PrimaryModule(nn.Module):
    def __init__(self,in_channels=3,out_channels=[32,16]):
        super(PrimaryModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.PrimaryModule = nn.Sequential(
                                           OrderedDict(
                                                       [
                                                        ('PrimaryBN',nn.BatchNorm2d(in_channels)),
                                                        ('PrimaryConv3x3',conv3x3(in_channels,out_channels[0],2,1,False,1)),
                                                        ('PrimaryConv3x3BN',nn.BatchNorm2d(out_channels[0])),
                                                        ('PrimaryConv3x3ReLU',nn.ReLU(inplace=True)),
                                                        ('PrimaryDepthwiseConv3x3',conv3x3(out_channels[0],out_channels[0],1,1,False,out_channels[0])),
                                                        ('PrimaryDepthwiseConv3x3BN',nn.BatchNorm2d(out_channels[0])),
                                                        ('PrimaryDepthwiseConv3x3ReLU',nn.ReLU(inplace=True)),
                                                        ('PrimaryConv1x1',conv1x1(out_channels[0],out_channels[1],False,1)),
                                                        ('PrimaryConv1x1BN',nn.BatchNorm2d(out_channels[1])),
                                                        ('PrimaryConv1x1ReLU',nn.ReLU(inplace=True))
                                                       ]
                                                       )
                                           )
    def forward(self,x):
        x = self.PrimaryModule(x)
        return x

class MnasNetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,_conv=conv3x3,padding=1,ratio=6,isAdd=True):
        super(MnasNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self._conv = _conv
        self.padding = padding
        self.ratio = ratio
        self.isAdd = isAdd
        
        self.MnasNetBlock = nn.Sequential(
                                          OrderedDict(
                                                      [
                                                       ('unCompressConv1x1',conv1x1(in_channels,out_channels*ratio,False,1)),
                                                       ('unCompressConv1x1BN',nn.BatchNorm2d(out_channels*ratio)),
                                                       ('unCompressConv1x1ReLU',nn.ReLU(inplace=True)),
                                                       ('DepthwiseConv',_conv(out_channels*ratio,out_channels*ratio,stride,padding,False,out_channels*ratio)),
                                                       ('DepthwiseConvBN',nn.BatchNorm2d(out_channels*ratio)),
                                                       ('DepthwiseConvReLU',nn.ReLU(inplace=True)),
                                                       ('CompressConv1x1',conv1x1(out_channels*ratio,out_channels,False,1)),
                                                       ('CompressConv1x1BN',nn.BatchNorm2d(out_channels)),
                                                       ('CompressConv1x1ReLU',nn.ReLU(inplace=True))
                                                      ]
                                                      )
                                          )
    def forward(self,input):
        x = self.MnasNetBlock(input)
        if self.isAdd:
            x += input
        return x

class FinalModule(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FinalModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.FC = nn.Sequential(
                                OrderedDict(
                                            [
                                             ('Dropout',nn.Dropout(0.5)),
                                             ('FC',conv1x1(in_channels,out_channels,True,1))
                                            ]
                                            )
                                )
    def forward(self,x):
        x = F.avg_pool2d(x,x.data.size()[-2:])
        x = self.FC(x)
        return x

class MnasNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=1000):
        super(MnasNet ,self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.out_channels = [16,24,40,80,96,192,320]
        
        self.PrimaryModule = PrimaryModule(in_channels,out_channels=[32,16])
        
        self.stage1 = self.stage(1,[1,2],conv3x3,1,3)
        self.stage2 = self.stage(2,[1,2],conv5x5,2,3)
        self.stage3 = self.stage(3,[1,2],conv5x5,2,6)
        self.stage4 = self.stage(4,[0,1],conv3x3,1,6)
        self.stage5 = self.stage(5,[1,3],conv5x5,2,6)
        self.stage6 = self.stage(6,[0,0],conv3x3,1,6)
        
        self.FC = FinalModule(self.out_channels[-1],num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)
    
    def stage(self,stage=1,BlockRepeat=[1,2],_conv=conv3x3,padding=1,ratio=6):
        modules = OrderedDict()
        name = 'MnasNetStage_{}'.format(stage)
        
        if BlockRepeat[0]==0:
            modules[name+'_0'] = MnasNetBlock(self.out_channels[stage-1],self.out_channels[stage],1,_conv,padding,ratio,False)
        elif BlockRepeat[0]==1:
            modules[name+'_0'] = MnasNetBlock(self.out_channels[stage-1],self.out_channels[stage],2,_conv,padding,ratio,False)
        else:
            raise ValueError('BlockRepeat[0] must be 0 or 1')
        
        for i in range(BlockRepeat[1]):
            modules[name+'_{}'.format(i+1)] = MnasNetBlock(self.out_channels[stage],self.out_channels[stage],1,_conv,padding,ratio,True)
        
        return nn.Sequential(modules)
            
    def forward(self,x):
        x = self.PrimaryModule(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.FC(x)
        x = x.view(x.size(0),x.size(1))
        return x



if __name__=='__main__':
    net = MnasNet(3,1000)
    input = torch.randn((1,3,224,224))
    output = net(input)
    print(output.shape)
    
    params = list(net.parameters())
    num = 0
    for i in params:
        l=1
        #print('Size:{}'.format(list(i.size())))
        for j in i.size():
            l *= j
        num += l
    print('All Parameters:{}'.format(num))