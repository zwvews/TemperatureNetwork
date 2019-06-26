import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb

''' 
    
    This Network is designed for Few-Shot Learning Problem. 

'''


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def define_Net(pretrained=False, model_root=None, which_model='relationNet', norm='batch', init_type='normal', use_gpu=True, **kwargs):
    CovarianceNet = None
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model == 'relationNet_64':
        CovarianceNet = RelationNet_64(norm_layer=norm_layer, **kwargs)
    elif which_model == 'relationNet_128':
        CovarianceNet = RelationNet_128(norm_layer=norm_layer, **kwargs)
    elif which_model == 'relationNet_256':
        CovarianceNet = RelationNet_256(norm_layer=norm_layer, **kwargs)
    elif which_model == 'RelationNet_residual':
        CovarianceNet = RelationNet_residual(Bottleneck, norm_layer=norm_layer, **kwargs)
    elif which_model == 'relationNet2':
        CovarianceNet = RelationNet2(norm_layer=norm_layer, **kwargs)
    elif which_model == 'alexnetNet':
        CovarianceNet = AlexnetNet(norm_layer=norm_layer, **kwargs)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)
    init_weights(CovarianceNet, init_type=init_type)

    if use_gpu:
        CovarianceNet.cuda()

    if pretrained:
        CovarianceNet.load_state_dict(model_root)

    return CovarianceNet


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



##############################################################################
# Classes: RelationNet_residual
##############################################################################

# Model: RelationNet_residual --- using residual block
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Covariance matrix layer --> Classifier layer  
# Dataset: 28 x 28 x 3, for example omniglot or mnist
#          84 x 84 x 3, for miniImageNet
# Filters: 64->64->96->128->256
# Mapping Sizes: 84->42->42->42->21->21


class RelationNet_residual(nn.Module):
    def __init__(self, block, norm_layer=nn.BatchNorm2d, num_classes=5):
        super(RelationNet_residual, self).__init__()
        self.inplanes = 64
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(               # 3*28*28      3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64*14*14     64*42*42
            self._make_layer(block, 64),
            self._make_layer(block, 96),
            self._make_layer(block, 128, stride=2),
            self._make_layer(block, 256),
        )
        
        self.covariance = CovaBlock2()  # 1*(49*num_classes)       1*(441*num_classes)

        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=441, stride=441, bias=use_bias),
        )

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion


        return nn.Sequential(*layers)



    def forward(self, input1, input2):

        # extract features of input1--query image
        pdb.set_trace()
        q = self.features(input1)
    

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            S.append(self.features(input2[i]))


        x = self.covariance(q, S) # get Batch*1*(h*w*num_classes)
        x = self.classifier(x)    # get Batch*1*num_classes
        x = x.squeeze(1)          # get Batch*num_classes

        return x



#========================== Define a resnet block ==========================#

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                norm_layer(dim),
                nn.ReLU(True)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




##############################################################################
# Classes: RelationNet_256
##############################################################################

# Model: RelationNet_256 --- using droput layers and leaky relu
# Input: One query image and a support set, both of them have the same class
# Base_model: 4 Convolutional layers --> Covariance matrix layer --> Classifier layer  
# Dataset: 28 x 28 x 3, for example omniglot or mnist
#          84 x 84 x 3, for miniImageNet
# Filters: 64->96->128->256
# Mapping Sizes: 84->42->21->21->21


class RelationNet_256(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5):
        super(RelationNet_256, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(               # 3*28*28      3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64*14*14     64*42*42

            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(96),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 96*7*7       96*21*21

            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),                 # 128*7*7      128*21*21

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),                 # 256*7*7      256*21*21
        )
        
        self.covariance = CovaBlock2()  # 1*(49*num_classes)       1*(441*num_classes)

        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=441, stride=441, bias=use_bias),
        )


    def forward(self, input1, input2):

        # extract features of input1--query image
        q = self.features(input1)

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            S.append(self.features(input2[i]))

        x = self.covariance(q, S) # get Batch*1*(h*w*num_classes)
        x = self.classifier(x)    # get Batch*1*num_classes
        x = x.squeeze(1)          # get Batch*num_classes

        return x





##############################################################################
# Classes: RelationNet_128
##############################################################################

# Model: RelationNet_128 --- using droput layers and leaky relu
# Input: One query image and a support set, both of them have the same class
# Base_model: 4 Convolutional layers --> Covariance matrix layer --> Classifier layer  
# Dataset: 28 x 28 x 3, for example omniglot or mnist
#          84 x 84 x 3, for miniImageNet
# Filters: 64->64->128->128
# Mapping Sizes: 84->42->21->21->21


class RelationNet_128(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5):
        super(RelationNet_128, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(               # 3*28*28      3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64*14*14     64*42*42

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64*7*7       64*21*21

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),                 # 128*7*7       128*21*21

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),                 # 128*7*7       128*21*21
        )
        
        self.covariance = CovaBlock2()  # 1*(49*num_classes)       1*(441*num_classes)

        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=441, stride=441, bias=use_bias),
        )


    def forward(self, input1, input2):

        # extract features of input1--query image
        q = self.features(input1)

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            S.append(self.features(input2[i]))

        x = self.covariance(q, S) # get Batch*1*(h*w*num_classes)
        x = self.classifier(x)    # get Batch*1*num_classes
        x = x.squeeze(1)          # get Batch*num_classes

        return x






##############################################################################
# Classes: RelationNet_64
##############################################################################

# Model: RelationNet_64 --- using droput layers and leaky relu
# Input: One query image and a support set, both of them have the same class
# Base_model: 4 Convolutional layers --> Covariance matrix layer --> Classifier layer  
# Dataset: 28 x 28 x 3, for example omniglot or mnist
#          84 x 84 x 3, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class RelationNet_64(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5):
        super(RelationNet_64, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(               # 3*28*28      3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64*14*14     64*42*42

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64*7*7       64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),                 # 64*7*7       64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),                 # 64*7*7       64*21*21
        )
        
        self.covariance = CovaBlock2()  # 1*(49*num_classes)       1*(441*num_classes)

        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=441, stride=441, bias=use_bias),
        )

        self.pdist = nn.PairwiseDistance(p=1)
    def forward(self, input1, input2, temprature_P, temprature_N, target,num_shot):

        # extract features of input1--query image
        q = self.features(input1)
        if torch.sum(torch.isnan(q))>0:
            print("NAN")

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            S.append(self.features(input2[i]))
        listsim = []

        q2 = q.view(q.shape[0], 64, 441)
        q2square = torch.sum(q2 ** 2, dim=1, keepdim=True).repeat(1, S[0].shape[0] * 441, 1)

        for i in range(len(S)):
            support1 = S[i]
            support2 = support1.squeeze()
            if num_shot == 5:
                support2 = support2.transpose(1,0).contiguous()

            support3 = support2.view(64, support1.shape[0]*441)
            support = support3.repeat(q.size()[0],1,1)
            #pair wise distance knroc product
            supportsquare = torch.sum(support**2,dim=1,keepdim = True).repeat(1,441,1)
            dis_q_support = nn.functional.relu(supportsquare + q2square.permute(0,2,1 ) - 2* torch.bmm(q2.permute(0,2,1),support)) + 0.000000001
            if self.training:
                temparature_all = torch.ones(50, 1)*temprature_N
                # temparature_all[(target-i).nonzero()] = temprature_N
                temparature_all[(target-i) == 0] = temprature_P
                # if torch.sum(temparature_all.nonzero()) < 75:
                #     print("b")
                # if torch.sum(dis_q_support <= 0) > 0:
                #     print("c")
                dis_q_support_sqrt_exp = torch.exp(-(dis_q_support.sqrt())/temparature_all.unsqueeze(1).expand(-1,441,441*support1.shape[0]).cuda()) # temperature mean
            else:
                dis_q_support_sqrt_exp = torch.exp(-(dis_q_support.sqrt()) / temprature_P)  # temperature mean
            sim_mean = torch.mean(dis_q_support_sqrt_exp, 2)
            sim_std = torch.var(dis_q_support_sqrt_exp, 2)
            # if (torch.sum(sim_std==0)>0):
            #     print(self.features._modules['0']._parameters['weight'][0,0,0,0])
            #     print("std==0")
            sim_sum = sim_mean * (sim_std+0.000000001).sqrt()#linear layer?
            listsim.append(sim_sum)
        x = torch.cat(listsim, 1).unsqueeze(2).permute(0, 2, 1)
        # x = self.covariance(q, S) # get Batch*1*(h*w*num_classes)

        x = self.classifier(x)    # get Batch*1*num_classes
        x = x.squeeze(1)          # get Batch*num_classes

        return x



#========================== Define a Covariance layer ==========================#
# Calculate the local covariance matrix of each category in the support set
# Calculate the Covariance Metric between a query sample and a category


class CovaBlock2(nn.Module):
    def __init__(self):
        super(CovaBlock2, self).__init__()


    # calculate the covariance matrix 
    def cal_covariance(self, input):

        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)
            support_set_sam = support_set_sam-mean_support

            covariance_matrix = support_set_sam@torch.transpose(support_set_sam, 0, 1)
            covariance_matrix = torch.div(covariance_matrix, h*w*B-1)
            CovaMatrix_list.append(covariance_matrix)

        return CovaMatrix_list    


    # calculate the mahalanobis distance  
    def cal_mahalanobis(self, input, CovaMatrix_list):

        B, C, h, w = input.size()
        Maha_list = []
    
        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            mean_query = torch.mean(query_sam, 1, True)
            query_sam = query_sam-mean_query

            if torch.cuda.is_available():
                maha_dis = torch.zeros(1, len(CovaMatrix_list)*h*w).cuda()

            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1)@CovaMatrix_list[j]@query_sam
                maha_dis[0, j*h*w:(j+1)*h*w] = temp_dis.diag()

            Maha_list.append(maha_dis.unsqueeze(0))

        Maha_Dis = torch.cat(Maha_list, 0) # get Batch*1*(h*w*num_classes)
        return Maha_Dis 


    def forward(self, x1, x2):

        CovaMatrix_list = self.cal_covariance(x2)
        Maha_Dis = self.cal_mahalanobis(x1, CovaMatrix_list)

        return Maha_Dis





