import torch 
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb
import fm_ops as SPD_ops
from torch_batch_svd import svd
import time


def weightNormalize(weights):
    out = []
    epsilon = 1e-6
    for row in weights.view(weights.shape[0],-1):
        row = row**2/(torch.sum(row**2)+epsilon)
        #temp = torch.clamp(row,min=1e-8)
        #out.append(temp/torch.sum(temp))
        out.append(row)
    return torch.stack(out).view(*weights.shape)

def weightNormalize_unconst(weights):
    out = []
    multiplier = []
    epsilon = 1e-6
    for row in weights.view(weights.shape[0],-1):
        #pdb.set_trace()
        #ww = torch.abs(row)/(torch.sum(torch.abs(row))+epsilon)
        #mult = torch.abs(row)/(ww+epsilon)
        #out.append(ww)
        #multiplier.append(mult)
        ww = row**2/(torch.sum(row**2)+epsilon)
        mult = row**2/(ww+epsilon)
        out.append(ww)
        multiplier.append(mult)
        #pdb.set_trace()
        #out = torch.clamp(row,min=-1,max = 1)

    return torch.stack(out).view(*weights.shape), torch.stack(multiplier).view(*weights.shape)

def weightNormalize_abs(weights):
    out = []
    multiplier = []
    epsilon = 1e-6
    for row in weights.view(weights.shape[0],-1):
        #pdb.set_trace()
        ww = torch.abs(row)/(torch.sum(torch.abs(row))+epsilon)
        mult = row/(ww+epsilon)
        out.append(ww)
        multiplier.append(mult)
        #pdb.set_trace()
        #out = torch.clamp(row,min=-1,max = 1)

    return torch.stack(out).view(*weights.shape), torch.stack(multiplier).view(*weights.shape)



class SPDLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SPDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_matrix = torch.nn.Parameter(torch.rand(out_features, in_features),requires_grad=True)

    def forward(self, x):
        #out1 = SPD_ops.recursiveFM(x, weightNormalize(self.weight_matrix))
        #out2 = SPD_ops.recursiveFM(torch.flip(x,[1]), torch.flip(weightNormalize(self.weight_matrix),[1]))
        out3 = SPD_ops.fastFM(x, weightNormalize(self.weight_matrix))

        return out3 

class SPDLinear_recur(nn.Module):
    def __init__(self, in_features, out_features):
        super(SPDLinear_recur, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_matrix = torch.nn.Parameter(torch.rand(out_features, in_features),requires_grad=True)

    def forward(self, x):
        out1 = SPD_ops.recursiveFM(x, weightNormalize(self.weight_matrix))
        #out2 = SPD_ops.recursiveFM(torch.flip(x,[1]), torch.flip(weightNormalize(self.weight_matrix),[1]))
        #out3 = SPD_ops.fastFM(x, weightNormalize(self.weight_matrix))

        return out1

class SPDLinear_unconst0(nn.Module):
    def __init__(self, in_features, out_features):
        super(SPDLinear_unconst0, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #weight_matrix =torch.nn.Parameter(weightNormalize(torch.rand(out_features, in_features)),requires_grad=True)
        self.weight_matrix = torch.nn.Parameter(torch.rand(out_features, in_features),requires_grad=True)
        self.scale_matrix = torch.nn.Parameter(torch.rand(out_features, in_features),requires_grad=True)

    def forward(self, x):
        ####################################################################
        #norm_weight, multiplier = weightNormalize_unconst(self.weight_matrix)
        #print(norm_weight, multiplier)
        #pdb.set_trace()
        ####################################################################
        return SPD_ops.fastFM_unconst(x, weightNormalize(self.weight_matrix), self.scale_matrix)


class SPDLinear_unconst1(nn.Module):
    def __init__(self, in_features, out_features):
        super(SPDLinear_unconst1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #weight_matrix =torch.nn.Parameter(weightNormalize(torch.rand(out_features, in_features)),requires_grad=True)
        self.weight_matrix = torch.nn.Parameter(torch.rand(out_features, in_features),requires_grad=True)
        self.scale_matrix = torch.nn.Parameter(torch.rand(out_features, in_features),requires_grad=True)

    def forward(self, x):
        ####################################################################
        #norm_weight, multiplier = weightNormalize_unconst(self.weight_matrix)
        #print(norm_weight, multiplier)
        #pdb.set_trace()
        ####################################################################
        return SPD_ops.fastFM_unconst(x, weightNormalize(self.weight_matrix), torch.clamp(self.scale_matrix, max = 1, min = -1))


class SPDLinear_unconst2(nn.Module):
    def __init__(self, in_features, out_features):
        super(SPDLinear_unconst2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #pdb.set_trace()
        self.weight_matrix =torch.nn.Parameter(weightNormalize(torch.rand(out_features, in_features)),requires_grad=True)
        #self.weight_matrix = torch.nn.Parameter(torch.rand(out_features, in_features),requires_grad=True)
        #self.multiple_matrix = torch.nn.Parameter(torch.rand(out_features, in_features),requires_grad=True)
        
    def forward(self, x):
        ####################################################################
        norm_weight, self.scale_matrix = weightNormalize_unconst(self.weight_matrix)
        #print(norm_weight, multiplier)
        #pdb.set_trace()


        ####################################################################
        return SPD_ops.fastFM_unconst(x, norm_weight, torch.clamp(self.scale_matrix, max = 1, min = 1e-3))



class SPDLinear_unconst3(nn.Module):
    def __init__(self, in_features, out_features):
        super(SPDLinear_unconst3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_matrix =torch.nn.Parameter(weightNormalize(torch.rand(out_features, in_features)),requires_grad=True)

    def forward(self, x):
        ####################################################################
        norm_weight, self.scale_matrix = weightNormalize_abs(self.weight_matrix)


        ####################################################################
        return SPD_ops.fastFM_unconst(x, norm_weight, torch.clamp( self.scale_matrix,max = 1, min = -1))


class SPDLinear_unconst4(nn.Module):
    def __init__(self, in_features, out_features):
        super(SPDLinear_unconst4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_matrix =torch.nn.Parameter(weightNormalize(torch.rand(out_features, in_features)),requires_grad=True)

    def forward(self, x):
        ####################################################################
        norm_weight, self.scale_matrix = weightNormalize_abs(self.weight_matrix)


        ####################################################################
        return SPD_ops.fastFM_unconst(x, norm_weight, self.scale_matrix)
















class SPDConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride):
        super(SPDConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.weight_matrix = torch.nn.Parameter(torch.rand(out_channels, (kern_size**2)*in_channels),requires_grad=True)


    # x: [batches, channels, rows, cols, 3, 3]
    def forward(self, x):
       # x: [batches, channels, rows, cols, 3, 3] -> 
        #    [batches, channels, 3, 3, rows, cols]
        x = x.permute(0,1,4,5,2,3).contiguous()

        # x_windows: [batches, channels, 3, 3, rows_reduced, cols_reduced, window_x, window_y]
        x_windows = x.unfold(4, self.kern_size, self.stride).contiguous()
        x_windows = x_windows.unfold(5, self.kern_size, self.stride).contiguous()

        x_s = x_windows.shape
        #x_windows: [batches, channels, 3, 3,  rows_reduced, cols_reduced, window]   
        x_windows = x_windows.view(x_s[0],x_s[1],x_s[2],x_s[3],x_s[4],x_s[5],-1)

        #x_windows: [batches, rows_reduced, cols_reduced, window, channels, 3,3]
        x_windows = x_windows.permute(0,4,5,6,1,2,3).contiguous()

        x_s = x_windows.shape
        x_windows = x_windows.view(x_s[0],x_s[1],x_s[2],-1,x_s[5],x_s[6]).contiguous()


        #Output format: [batches, sequence, out_channels, cov_x, cov_y]
        return spd_ops.recursiveFM2D(x_windows, weightNormalize(self.weight_matrix)), 0



class SPD_to_vec(nn.Module):
    def __init__(self):
        super(SPD_to_vec, self).__init__()
        self.A = torch.rand(2,288).cuda()

    #X: [-1, 3,3]
    #Y: [-1, 3,3]
    def GLmetric(self, X, Y):
        inner = torch.matmul(torch.inverse(X), Y)


        u,s,v = svd(inner)
        s_log = torch.diag_embed(torch.log(s))
        log_term = torch.matmul(u,torch.matmul(s_log,v.permute(0,2,1)))
        dist = torch.sum(torch.diagonal(torch.matmul(log_term,log_term), dim1=-2, dim2=-1),1)
        return dist
    
    #x: [batch, channels, rows, cols, 3,3]
    def forward(self, x):
        x_s = x.shape

        #x: [batch*channels, rows*cols, 3,3]
        x = x.view(x.shape[0]*x.shape[1], -1, x.shape[4], x.shape[5])

        #x: [batch*channels, 1, 1, rows*cols, 3,3]
        x = x.unsqueeze(1).unsqueeze(2)


        #weights: [1,rows*cols-1]
        weights = (1.0/torch.arange(start=2.0,end=x.shape[3]+1)).unsqueeze(0).cuda()
        
        #unweightedFM: [batches*channels, 1,1,1, 3,3]
        unweighted_FM = spd_ops.recursiveFM2D(x,weights)


        #unweightedFM: [batches*channels,3,3]
        unweighted_FM = unweighted_FM.view(-1, x_s[4], x_s[5])
        
        #unweightedFM: [batches*channels,rows*cols,3,3]
        unweighted_FM = unweighted_FM.unsqueeze(1).repeat(1, x_s[2]*x_s[3], 1, 1)

        #unweightedFM: [batches*channels*rows*cols,3,3]
        unweighted_FM = unweighted_FM.view(-1, x_s[4], x_s[5])


        #x: [batches*channels,rows*cols,3,3]
        x = x.view(-1, x_s[2]*x_s[3], x_s[4], x_s[5])
        #x: [batches*channels*rows*cols,3,3]
        x = x.view(-1, x_s[4], x_s[5])

        out = self.GLmetric(x, unweighted_FM)

        #out: [batch, channels*rows*cols]
        out = out.view(x_s[0], x_s[1]*x_s[2]*x_s[3])


        return out

