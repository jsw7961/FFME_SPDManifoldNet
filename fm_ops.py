import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_batch_svd import svd
import pdb

def weightNormalize(weights):
    out = []
    for row in weights.view(weights.shape[0],-1):
        row = row**2/torch.sum(row**2)
        #temp = torch.clamp(row,min=1e-8)
        #out.append(temp/torch.sum(temp))
    return torch.stack(out).view(*weights.shape)


def mexp(x, exp):
    #u,s,v = svd(x)
    u,s,v = svd(x)
    ep = torch.diag_embed(torch.pow(s,exp))
    v = torch.einsum('...ij->...ji', v)
    return torch.matmul(torch.matmul(u,ep),v)

def dist_(X, Y):
    inner = torch.matmul(torch.inverse(X), Y)
    #u,s,v = svd(inner)
    u,s,v = svd(inner)
    s_log = torch.diag_embed(torch.log(s))
    log_term = torch.matmul(u,torch.matmul(s_log,v.permute(0,2,1)))
    dist = torch.sum(torch.diagonal(torch.matmul(log_term,log_term), dim1=-2, dim2=-1),1)
    return dist
    
def log_map(SS):
    u,s,v = svd(SS)
    log_s = torch.log(s)
    s_mat = torch.diag_embed(log_s)
    return torch.matmul(u, torch.matmul(s_mat, v.permute(0,2,1)))
    
def exp_map(SS):
    u,s,v = svd(SS)
    exp_s = torch.exp(s)
    s_mat = torch.diag_embed(exp_s)
    return torch.matmul(u, torch.matmul(s_mat, v.permute(0,2,1)))

def Rot_multiplier(SS, mult):
    U,S,V = svd(SS)
    #pdb.set_trace()
    S_pow = torch.diag_embed(torch.pow(S,mult.unsqueeze(1).repeat(1,3)))
    return torch.matmul(U, torch.matmul(S_pow, V.permute(0,2,1)))


#M: [-1,3,3] 
#N: [-1,3,3]
#w: [-1]
def batchSPDMean(S,M,ww):
    
    # w:[-1, 3, 3]

    WW = torch.sum(ww,axis=1)
    normalized_w = ww[:,-1]/WW # t

    #u,s,v = svd(M)
    u,s,v = svd(M)

    s_pow = torch.diag_embed(torch.pow(s,0.5))
    
    M_sqrt = torch.matmul(u, torch.matmul(s_pow, v.permute(0,2,1)))

    M_sqrt_inv = torch.inverse(M_sqrt)

    inner_term = torch.matmul(M_sqrt_inv, torch.matmul(S, M_sqrt_inv))

    
    #u_i, s_i, v_i = svd(inner_term)
    u_i, s_i, v_i = svd(inner_term)

    s_i_c = s_i.view(-1)
    #pdb.set_trace()
    s_i_c_pow = s_i_c**(normalized_w.unsqueeze(1).repeat(1,3).view(-1))
    s_i_pow = s_i_c_pow.view(*s.shape)

    s_i_pow = torch.diag_embed(s_i_pow)

    inner_term_weighted = torch.matmul(u_i, torch.matmul(s_i_pow, v_i.permute(0,2,1)))


    return torch.matmul(M_sqrt, torch.matmul(inner_term_weighted, M_sqrt))


def batchSPDMean_f(S1,S2,w1,w2):
    
    # w:[-1, 3, 3]
    epsilon = 1e-10
    normalized_w = w2/(w1+w2+epsilon) # t

    # w1 1, w2 0 -> t = 0 -> S1
    # w1 0, w2 1 -> t = 1 -> S2

    #u,s,v = svd(M)
    u,s,v = svd(S1)

    s_pow = torch.diag_embed(torch.pow(s,0.5))
    
    M_sqrt = torch.matmul(u, torch.matmul(s_pow, v.permute(0,2,1)))

    M_sqrt_inv = torch.inverse(M_sqrt)

    inner_term = torch.matmul(M_sqrt_inv, torch.matmul(S2, M_sqrt_inv))

    
    #u_i, s_i, v_i = svd(inner_term)
    u_i, s_i, v_i = svd(inner_term)

    s_i_c = s_i.view(-1)
    #pdb.set_trace()
    s_i_c_pow = s_i_c**(normalized_w.unsqueeze(1).repeat(1,3).view(-1))
    s_i_pow = s_i_c_pow.view(*s.shape)

    s_i_pow = torch.diag_embed(s_i_pow)

    inner_term_weighted = torch.matmul(u_i, torch.matmul(s_i_pow, v_i.permute(0,2,1)))


    return torch.matmul(M_sqrt, torch.matmul(inner_term_weighted, M_sqrt))



#windows: [batches, rows_reduced, cols_reduced, window, 3, 3]
#weights: [out_channels, in_channels*kern_size**2}
def recursiveFM2D(windows, weights):
    w_s = windows.shape

    # windows: [batches*rows_reduced*cols_reduced, window, 3, 3]
    windows = windows.view(-1, windows.shape[3], windows.shape[4], windows.shape[5])

    oc = weights.shape[0]

    # weights: [batches*rows_reduced*cols_reduced, out_channels, in_channels*kern_size**2]\
    weights = weights.unsqueeze(0).repeat(windows.shape[0],1,1)

    # weights: [batches*rows_reduced*cols_reduced*out_channels, in_channels*kern_size**2]\
    weights = weights.view(-1, weights.shape[2])

    # [batches*rows_reduced*cols_reduced*channels_out, 3,3]
    running_mean = windows[:,0,:,:].unsqueeze(1).repeat(1,oc,1,1)
    running_mean = running_mean.view(-1,running_mean.shape[2], running_mean.shape[3])


    for i in range(1,weights.shape[1]):
        current_fiber = windows[:,i,:,:]
        
        #[batches*rows_reduced*cols_reduced, channels_out, 3, 3]
        current_fiber = current_fiber.unsqueeze(1).repeat(1,oc,1,1)
        cf_s = current_fiber.shape
        
        # [batches*rows_reduced*cols_reduced*channels_out, 3, 3]
        current_fiber = current_fiber.view(-1, cf_s[2], cf_s[3])


        running_mean = batchSPDMean(current_fiber, running_mean, weights[:,i])

    #out: [batches, rows_reduced, cols_reduced, channels_out, 3, 3]
    out = running_mean.view(w_s[0], w_s[1], w_s[2], oc, w_s[4], w_s[5])

    #out: [batches, channels_out, rows_reduced, cols_reduced, 3, 3]
    out = out.permute(0,3,1,2,4,5).contiguous()
    
    return out


def recursiveFM(windows, weights):
    # w_s : [batches, inpuut_channels,3,3]
    w_s = windows.shape
    oc = weights.shape[0] 
    if weights.shape[1] != windows.shape[1]:
        print('input channel size is not matched with weight input channel size')
        sys.exit()
        #exit()
    #pdb.set_trace()
    weights = weights.unsqueeze(0).repeat(windows.shape[0],1,1)
    weights = weights.view(-1, weights.shape[2])

    running_M = windows[:,0,:,:].unsqueeze(1).repeat(1,oc,1,1)
    running_M = running_M.view(-1,running_M.shape[2], running_M.shape[3])
    
    for i in range(1,weights.shape[1]):
        #print(i)
        
        current_mat = windows[:,i,:,:].unsqueeze(1).repeat(1,oc,1,1)
        cf_s = current_mat.shape
        current_mat = current_mat.view(-1, cf_s[2], cf_s[3])
        running_M = batchSPDMean(current_mat, running_M, weights[:,:i+1])

    out = running_M.view(w_s[0], oc, w_s[2], w_s[3])

    return out



def fastFM(windows, weights):
    # w_s : [batches, inpuut_channels,3,3]
    w_s = windows.shape
    weights_s = weights.shape
    #pdb.set_trace()
    oc = weights.shape[0] 
    if weights.shape[1] != windows.shape[1]:
        print(weights.shape[1], windows.shape[1])
        print('input channel size is not matched with weight input channel size')
        
        sys.exit()

    weights = weights.unsqueeze(0).repeat(windows.shape[0],1,1)
    weights = weights.view(-1, weights.shape[2])

    windows = windows.unsqueeze(1).repeat(1,oc,1,1,1).view(-1,weights_s[1],w_s[-2],w_s[-1])

    while windows.shape[-3] != 1:
        #print(windows.shape[-3])
        new_w_s = windows.shape
        #if torch.isnan(windows).any() > 0:
        #    pdb.set_trace()
        #if torch.isnan(weights).any() > 0:
        #    pdb.set_trace()
        if new_w_s[1]%2 ==0 :
            Rot_b = windows[:,1::2,:,:].reshape(-1,new_w_s[-2],new_w_s[-1])
            Rot_f = windows[:,0::2,:,:].reshape(-1,new_w_s[-2],new_w_s[-1])
            new_wt_s = weights.shape
            weights_b = weights[:,1::2].reshape(-1)
            weights_f = weights[:,0::2].reshape(-1)
            windows = batchSPDMean_f(Rot_f,Rot_b,weights_f,weights_b).reshape(new_w_s[0],-1,new_w_s[-2],new_w_s[-1])
            weights = (weights_f+weights_b).reshape(new_wt_s[0],-1)

        else :
            Rot_r = windows[:,-1,:,:].unsqueeze(1)
            weights_r = weights[:,-1].unsqueeze(1)

            Rot_b = windows[:,1::2,:,:].reshape(-1,new_w_s[-2],new_w_s[-1])
            Rot_f = windows[:,0:-1:2,:,:].reshape(-1,new_w_s[-2],new_w_s[-1])
            
            new_wt_s = weights.shape
            weights_b = weights[:,1::2].reshape(-1)
            weights_f = weights[:,0:-1:2].reshape(-1)
            windows = batchSPDMean_f(Rot_f,Rot_b,weights_f,weights_b).reshape(new_w_s[0],-1,new_w_s[-2],new_w_s[-1])
            weights = (weights_f+weights_b).reshape(new_wt_s[0],-1)
            windows = torch.cat((windows,Rot_r),axis=1)
            weights = torch.cat((weights,weights_r),axis=1)
    
    #out : [batches, out_channel, 3, 3]
    out = windows.view(w_s[0], oc, w_s[2], w_s[3])
    
    return out


def fastFM_unconst(windows, weights, multiplier):
    # w_s : [batches, inpuut_channels,3,3]
    w_s = windows.shape
    weights_s = weights.shape
    #pdb.set_trace()
    oc = weights.shape[0] 
    if weights.shape[1] != windows.shape[1]:
        print(weights.shape[1], windows.shape[1])
        print('input channel size is not matched with weight input channel size')
        
        sys.exit()

    weights = weights.unsqueeze(0).repeat(windows.shape[0],1,1)
    weights = weights.view(-1, weights.shape[2])

    multiplier = multiplier.unsqueeze(0).repeat(windows.shape[0],1,1) 
    multiplier = multiplier.view(-1, multiplier.shape[2])

    windows = windows.unsqueeze(1).repeat(1,oc,1,1,1).view(-1,weights_s[1],w_s[-2],w_s[-1])

    w_ss = windows.shape

    windows = Rot_multiplier(windows.view(-1,w_s[-2],w_s[-1]), multiplier.view(-1)).view(w_ss)

    while windows.shape[-3] != 1:
        #print(windows.shape[-3])
        new_w_s = windows.shape
        #if torch.isnan(windows).any() > 0:
        #    pdb.set_trace()
        #if torch.isnan(weights).any() > 0:
        #    pdb.set_trace()
        if new_w_s[1]%2 ==0 :
            Rot_b = windows[:,1::2,:,:].reshape(-1,new_w_s[-2],new_w_s[-1])
            Rot_f = windows[:,0::2,:,:].reshape(-1,new_w_s[-2],new_w_s[-1])
            new_wt_s = weights.shape
            weights_b = weights[:,1::2].reshape(-1)
            weights_f = weights[:,0::2].reshape(-1)
            windows = batchSPDMean_f(Rot_f,Rot_b,weights_f,weights_b).reshape(new_w_s[0],-1,new_w_s[-2],new_w_s[-1])
            weights = (weights_f+weights_b).reshape(new_wt_s[0],-1)

        else :
            Rot_r = windows[:,-1,:,:].unsqueeze(1)
            weights_r = weights[:,-1].unsqueeze(1)

            Rot_b = windows[:,1::2,:,:].reshape(-1,new_w_s[-2],new_w_s[-1])
            Rot_f = windows[:,0:-1:2,:,:].reshape(-1,new_w_s[-2],new_w_s[-1])
            
            new_wt_s = weights.shape
            weights_b = weights[:,1::2].reshape(-1)
            weights_f = weights[:,0:-1:2].reshape(-1)
            windows = batchSPDMean_f(Rot_f,Rot_b,weights_f,weights_b).reshape(new_w_s[0],-1,new_w_s[-2],new_w_s[-1])
            weights = (weights_f+weights_b).reshape(new_wt_s[0],-1)
            windows = torch.cat((windows,Rot_r),axis=1)
            weights = torch.cat((weights,weights_r),axis=1)
    
    #out : [batches, out_channel, 3, 3]
    out = windows.view(w_s[0], oc, w_s[2], w_s[3])
    
    return out


