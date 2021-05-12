import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import pdb

def unfold_dilation(x, dim, kernel_size, dilation = 1):
    x_s = x.shape

    t1 = x_s[:dim]
    t2 = tuple([x_s[dim]-dilation*(kernel_size-1)])
    t3 = x_s[dim+1:]
    t4 = tuple([kernel_size])
    out = torch.zeros(size=t1+t2+t3+t4,device=x.device)
    o_s = out.shape
    
    for ix in range(dilation):
        temp = x.index_select(dim,torch.arange(ix,x_s[dim],dilation,device=x.device)).unfold(dim, kernel_size, 1).contiguous()
        out = out.index_add_(dim,torch.arange(ix,o_s[dim],dilation,device=x.device),temp)

    return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unfold_dilation_fast(x, dim, kernel_size, dilation = 1):

    x_s = x.shape
    
    dilation_len = x_s[dim]-dilation*(kernel_size-1)
    out = x.unsqueeze(-1).transpose(-1,dim).squeeze(dim)
    
    out = F.pad(out,pad=(0,dilation-x_s[dim]%dilation)).unsqueeze(dim).transpose(-1,dim).squeeze(-1)
    out_s = out.shape
    out= out.view(x_s[:dim]+(out_s[dim]//dilation,dilation)+x_s[dim+1:]).transpose(dim,dim+1).reshape(out_s).unfold(dim,kernel_size,1)
    out_s = out.shape
    out = out.unsqueeze(-1).transpose(-1,dim).squeeze(dim)
    out_s2 = out.shape
    out = F.pad(out,pad=(0,dilation-out_s[dim]%dilation))
    out_s3 = out.shape
    out = out.reshape(out_s2[:-1]+(dilation,-1)).transpose(-2,-1).reshape(out_s3)[...,:dilation_len].unsqueeze(dim).transpose(-1,dim).squeeze(-1)
    
    return out

def time_to_string(tm):
    string = time.strftime('%Y-%m-%d %H%M%S', tm)
    return string

def logging(log, string):
    word = time_to_string(time.localtime(time.time())) + ' : ' + string
    print(word)
    log.write(word+'\n')
    return

