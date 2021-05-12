import torch
import torch.optim as optim
from torch.utils import data
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils import data
import SPD
import pdb
import fm_ops
import geoopt
from tool import *



def genetate_SPDMat(manifold, radius):

    sample = manifold.random(3,3)
    return sample
        

class SPDNet_single_layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SPDNet_single_layer, self).__init__()
        self.SPD_Linear1 = SPD.SPDLinear(in_channel,out_channel)
        
    def forward(self, x):
        x = self.SPD_Linear1(x)
        return x

class SPDNet_single_layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SPDNet_single_layer, self).__init__()
        self.SPD_Linear1 = SPD.SPDLinear(in_channel,out_channel)
        
    def forward(self, x):
        x = self.SPD_Linear1(x)
        return x


def dist_loss_mean(output, target):
    bs = output.shape[0]
    oc = output.shape[1]
    rl = output.shape[2]
    cl = output.shape[3]
    #pdb.set_trace()
    return (fm_ops.dist_(output.view(-1,rl,cl),target.view(-1,rl,cl)).view(bs,-1)**2).mean()

def dist_loss(output, target):
    bs = output.shape[0]
    oc = output.shape[1]
    rl = output.shape[2]
    cl = output.shape[3]
    #pdb.set_trace()
    return (fm_ops.dist_(output.view(-1,rl,cl),target.view(-1,rl,cl)).view(bs,-1))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    torch.cuda.device(device)
    
    #weights = torch.tensor([1,0,0,0]).reshape(-1,4).to(device) # Two channel
    

    ############################## data ##############################
    SPD_manifold = geoopt.SymmetricPositiveDefinite()
    r = np.pi
    obj_size = 5
    data_size = 2
    img_size = 1000


    max_iter = 1000


    data = []
    obj = [] 

    for ix in range(data_size):
        tmp = []
        for jx in range(img_size):
            tmp.append(genetate_SPDMat(SPD_manifold,r).view(1,3,3))
        data.append(torch.cat(tmp).unsqueeze(0))
    data = torch.cat(data,axis=0).to(device)

    for ix in range(data_size):
        tmp = []
        for ix in range(obj_size):
            tmp.append((genetate_SPDMat(SPD_manifold,r).view(1,3,3)))
        obj.append(torch.cat(tmp).unsqueeze(0))
    obj = torch.cat(obj,axis=0).to(device)

    single_layer = SPDNet_single_layer(img_size,obj_size).to(device)

    optimizer = geoopt.optim.RiemannianAdam(single_layer.parameters(), lr=1e-1)

    for iter in range(max_iter):

        optimizer.zero_grad()


        out1, out2, out3 = single_layer(data)
        pdb.set_trace()
        

        loss = dist_loss_mean(out,obj)


        loss.backward()


        optimizer.step()

        if iter % 1 == 0:
            f1 = open("experiment/experiment_tot/loss_%d_%d_%d.txt"%(img_size, data_size, max_iter), "a")
            f2 = open("experiment/experiment_tot/count_%d_%d_%d.txt"%(img_size, data_size, max_iter), "a")
            f3 = open("experiment/experiment_tot/dist_%d_%d_%d.txt"%(img_size, data_size, max_iter), "a")

            converge = torch.count_nonzero(dist_loss(out,obj)<0.05*np.pi)
            logging(f1,'%d %f'%(iter, loss.item()))
            logging(f2,'%d'%(converge))
            logging(f3,'%s'%(dist_loss(out,obj).cpu().detach().numpy()))
            
            f1.close()
            f2.close()
            f3.close()
            #time.sleep(1)