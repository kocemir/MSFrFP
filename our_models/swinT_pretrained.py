import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os
import sys
import numpy as np
from torch.nn import init


from our_models.torch_frft.torch_frft.layer import DFrFTLayer2D
from our_models.custom_frft_layers import FrFT_Pool,FrFT_MaxAttent

class SwinTiny_V2(nn.Module):
    
    def __init__(self, pre_trained=True, classes=200, domain="time",N=16):
        super(SwinTiny_V2, self).__init__()
        self.n_class = classes
        self.domain=domain
        
        self.N=N
        self.frac=DFrFTLayer2D() 
        self.crop= FrFT_MaxAttent(self.frac,domain=self.domain,N=self.N)


       
        self.base_model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        self.features= self.base_model.features
        self.norm=self.base_model.norm
        self.permute=self.base_model.permute
        self.flatten=self.base_model.flatten

         
        
        if self.domain=="time":
            self.avgpool=self.base_model.avgpool
            self.fc = nn.Linear(self.base_model.head.in_features, self.n_class)
        else:
            self.avgpool = self.crop
            self.fc = nn.Linear(self.base_model.head.in_features*self.N, self.n_class)
       
         #this part is crucial!!!!!!!!!!!!
        self.fc.apply(weight_init_kaiming)
        


    def forward(self, x):
        N=x.size(0)
       
        x=self.features(x)   
        x=self.norm(x)
        x= self.permute(x)
        x=self.avgpool(x)   
        x=self.flatten(x)
        x=self.fc(x)
        assert x.size() == (N, self.n_class)
        return x
        
def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Linear') == -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.bias.data,a=0, mode='fan_in')

        

if __name__=="__main__":

   swinT=  models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT) #SwinTiny_V2()
   print(swinT)
   rndn= torch.rand(100,3,224,224)
   res= swinT.features(rndn)
   print(res.shape)
   