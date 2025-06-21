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

class MobileNet_V2(nn.Module):
    
    def __init__(self, pre_trained=True, classes=200, domain="time",N=16,FrFT_Plus=False):
        super(MobileNet_V2, self).__init__()
        self.n_class = classes
        self.domain=domain
        self.FrFT_Plus=FrFT_Plus
        
        self.N=N
        self.frac=DFrFTLayer2D() 
        self.crop= FrFT_MaxAttent(self.frac,domain=self.domain,N=self.N)
       
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features= self.base_model.features
        
         
        
        if self.domain=="time":
            self.avgpool=nn.AdaptiveAvgPool2d((1,1))
            self.base_model.classifier[1]= nn.Linear(self.base_model.last_channel,self.n_class)
        else:
            self.avgpool = self.crop
            self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel*self.N, self.n_class)
            print(self.n_class)
            print(self.base_model.classifier[1])
       
         #this part is crucial!!!!!!!!!!!!
        self.base_model.classifier[1].apply(weight_init_kaiming)
        
        # Then:
        self.classifier=self.base_model.classifier
       
        


    def forward(self, x):
        N=x.size(0)
       
        x=self.features(x)   
        x=self.avgpool(x)   
        x=x.contiguous().view(x.size(0), -1)
        x=self.classifier(x)
        assert x.size() == (N, self.n_class)
        return x
        
def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Linear') == -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.bias.data,a=0, mode='fan_in')



if __name__=="__main__":

   mobile=  models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
   print(mobile)
   rndn= torch.rand(100,3,224,224)
   res= mobile.features(rndn)
   print(res.shape)
   