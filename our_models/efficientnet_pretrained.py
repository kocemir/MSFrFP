
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

class EfficientNet(nn.Module):
    
    def __init__(self, pre_trained=True, classes=200, domain="time",N=16,FrFT_Plus=False):
        super(EfficientNet, self).__init__()
        self.n_class = classes
        self.domain=domain
        self.FrFT_Plus=FrFT_Plus
        
        self.N=N
        self.frac=DFrFTLayer2D() 
        self.crop= FrFT_MaxAttent(self.frac,domain=self.domain,N=self.N)
       
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features= self.base_model.features
        
         
        
        if self.domain=="time":
            self._avg_pooling=nn.AdaptiveAvgPool2d((1,1))
            self.base_model.classifier[1]= nn.Linear(1280,self.n_class)
        else:
            self._avg_pooling = self.crop
            print(self.n_class)
            self.base_model.classifier[1] = nn.Linear(1280*self.N, self.n_class)
            
       
         #this part is crucial!!!!!!!!!!!!
        self.base_model.classifier[1].apply(weight_init_kaiming)
        
        # Then:
        self.classifier=self.base_model.classifier
       
        


    def forward(self, x):
        N=x.size(0)
       
        x=self.features(x)   
        x=self._avg_pooling(x)   
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

   efficient=  models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
   print(efficient)
   rndn= torch.rand(100,3,224,224)
   res= efficient.features(rndn)
   print(res.shape)
   
