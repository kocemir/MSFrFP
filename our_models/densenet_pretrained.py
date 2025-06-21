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

class DenseNet121(nn.Module):
    
    def __init__(self, pre_trained=True, classes=200, domain="time",N=16):
        super(DenseNet121, self).__init__()
        self.n_class = classes
        self.domain=domain
        
        self.N=N
        self.frac=DFrFTLayer2D() 
        self.crop= FrFT_MaxAttent(self.frac,domain=self.domain,N=self.N)

       
        self.base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.features= self.base_model.features
        self.relu=nn.ReLU(inplace=True)
         
        
        if self.domain=="time":
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(self.base_model.classifier.in_features, self.n_class)
        else:

            self.avgpool = self.crop
            self.fc = nn.Linear(self.base_model.classifier.in_features*self.N, self.n_class)
       
         #this part is crucial!!!!!!!!!!!!
        self.fc.apply(weight_init_kaiming)
        


    def forward(self, x):
        N=x.size(0)
       
        x=self.features(x)
        x=self.relu(x)
        x=self.avgpool(x)   
        x=x.contiguous().view(x.size(0), -1)
        x=self.fc(x)
        assert x.size() == (N, self.n_class)
        return x
        
def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
        #init.constant_(m.bias.data, 0.0)


if __name__=="__main__":

   densenet= models.densenet121(weights=models.DenseNet121_Weights.DEFAULT) #DenseNet121()
   
   rndn= torch.rand(100,3,224,224)
   print(densenet.features.transition3)
   res= densenet(rndn)
   print(res.shape)
   