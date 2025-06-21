import torch
import torch.nn as nn
from torchvision import models
import os
import sys
import numpy as np
from torch.nn import init


from our_models.torch_frft.torch_frft.layer import DFrFTLayer2D
from our_models.custom_frft_layers import FrFT_Pool,FrFT_MaxAttent

class ResNet(nn.Module):
    
    def __init__(self, pre_trained=True, n_class=200, model_choice=50,domain="time",N=16,FrFT_Plus=False):
        super(ResNet, self).__init__()
        expansion=4
        if model_choice ==18 or model_choice==34:
            expansion=1
        
        self.n_class = n_class
        self.domain=domain
        self.FrFT_Plus=FrFT_Plus
        
        self.N=N
        self.frac=DFrFTLayer2D() 
        self.crop= FrFT_MaxAttent(self.frac,domain=self.domain,N=self.N)
       
        self.base_model = self._model_choice(pre_trained, model_choice)
        if self.domain=="time":
            self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.base_model.fc = nn.Linear(512*expansion, n_class)
        else:
            self.base_model.avgpool = self.crop
            self.base_model.fc = nn.Linear(512*expansion*self.N, n_class)
       
         #this part is crucial!!!!!!!!!!!!
        self.base_model.fc.apply(weight_init_kaiming)
        
       
        self.features1 = torch.nn.Sequential(
                self.base_model.conv1,
                self.base_model.bn1,
                self.base_model.relu,
                self.base_model.maxpool,
                self.base_model.layer1,
                self.base_model.layer2,  self.base_model.layer3,self.base_model.layer4)
    


    def forward(self, x):
        N=x.size(0)
       
        x=self.features1(x)

        x=self.base_model.avgpool(x)   
        x=x.contiguous().view(x.size(0), -1)
        x=self.base_model.fc(x)
        assert x.size() == (N, self.n_class)
        return x

    def _model_choice(self, pre_trained, model_choice):
        
        if model_choice == 50:
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_choice == 101:
            return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif model_choice == 152:
            return models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        elif model_choice==18:
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_choice==34:
              return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

def ResNet18(classes,domain="time",N=16):
    return ResNet(n_class=classes, model_choice=18,domain=domain,N=N)
def ResNet34(classes,domain="time",N=16):
    return ResNet(n_class=classes,model_choice=34,domain=domain,N=N)
def ResNet50(classes,domain="time",N=16):
    return ResNet(n_class=classes,model_choice=50,domain=domain,N=N)
def ResNet101(classes,domain="time",N=16):
    return ResNet(n_class=classes,model_choice=101,domain=domain,N=N)
def ResNet152(classes,domain="time",N=16):
    return ResNet(n_class=classes,model_choice=152,domain=domain,N=N)


def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
  

#check from which maxpooling layers you want
if __name__== "__main__":
   resnet50=  models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
   print(resnet50)
