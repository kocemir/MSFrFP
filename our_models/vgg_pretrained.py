import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import init


from our_models.torch_frft.torch_frft.layer import DFrFTLayer, FrFTLayer,DFrFTLayer2D,FrFTLayer2D
from our_models.custom_frft_layers import FrFT_Pool,FrFT_MaxAttent


class VGG16(nn.Module):

    def __init__(self,domain,classes,N=16,FrFT_Plus=False):
        
        super(VGG16, self).__init__()
        self.model=models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.classes=classes
        self.domain=domain
        self.N=N
        self.frac=DFrFTLayer2D() #DFrFT2DM()
        self.crop= FrFT_MaxAttent(self.frac,domain=self.domain,N=self.N)
        self.FrFT_Plus=FrFT_Plus
        
        if self.domain!="time":
           self.model.avgpool= self.crop
           flat_dim=self.N
                                 # If we connect the upper layer
           if self.FrFT_Plus==True:
                self.frac2= DFrFTLayer2D()
                self.crop2 = FrFT_MaxAttent(self.frac2,domain=self.domain,N=self.N) 
                flat_dim=2*self.N

        
        elif self.domain=="time":
            self.model.avgpool=  nn.AdaptiveAvgPool2d((1,1))
            flat_dim=1*1
        
        self.model.classifier[0]=nn.Linear(in_features=512*flat_dim, out_features=4096, bias=True)
        self.model.classifier[6]= nn.Linear(in_features=4096, out_features=self.classes, bias=True) #4096 here is from original pretrained model with imagenet
        self.model.classifier[6].apply(weight_init_kaiming)
        self.model.classifier[0].apply(weight_init_kaiming)
        
       

    def forward(self,x):
          if self.FrFT_Plus==False:
            x= self.model.features(x)
            x= self.model.avgpool(x)

          elif self.FrFT_Plus==True:
              intermediate_model= self.model.features[:30]
              last_maxpool=self.model.features[30]
              
              out1=intermediate_model(x)
              out2=self.crop2(out1)
              out3=last_maxpool(out1)
              out4=self.model.avgpool(out3)

              x=torch.cat((out2,out4),axis=-1)
    
              
          x=x.contiguous().view(x.size(0), -1)
          x=self.model.classifier(x)
          
          
          return x



class VGG13(nn.Module):

    def __init__(self,domain,classes,N=16,FrFT_Plus=False):
        
        super(VGG13, self).__init__()
        self.model=models.vgg13(weights=models.VGG13_Weights.DEFAULT)
        self.classes=classes
        self.domain=domain
        self.N=N
        self.FrFT_Plus=FrFT_Plus
        
        self.frac=DFrFTLayer2D() #DFrFT2DM()
        self.crop=  FrFT_MaxAttent(self.frac,domain=self.domain,N=self.N)      
        
        if self.domain!="time":
           self.model.avgpool= self.crop
           flat_dim=self.N
                      # If we connect the upper layer
           if self.FrFT_Plus==True:
                self.frac2= DFrFTLayer2D()
                self.crop2 = FrFT_MaxAttent(self.frac2,domain=self.domain,N=self.N) 
                flat_dim=2*self.N

        
        elif self.domain=="time":
            self.model.avgpool=  nn.AdaptiveAvgPool2d((1,1))
            flat_dim=1*1
        
        self.model.classifier[0]=nn.Linear(in_features=512*flat_dim, out_features=4096, bias=True)
        self.model.classifier[6]= nn.Linear(in_features=4096, out_features=self.classes, bias=True) #4096 here is from original pretrained model with imagenet
        self.model.classifier[6].apply(weight_init_kaiming)
        self.model.classifier[0].apply(weight_init_kaiming)
      
   
    def forward(self,x):
          if self.FrFT_Plus==False:
            x= self.model.features(x)
            x= self.model.avgpool(x)

          elif self.FrFT_Plus==True:
              intermediate_model= self.model.features[:24]
              last_maxpool=self.model.features[24]
              
              out1=intermediate_model(x)
              out2=self.crop2(out1)
              out3=last_maxpool(out1)
              out4=self.model.avgpool(out3)

              x=torch.cat((out2,out4),axis=-1)
    
              
          x=x.contiguous().view(x.size(0), -1)
          x=self.model.classifier(x)
          
          
          return x
    

class VGG11(nn.Module):

    def __init__(self,domain,classes, N=16, FrFT_Plus=False):
        
        super(VGG11, self).__init__()
        self.model=models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        self.classes=classes
        self.domain=domain
        self.N=N
        self.FrFT_Plus=FrFT_Plus
    
        self.frac=DFrFTLayer2D() #DFrFT2DM()
        self.crop= FrFT_MaxAttent(self.frac,domain=self.domain,N=self.N)

         
        
        if self.domain!="time":
           self.model.avgpool= self.crop
           flat_dim=self.N
           
           # If we connect the upper layer
           if self.FrFT_Plus==True:
                self.frac2= DFrFTLayer2D()
                self.crop2 = FrFT_MaxAttent(self.frac2,domain=self.domain,N=self.N) 
                flat_dim=2*self.N

        elif self.domain=="time":
            self.model.avgpool=  nn.AdaptiveAvgPool2d((1,1))
            flat_dim=1*1


        self.model.classifier[0]=nn.Linear(in_features=512*flat_dim, out_features=4096, bias=True)
        self.model.classifier[6]= nn.Linear(in_features=4096, out_features=self.classes, bias=True) #4096 here is from original pretrained model with imagenet
        self.model.classifier[6].apply(weight_init_kaiming)
        self.model.classifier[0].apply(weight_init_kaiming)


        
 
    def forward(self,x):
          if self.FrFT_Plus==False:
            x= self.model.features(x)
            x= self.model.avgpool(x)

          elif self.FrFT_Plus==True:
              intermediate_model= self.model.features[:20]
              last_maxpool=self.model.features[20]
              
              out1=intermediate_model(x)
              out2=self.crop2(out1)
              out3=last_maxpool(out1)
              out4=self.model.avgpool(out3)

              x=torch.cat((out2,out4),axis=-1)
    
              
          x=x.contiguous().view(x.size(0), -1)
          x=self.model.classifier(x)
          
          
          return x
        

def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)


#check from which maxpooling layers you want
if __name__== "__main__":

        # Step 1: Load pretrained VGG-16 model
    vgg16 = models.vgg13(pretrained=True)


        # Step 2: Print the model to inspect its architectur
    print(vgg16.features)



# python3 main_v2.py --batch_size 64 --dataset cifar100 --datadir cifar100_data --epoch 10 --lr 0.001 --domain frft --run_count 1 --model vgg16 --gpu 1 --pool_size 16 --patience 10
