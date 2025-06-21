import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import argparse
import dataloader
from dataloader import cifar_data_loader,cub2011_data_loader
from our_models import resnet_pretrained, resnet_small,vgg,vgg_pretrained

from train import train
from test import test
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from utils import create_path

print("Hey")

#parser
parser = argparse.ArgumentParser(description='Training specifications')
parser.add_argument("--batch_size", type=int, default="128")
parser.add_argument("--dataset", type=str, default="cifar10",choices=["cifar10","cub2011"])
parser.add_argument("--datadir", type=str,required=True)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--classes",type=int,required=True)
parser.add_argument("--domain",type=str,choices=["time","frft","fft"],required=True)
parser.add_argument("--run_count",type=int,default=1)
parser.add_argument("--model",type=str,required=True, choices=["resnet18","resnet34","resnet50","resnet101","resnet152","vgg16","vgg13","vgg11"])
parser.add_argument('--gpu',type=int,choices=[0,1,2,3])
# check gpu availability


args = parser.parse_args()

available_gpu= args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(available_gpu) 

if torch.cuda.is_available():
     print("yeah")
     torch.cuda.set_device(available_gpu) 
else:
       print("fuck")

_BATCH_SIZE=args.batch_size
_DATA_DIR=  args.datadir #'./cifar10-data' # './cub_data'

if args.dataset=="cifar10":
     train_loader, valid_loader,test_loader = cifar_data_loader(data_dir=_DATA_DIR,
                                         batch_size=_BATCH_SIZE)
elif args.dataset=="cub2011":
     train_loader, valid_loader,test_loader= cub2011_data_loader(data_dir=_DATA_DIR,
                                                                 batch_size=_BATCH_SIZE)



_LR=args.lr
_CLASSES=args.classes
_EPOCHS=args.epoch
_DOMAIN=args.domain
_RUN=args.run_count


for r in range(args.run_count):

         
          #create the model here
          if args.dataset=="cifar10" and args.model=="resnet18":
               model= resnet_small.ResNet18(_CLASSES,domain=_DOMAIN).to(device)
          elif args.dataset=="cifar10" and args.model=="resnet34":
               model= resnet_small.ResNet34(_CLASSES,domain=_DOMAIN).to(device)
          elif  args.dataset=="cifar10" and args.model=="resnet50":
               model= resnet_small.ResNet50(_CLASSES,domain=_DOMAIN).to(device)


          elif  args.dataset=="cifar10" and args.model=="vgg16":
               model= vgg.VGG("VGG16",domain=_DOMAIN,classes=_CLASSES).to(device)
          elif  args.dataset=="cifar10" and args.model=="vgg11":
               model= vgg.VGG("VGG11",domain=_DOMAIN,classes=_CLASSES).to(device)      
          elif  args.dataset=="cifar10" and args.model=="vgg13":        
                model= vgg.VGG("VGG13",domain=_DOMAIN,classes=_CLASSES).to(device)      


          elif args.dataset=="cub2011" and args.model=="resnet18":
                 model= resnet_pretrained.ResNet18(_CLASSES,domain=_DOMAIN).to(device)
          elif args.dataset=="cub2011" and args.model=="resnet34":
                 model= resnet_pretrained.ResNet34(_CLASSES,domain=_DOMAIN).to(device)          
          elif args.dataset=="cub2011" and args.model=="resnet50":
                 model= resnet_pretrained.ResNet50(_CLASSES,domain=_DOMAIN).to(device)       
          elif args.dataset=="cub2011" and args.model=="resnet101":
                 model= resnet_pretrained.ResNet101(_CLASSES,domain=_DOMAIN).to(device)
          elif args.dataset=="cub2011" and args.model=="resnet152":
                 model= resnet_pretrained.ResNet152(_CLASSES,domain=_DOMAIN).to(device)
          
          

          elif args.dataset=="cub2011" and args.model=='vgg16':
                  model= vgg_pretrained.VGG16(domain=_DOMAIN,classes=_CLASSES).to(device)
          elif args.dataset=="cub2011" and args.model=='vgg13':
                  model= vgg_pretrained.VGG13(domain=_DOMAIN,classes=_CLASSES).to(device)          
          elif args.dataset=="cub2011" and args.model=='vgg11':
                  model= vgg_pretrained.VGG11(domain=_DOMAIN,classes=_CLASSES).to(device)

                 
          else:
                 raise Exception("No proper model and dataset combination is chosen!")

          if args.dataset=="cifar10":
      
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(model.parameters(), lr=_LR, momentum=0.9, weight_decay=5e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_EPOCHS) 
          elif args.dataset=="cub2011":
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.SGD(model.parameters(), lr=_LR, momentum=0.9, weight_decay=1e-4)
                        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_EPOCHS) 
                
                    
          print("******************************** RUN {} ***************************".format(r+1))


          start_time= time.time()
          _TRAIN_ACC,_VALID_ACC,train_loss_iter,valid_loss_iter,frac_a,frac_b=train(_EPOCHS,model,train_loader,valid_loader,optimizer,device,criterion,scheduler)
          end_time = time.time()
          elapsed_time = end_time - start_time
          print("Elapsed time: {:.2f} seconds".format(elapsed_time))

          _TEST_ACC, mean_inference_time=test(model,test_loader,device,criterion)

          print(_TRAIN_ACC)
          print(_VALID_ACC)
          

          print(_TEST_ACC)


                    # Save model here
          state = {
               'net': model.state_dict(),
               'opt_dict':optimizer.state_dict(),
               'test_acc': _TEST_ACC,
               'train_acc':_TRAIN_ACC,
               'valid_acc':_VALID_ACC,
               'epoch': _EPOCHS,
               'mean_inference_time': mean_inference_time,
               'scheduler':scheduler.state_dict(),
               'training_time':elapsed_time,
               'frac_a':frac_a,
               'frac_b':frac_b,
               'train_loss':train_loss_iter,
               'valid_loss':valid_loss_iter

          }
          

          
          if not os.path.isdir('checkpoints'):
             os.mkdir('checkpoints')
          base_folder="{}/{}/{}".format(args.model,args.dataset,args.domain)
          base_folder= os.path.join("checkpoints", base_folder)
          
          if not os.path.isdir(base_folder):
              os.makedirs(base_folder)
          path_name= create_path(base_folder,_BATCH_SIZE,_EPOCHS,_DOMAIN,r+1,_LR)
          

          torch.save(state, path_name)

          


     

          
          


          





      



