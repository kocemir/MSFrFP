import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import argparse
import dataloader
from dataloader import cifar10_data_loader,cifar100_data_loader,cub2011_data_loader, caltech101_data_loader
from our_models import resnet_pretrained,vgg_pretrained, densenet_pretrained, swinT_pretrained, mobilev2_pretrained, efficientnet_pretrained

from train_v2 import train
from test import test
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from utils import create_path

print("Imported all!")

#parser
parser = argparse.ArgumentParser(description='Training specifications')
parser.add_argument("--batch_size", type=int, default="128")
parser.add_argument("--dataset", type=str, default="cifar10",choices=["cifar10","cifar100","cub2011","caltech101"])
parser.add_argument("--datadir", type=str,required=True)
parser.add_argument("--epoch", type=int)
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--domain",type=str,choices=["time","frft","fft"],required=True)
parser.add_argument("--run_count",type=int,default=1)
parser.add_argument("--model",type=str,required=True, choices=["resnet18","resnet34","resnet50","resnet101","resnet152","vgg16","vgg13","vgg11","densenet121","swint","mobilenet","efficientnet"])
parser.add_argument('--gpu',type=int,choices=[0,1,2,3])
parser.add_argument('--pool_size',type=int, default=16,choices=[16,8,1])
parser.add_argument('--patience',type=int, default=10)
parser.add_argument('--frft_plus', type=bool,default=False)
# check gpu availability

# environment is train_frft
# python3 main_v2.py --batch_size 64 --dataset cifar100 --datadir cifar100_data --epoch 10 --lr 0.001 --domain frft --run_count 1 --model vgg16 --gpu 1 --pool_size 8 --patience 10 --frft_plus True
# python3 main_v2.py --batch_size 64 --dataset cifar100 --datadir cifar100_data --epoch 10 --lr 0.001 --domain frft --run_count 1 --model resnet50 --gpu 1 --pool_size 16 --patience 10 --frft_plus False

args = parser.parse_args()

available_gpu= args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(available_gpu) 

if torch.cuda.is_available():
     print("Yeay! We have GPU.")
     torch.cuda.set_device(available_gpu) 
else:
       print("No, we do not have a GPU.")

_BATCH_SIZE=args.batch_size
_DATA_DIR=  args.datadir #'./cifar10_data' # './cub_data'

if args.dataset=="cifar10":
     _CLASSES=10
     train_loader, valid_loader,test_loader = cifar10_data_loader(data_dir=_DATA_DIR,
                                         batch_size=_BATCH_SIZE)
   
if args.dataset=="cifar100":
          _CLASSES=100
          train_loader, valid_loader,test_loader = cifar100_data_loader(data_dir=_DATA_DIR,
                                         batch_size=_BATCH_SIZE)
          
if args.dataset=="cub2011":
     _CLASSES=200
     train_loader, valid_loader,test_loader= cub2011_data_loader(data_dir=_DATA_DIR,
                                                                 batch_size=_BATCH_SIZE)

elif args.dataset=="caltech101":
     _CLASSES=101
     train_loader, valid_loader,test_loader= caltech101_data_loader(data_dir=_DATA_DIR,
                                                                 batch_size=_BATCH_SIZE)

data = args.dataset

_LR=args.lr
_EPOCHS=args.epoch
_DOMAIN=args.domain
_RUN=args.run_count
patience=args.patience
pool_size=args.pool_size

if _DOMAIN=="time":
     pool_size=1



for r in range(args.run_count):

         
          #create the model here
          if args.model=="resnet18":
               model= resnet_pretrained.ResNet18(_CLASSES,domain=_DOMAIN,N=pool_size).to(device)
          elif args.model=="resnet34":
               model= resnet_pretrained.ResNet34(_CLASSES,domain=_DOMAIN,N=pool_size).to(device)
          elif  args.model=="resnet50":
               model= resnet_pretrained.ResNet50(_CLASSES,domain=_DOMAIN,N=pool_size).to(device)
          elif args.model=="resnet101":
               model= resnet_pretrained.ResNet101(_CLASSES,domain=_DOMAIN,N=pool_size).to(device)
          elif args.model=="resnet152":
               model= resnet_pretrained.ResNet152(_CLASSES,domain=_DOMAIN,N=pool_size).to(device)
          

          elif args.model=='vgg16':
                  model= vgg_pretrained.VGG16(domain=_DOMAIN,classes=_CLASSES,N=pool_size, FrFT_Plus=args.frft_plus).to(device)
          elif args.model=='vgg13':
                  model= vgg_pretrained.VGG13(domain=_DOMAIN,classes=_CLASSES,N=pool_size, FrFT_Plus=args.frft_plus).to(device)
          elif args.model=='vgg11':
                  model= vgg_pretrained.VGG11(domain=_DOMAIN,classes=_CLASSES,N=pool_size, FrFT_Plus=args.frft_plus).to(device)
                    

          elif args.model=="densenet121":
                  model= densenet_pretrained.DenseNet121(domain=_DOMAIN,classes=_CLASSES,N=pool_size).to(device)
          
          elif args.model=="swint":
                model= swinT_pretrained.SwinTiny_V2(domain=_DOMAIN,classes=_CLASSES,N=pool_size).to(device)
                
          elif args.model=="mobilenet":
               model=  mobilev2_pretrained.MobileNet_V2(domain=_DOMAIN,classes=_CLASSES,N=pool_size,FrFT_Plus=args.frft_plus).to(device)
     
          elif args.model=="efficientnet":
               model= efficientnet_pretrained.EfficientNet(domain=_DOMAIN,classes=_CLASSES,N=pool_size,FrFT_Plus=args.frft_plus).to(device)
          
                
                 
          else:
                 raise Exception("No proper model and dataset combination is chosen!")
          
          
          criterion = nn.CrossEntropyLoss()
          optimizer = optim.SGD(model.parameters(), lr=_LR, momentum=0.9, weight_decay=5e-4)
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_EPOCHS) 
                
                    
          print("******************************** RUN {} ***************************".format(r+1))


          start_time= time.time()
          _TRAIN_ACC,_VALID_ACC,_VALID_TOP5_ACC,train_loss_iter,valid_loss_iter,frac_a,frac_b,best_model,best_epoch,best_valid_acc,best_valid_top5_acc,best_optimizer,best_scheduler,frac2_a,frac2_b=train(data,_EPOCHS,model,train_loader,valid_loader,optimizer,device,criterion,scheduler,patience)
          end_time = time.time()
          elapsed_time = end_time - start_time
          print("Elapsed time: {:.2f} seconds".format(elapsed_time))
          
          _TEST_ACC,_TEST_TOP5_ACC, _=test(data, model,test_loader,device,criterion)
          _BEST_TEST_ACC,_BEST_TEST_TOP5_ACC,mean_inference_time=test(data, best_model,test_loader,device,criterion)
          

          print("Test Accuracy", _TEST_ACC)
          print("Test Top5 Accuracy",_TEST_TOP5_ACC)
          print("Best Test Accuracy", _BEST_TEST_ACC)
          print("Best Test  Top5 Accuracy", _BEST_TEST_TOP5_ACC)




                    # Save model here
          state = {
               'net': model.state_dict(),
               'best_net':best_model.state_dict(),
               'opt_dict':optimizer.state_dict(),
               'test_acc': _TEST_ACC,
               'test_top5_acc':_TEST_TOP5_ACC,
               'train_acc':_TRAIN_ACC,
               'valid_acc':_VALID_ACC,
               'valid_top5_acc':_VALID_TOP5_ACC,
               'best_valid_acc':best_valid_acc,
               'best_valid_top5_acc':best_valid_top5_acc,
               'best_test_acc':_BEST_TEST_ACC,
               'best_test_top5_acc':_BEST_TEST_TOP5_ACC,
               'epoch': _EPOCHS,
               'best_epoch':best_epoch,
               'scheduler':scheduler.state_dict(),
               'mean_inference_time': mean_inference_time,
               'training_time':elapsed_time,
               'frac_a':frac_a,
               'frac_b':frac_b,
               'frac2_a':frac2_a,
               'frac2_b':frac2_b,
               'train_loss':train_loss_iter,
               'valid_loss':valid_loss_iter,
               'best_optimizer':best_optimizer.state_dict(),
               'best_scheduler':best_scheduler.state_dict()
   

          }
          
          if not os.path.isdir('checkpoints'):
             os.mkdir('checkpoints')
          base_folder="{}/{}/{}".format(args.model,args.dataset,args.domain)
          base_folder= os.path.join("checkpoints", base_folder)
          
          if not os.path.isdir(base_folder):
              os.makedirs(base_folder)
          path_name= create_path(base_folder,_BATCH_SIZE,best_epoch,_DOMAIN,r+1,_LR,pool_size,patience,frft_plus=model.FrFT_Plus)
          
          
          torch.save(state, path_name)


     


     

          
          


          





      



