import torch
import torch.nn as nn
from tqdm import tqdm
from utils import  min_value,max_value
from test import test
import copy


def train(data,epochs,model,train_loader,valid_loader,optimizer,device,criterion,scheduler,patience):

    BEST_VALID_ACC=0
    print('frac a and b is initialized as {} and {}'.format(
    model.frac.frac1.order.item(),
    model.frac.frac2.order.item()))
    frac_a=[]
    frac_a.append(round(model.frac.frac1.order.item(),3))
    frac_b =[]
    frac_b.append(round(model.frac.frac2.order.item(),3))
    
    ###################
     #Fraction orders for FrFT+
    frac2_a=[]
    frac2_b =[]
    
    if model.FrFT_Plus==True:
            
            frac2_a.append(round(model.frac2.frac1.order.item(),3))     
            frac2_b.append(round(model.frac2.frac2.order.item(),3))
    ########### 




    

    flag=True
    _TRAIN_ACC=0
    _VALID_ACC=0
    temp_VALID_ACC=0


    train_loss_iter = [] # loss for each epoch
    valid_loss_iter=[]

    

    total_step = len(train_loader)
    valid_total_step=len(valid_loader)
    total_sample=0
    correct=0
    correct_top5=0

    earlystop_counter=0

    for epoch in range(epochs):
        running_loss=0
     
       # for i, (images, labels) in tqdm(enumerate(train_loader)):
        # Move tensors to the configured device
        model.train()
        for images, labels in tqdm(train_loader):  
          
          '''
          with torch.no_grad():
               model.frac.frac1.order.data = torch.clamp(model.frac.frac1.order.data, min=min_value, max=max_value)
               model.frac.frac2.order.data = torch.clamp(model.frac.frac2.order.data, min=min_value, max=max_value)
          '''
            

          images = images.to(device)
          labels = labels.to(device)
          optimizer.zero_grad()
          
          # Forward pass
          outputs = model(images)

          

          if data == "caltech101":
               
            labels = torch.max(labels, 1)[1]

          loss = criterion(outputs, labels)

            
          # Backward and optimize
          loss.backward()
          optimizer.step()
           

          _, predicted = outputs.max(1)
          total_sample += labels.size(0)

          
          correct += predicted.eq(labels).sum().item()
          running_loss+=loss.item()*labels.size(0)


          top5_probabilities, top5_classes = torch.topk(outputs, k=5, dim=1)
        
          correct_top5 += torch.eq(top5_classes, labels.view(-1,1)).sum().item()

        
        top5_acc= 100*correct_top5/total_sample
        train_loss_iter.append(running_loss/total_sample)
        print ('Epoch [{}/{}], Average Training Loss: {:.4f}, Training Accuracy: {:.4f}, Training Top-5 Accuracy: {:.4f} frac a: {}, frac b: {}'.format(epoch+1, epochs, running_loss/total_sample,100.*correct/total_sample,top5_acc,model.frac.frac1.order.item()%4,model.frac.frac2.order.item()%4))
        frac_a.append(round(model.frac.frac1.order.item()%4,3))
        frac_b.append(round(model.frac.frac2.order.item()%4,3))

        if model.FrFT_Plus==True:

                    print('Upper layer frac a: {}'.format(model.frac2.frac1.order.item()%4))
                    print('Upper layer frac b: {}'.format(model.frac2.frac2.order.item()%4))
                    frac2_a.append(round(model.frac2.frac1.order.item()%4,3))
                    frac2_b.append(round(model.frac2.frac2.order.item()%4,3))
       
        
        scheduler.step()
        model.eval()
    
           # Validation
        with torch.no_grad():
            valid_correct = 0
            valid_total_sample = 0
           
            valid_running_loss=0
            valid_correct_top5=0
            for images, labels in tqdm(valid_loader):

                if data == "caltech101":            
                  labels = torch.max(labels, 1)[1]

                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                valid_loss = criterion(outputs, labels)
               
                _, predicted = outputs.max(1)
                valid_total_sample += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()
                valid_running_loss+=valid_loss.item()*labels.size(0)

                top5_probabilities, top5_classes = torch.topk(outputs, k=5, dim=1)
        
                valid_correct_top5 += torch.eq(top5_classes, labels.view(-1,1)).sum().item()


 
        valid_top5_acc=100*valid_correct_top5/valid_total_sample   
        valid_loss_iter.append(valid_running_loss/valid_total_sample)
        print ('Epoch [{}/{}], Average Valid Loss: {:.4f}, Valid Accuracy: {:.4f}, Valid Top-5 Accuracy: {:.4f}'.format(epoch+1,epochs, valid_running_loss/valid_total_sample,100*valid_correct/valid_total_sample,valid_top5_acc))
        

        _VALID_ACC=100*valid_correct/valid_total_sample

        if BEST_VALID_ACC>_VALID_ACC:
              earlystop_counter+=1
        else:
            earlystop_counter=0

        if earlystop_counter==patience:
            print("Early stopping occured!")
            break

        
        if _VALID_ACC > BEST_VALID_ACC: 
            best_model=copy.deepcopy(model)
            best_epoch= epoch+1
            best_optimizer=copy.deepcopy(optimizer)
            best_scheduler=copy.deepcopy(scheduler)
            BEST_VALID_TOP5_ACC=valid_top5_acc
            BEST_VALID_ACC=_VALID_ACC

        
        

        
    _TRAIN_ACC=100*correct/total_sample
    _VALID_ACC=100*valid_correct/valid_total_sample
    _VALID_TOP5_ACC= valid_top5_acc
    
    print("Training is done!")

    return _TRAIN_ACC,_VALID_ACC,_VALID_TOP5_ACC, train_loss_iter,valid_loss_iter,frac_a,frac_b,best_model,best_epoch,BEST_VALID_ACC,BEST_VALID_TOP5_ACC,best_optimizer,best_scheduler,frac2_a,frac2_b







