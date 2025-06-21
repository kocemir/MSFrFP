import torch
import torch.nn as nn
from tqdm import tqdm
from utils import  min_value,max_value



def train(epochs,model,train_loader,valid_loader,optimizer,device,criterion,scheduler):

    
    print('frac a and b is initialized as {} and {}'.format(
    model.frac.frac1.order.item(),
    model.frac.frac2.order.item()))
    frac_a=[]
    frac_a.append(round(model.frac.frac1.order.item(),3))
    frac_b =[]
    frac_b.append(round(model.frac.frac2.order.item(),3))

    
    _TRAIN_ACC=0
    _VALID_ACC=0


    train_loss_iter = [] # loss for each epoch
    valid_loss_iter=[]

    

    total_step = len(train_loader)
    valid_total_step=len(valid_loader)
    total_sample=0
    correct=0
    for epoch in range(epochs):
        running_loss=0
     
       # for i, (images, labels) in tqdm(enumerate(train_loader)):
        # Move tensors to the configured device
        
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
          loss = criterion(outputs, labels)

          # Backward and optimize
        
          loss.backward()
          optimizer.step()

       
          _, predicted = outputs.max(1)
          total_sample += labels.size(0)
          correct += predicted.eq(labels).sum().item()
          running_loss+=loss.item()*labels.size(0)

       
        train_loss_iter.append(running_loss/total_step)
        print ('Epoch [{}/{}], Average Training Loss: {:.4f}, Training Accuracy: {:.4f}, frac a: {}, frac b: {}'.format(epoch+1, epochs, running_loss/total_sample,100.*correct/total_sample,model.frac.frac1.order.item()%4,model.frac.frac2.order.item()%4))
        frac_a.append(round(model.frac.frac1.order.item()%4,3))
        frac_b.append(round(model.frac.frac2.order.item()%4,3))

        model.eval()
           # Validation
        with torch.no_grad():
            valid_correct = 0
            valid_total_sample = 0
           
            valid_running_loss=0
            for images, labels in tqdm(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                valid_loss = criterion(outputs, labels)
               
                _, predicted = outputs.max(1)
                valid_total_sample += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()
                valid_running_loss+=valid_loss.item()*labels.size(0)
        model.train()
        
        valid_loss_iter.append(valid_running_loss/valid_total_step)
        print ('Epoch [{}/{}], Average Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch+1,epochs, valid_running_loss/valid_total_sample,100*valid_correct/valid_total_sample))
     
        scheduler.step()
        
    _TRAIN_ACC=100*correct/total_sample
    _VALID_ACC=100*valid_correct/valid_total_sample

    
    print("Training is done!")

    return _TRAIN_ACC,_VALID_ACC,train_loss_iter,valid_loss_iter,frac_a,frac_b





'''
import time
start_time = time.time()
train(_EPOCHS)
# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))
'''







