import torch
import torch.nn as nn
import time 
from tqdm import tqdm

def test(data, model,test_loader,device,criterion):
     
    _TEST_ACC=0
    mean_inference_time=0
    
    model.eval() # eval mode
    test_loss = 0
    correct = 0
    total = 0
    test_correct_top5=0
    
    flag=True
    start_time=time.time()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            end_time=time.time()

            if data == "caltech101":
              targets = torch.max(targets, 1)[1]

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            top5_probabilities, top5_classes = torch.topk(outputs, k=5, dim=1)
        
            test_correct_top5 += torch.eq(top5_classes, targets.view(-1,1)).sum().item()
 

 
    mean_inference_time=(end_time-start_time)/total
    test_acc = 100.*correct/total
    test_top5_acc= 100.*test_correct_top5/total
    _TEST_ACC=test_acc
    _TEST_TOP5_ACC=test_top5_acc
    print("Mean inference time is {} seconds".format(mean_inference_time))

    return _TEST_ACC, _TEST_TOP5_ACC, mean_inference_time
