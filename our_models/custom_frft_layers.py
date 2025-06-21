
import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.append('/auto/k2/aykut3/emirhan_frft')
from our_models.torch_frft.torch_frft.layer import DFrFTLayer2D


'''
This is not the final version of the FRFTPOOL layer, but it gives an idea of what of it

'''
class FrFT_Pool(nn.Module):
          def __init__(self,frac,domain,N=16):
           
            super(FrFT_Pool,self).__init__()  
            self.frac=frac
            self.domain=domain

          def forward(self,mtrx):
              
            if self.domain!='fft':
                    
                    mtrx=self.frac(mtrx)
                    mtrx=torch.fft.fftshift(mtrx,dim=-1)
                    mtrx=torch.fft.fftshift(mtrx,dim=-2)
                    B,N,H,W= mtrx.size()
                
                
                    st_H=H//4+1
                    end_H=H-st_H+1
                
                    st_W=W//4+1
                    end_W=W-st_W+1
                    
            
                    return torch.abs(mtrx[:,:,st_H:end_H,st_W:end_W])
          
            elif self.domain=='fft':
                    
                    mtrx=torch.fft.fftshift(torch.fft.fft2(mtrx,norm="ortho"),dim=-1)
                    mtrx=torch.fft.fftshift(mtrx,dim=-2)

                    B,N,H,W= mtrx.size()
                    st_H=H//4+1
                    end_H=H-st_H+1

                    st_W=W//4+1
                    end_W=W-st_W+1
                    return torch.abs(mtrx[:,:,st_H:end_H,st_W:end_W])
          
            


class FrFT_MaxAttent(nn.Module):
          def __init__(self,frac,domain="frft",N=16):
           
            super(FrFT_MaxAttent,self).__init__()  
            self.frac=frac
            self.N=N
            self.domain=domain
         
          def forward(self,mtrx):
              
                    if self.domain=="frft":
                      mtrx=torch.abs(self.frac(mtrx))
                      flattened_matrix = mtrx.reshape(mtrx.size(0),mtrx.size(1),-1)
                      topk_values, topk_indices = torch.topk(flattened_matrix, self.N,dim=-1)
                    elif self.domain=="fft":
                      mtrx=torch.abs(torch.fft.fft2(mtrx,norm="ortho"))
                      flattened_matrix = mtrx.reshape(mtrx.size(0),mtrx.size(1),-1)
                      topk_values, topk_indices = torch.topk(flattened_matrix, self.N,dim=-1)
                      

                    return topk_values
          

if __name__=="__main__":
       frft_max= FrFT_MaxAttent(DFrFTLayer2D())
       rnd_mtrx= torch.rand(50,512,7,7)
       print(frft_max(rnd_mtrx).shape)