from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import time


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

    
class TemporalAttention(nn.Module):
    def __init__(self, input_size=512, num_heads=8, output_size=2, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x = x.permute(1, 0, 2)  
        
        attn_output, _ = self.attention(x, x, x)  
        output = attn_output.permute(1, 0, 2) 
        
        output = self.fc(output)  
        return output

class DSTA(nn.Module):
    def __init__(self, dataset='dad'):
        super().__init__()
        self.device=torch.device('cuda')
        self.dataset=dataset
        class_weights = torch.tensor([1,5], dtype=torch.float32)
        if self.dataset=='dad':
            self.fps=20
        elif self.dataset=='ccd':
            self.fps=10
        elif self.dataset=='a3d':
            self.fps=20
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(512, 512 * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(512 * 4, 512))
        ]))


        self.temp=TemporalAttention()
        
    def forward(self, features,label,toa,text_feature):
        visual_features = features
        visual_features = visual_features.float()
        
        visual_features =LayerNorm(512).to(self.device)(visual_features)
        
        out= self.temp(visual_features)
       
        logits_attn = out.permute(0, 2, 1)  
        
        visual_attn = torch.matmul(logits_attn, visual_features)  
        
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        
        text_feature =LayerNorm(512).to(self.device)(text_feature)

        text_features = text_feature + visual_attn

        text_features = text_features + self.mlp2(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        

        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        
        text_features_norm = text_features_norm.permute(0, 2, 1)

       
        visual_features_norm = torch.where(torch.isnan(visual_features_norm), torch.tensor(1e-10, dtype=visual_features_norm.dtype), visual_features_norm)
        text_features_norm = torch.where(torch.isnan(text_features_norm), torch.tensor(1e-10, dtype=text_features_norm.dtype), text_features_norm)
        
        out2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07
        

        
        losses = {'cross_entropy': 0,
                  'L4':0,
                  'L5':0,
                  'total_loss': 0}
        
        
        L4=0
        

        L3=0
        

        
        L5=self.compute_mil_loss(out2, label, self.device)

        for i in range(visual_features.size(0)):
            
            video_pred, _ = torch.max(out[i, :90, :], dim=0)  # shape: [feature_dim]
            video_pred = video_pred.unsqueeze(0)  # shape: [1, feature_dim]
            
            
            video_label = label[i, 1].unsqueeze(0).to(torch.long)  # shape: [1]
            
          
            video_loss = self.ce_loss(video_pred, video_label)
            
          
            L4 += video_loss

        for t in range(visual_features.size(1)):
            L3 += self._exp_loss(out[:, t, :], label, t, toa=toa, fps=self.fps)
           
        
        losses['cross_entropy'] += L3
        losses['L4'] += L4
        losses['L5'] += L5
            
            
            

        return losses,out
    
    def compute_mil_loss(self,out, label, device):
        batch_size, num_instances, num_classes = out.shape
        

        labels = label / torch.sum(label, dim=1, keepdim=True)
        labels = labels.to(device)
        
  
        instance_logits = torch.zeros(0).to(device)
        
        for i in range(batch_size):

            tmp, _ = torch.topk(out[i], k=20, largest=True, dim=0)
 
            instance_logits = torch.cat([instance_logits, torch.mean(tmp, dim=0, keepdim=True)], dim=0)
        

        milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
        return milloss





    def _exp_loss(self, pred, target, time, toa, fps):
        '''
        :param pred:
        :param target: onehot codings for binary classification
        :param time:
        :param toa:
        :param fps:
        :return:
        '''
        # positive example (exp_loss)
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        penalty = -torch.max(torch.zeros_like(toa).to(self.device, pred.dtype), (toa.to(self.device,pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        # negative example
        neg_loss = self.ce_loss(pred, target_cls)
        
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        # print(loss.shape)
        return loss
    



    
                     

                