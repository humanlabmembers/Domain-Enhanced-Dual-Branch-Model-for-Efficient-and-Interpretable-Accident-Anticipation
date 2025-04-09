from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import cv2
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        VGG = models.vgg16(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
        self.dim_feat = 4096
        

    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output


def extract_object_features_from_video(video_path, output_path):
    for video_file in os.listdir(video_path):
        if not video_file.endswith('.mp4'):
            continue
        output_file_path = os.path.join(output_path, video_file.replace('.mp4', '.npz'))
        # if os.path.exists(output_file_path):
        #     continue

        device = "cuda:0"
        extractor = VGG16().to(device)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


        id = video_file[:-4]
        folder_name = os.path.basename(video_path)
       
        if folder_name == 'positive':
            label = 1
        elif folder_name == 'negative':
            label = 0
        else:
            raise ValueError("The folder name is neither 'positive' nor 'negative'")
        
        if label == 1:
            toa=90
        else:
            toa=101

        

        video = os.path.join(video_path, video_file)
        video_capture = cv2.VideoCapture(video)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
 
        num_frames=100
        ffeatures= np.zeros((num_frames,  4096))   #frames 100 dad, 50 ccd,
        frame_id = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret or frame_id >= num_frames:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            transformed_frame = transform(frame_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                ffeat = extractor(transformed_frame)
                ffeat =ffeat.cpu().numpy().squeeze()
                ffeatures[frame_id,  :] = ffeat

            frame_id += 1


        np.savez(output_file_path, ffeat=ffeatures,toa=toa,label=label)
        print(f'{video_file} finished')
        video_capture.release()

def reduce_features(input_path, output_path):
    fc_layer = nn.Linear(4096, 512, bias=False).to('cuda')

    fc_layer.eval()

    for video_file in os.listdir(input_path):

        
        input_file_path = os.path.join(input_path, video_file)
        
        output_file_path = os.path.join(output_path, video_file)

        data = np.load(input_file_path, allow_pickle=True)
        new_data = {}

        for array_name in data.files:
            array = data[array_name]
            if array.size == 0 or len(array.shape) == 0:
                new_data[array_name] = array
            elif array.shape[-1] == 4096:
                tensor = torch.tensor(array).to('cuda')
                with torch.no_grad():
                    reduced_tensor = fc_layer(tensor).cpu().numpy()
                new_data[array_name] = reduced_tensor
            else:
                new_data[array_name] = array

        np.savez(output_file_path, **new_data)

if __name__ == '__main__':
    set_seed(42)
    for videos in ['training/positive','training/negative','testing/negative', 'testing/positive']:
        video_path = yours
        out_path = yours
        extract_object_features_from_video(
            video_path=video_path,
            output_path=out_path
        )
    
    for videos in ['training/positive','training/negative','testing/negative', 'testing/positive']:
        input_path = yours
        output_path = yours
        reduce_features(
            input_path=input_path,
            output_path=output_path
        )
