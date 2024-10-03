import torch
import torch.nn.functional as F
from torch import nn

#Thiss loss Function calculates distance between output of some VGG16 layers when input is orignal image vs generated image

class SemanticLoss(torch.nn.Module):
    def __init__(self):
        super(SemanticLoss, self).__init__()
        
    def forward(self, original_image_feature_maps, generated_image_feature_maps, original_image, generated_image):
        CL = torch.mean(torch.norm(original_image_feature_maps[2] - generated_image_feature_maps[2], dim = 1))
        CL += torch.mean(torch.norm(original_image_feature_maps[3] - generated_image_feature_maps[3], dim = 1))
        CL += torch.mean(torch.norm(original_image_feature_maps[4] - generated_image_feature_maps[4], dim = 1))
        
        v1 = original_image_feature_maps[1]
        v2 = generated_image_feature_maps[1]
        
        KL = F.kl_div(F.log_softmax(v1[0],dim = 0), F.softmax(v2[0], dim = 0))
        KL += F.kl_div(F.log_softmax(v2[0],dim = 0), F.softmax(v1[0], dim = 0))
        
        noise = noise_generate(original_image, 0.1)
        
        ED = torch.mean(torch.norm(generated_image - noise))
        loss = ED + (KL * 100000) + CL

        return loss
    
