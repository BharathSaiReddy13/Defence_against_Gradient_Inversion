
import torch
import torch.nn.functional as F
from torch import nn

#This Loss fuction caluculates crossentropy loss between model outputs on original image and model outputs on perturbed image

class perturbation_Classification_loss(torch.nn.Module):
    def __init__(self):
        super(perturbation_Classification_loss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, original_image_final_output, generated_image_final_output):
        o_pred = torch.argmax(original_image_final_output, dim = 1)
        loss = self.cls_loss(generated_image_final_output, o_pred)
        return loss