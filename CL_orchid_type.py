#import torch
from torch import nn, load, reshape
from torch.nn import functional as F
from torchvision import models, transforms
#import torchvision.transforms.functional as TF

#from os import listdir
from PIL import Image, ImageFile
#import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CL_orchid_type():
    def __init__(self, model_path, device, image_size=(224, 224), class_count=5) -> None:
        self.device = device
        self.class_count = class_count

        self.model = models.resnet50(weights=False).to(self.device)
        self.model.fc = nn.Sequential(
                    nn.Linear(2048, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 5)).to(self.device)
        self.model.load_state_dict(load(model_path, map_location=self.device))
        
        self.model.eval()

        self.normalize = transforms.Normalize(mean=[0.5685, 0.5245, 0.4742],
                                              std=[0.3274, 0.3247, 0.3508])

        self.data_transform = transforms.Compose([
                                                transforms.Resize(image_size),
                                                transforms.ToTensor(),
                                                self.normalize
                                            ])
    
    def change_device(self, device):
        self.device = device
        self.model.to(device)
        
    def classify(self, image_path):
        
        predictions = []
        preds = []

        image = Image.open(image_path)
        x = self.data_transform(image).to(self.device)
        x = reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))
        
        output = self.model(x)
        
        preds.append(F.softmax(output, dim=1).cpu().data.numpy())
        predictions.append(output)
        
        pred = 0
        for cl in predictions:
            pred += cl[0]
            
        pred_avg = F.softmax(pred, dim=0).cpu().data.numpy()
        pred = pred.cpu().data.numpy()

        return (pred, pred_avg)#preds
