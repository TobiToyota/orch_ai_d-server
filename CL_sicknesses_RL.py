from torch import nn, load, reshape
#from torch.nn import functional as F
from torchvision import models, transforms
#import torchvision.transforms.functional as TF

#from os import listdir
#from PIL import Image


class CL_sicknesses():
    def __init__(self, model_path, device, image_size=(32, 32), class_count=5) -> None:
        self.device = device
        self.class_count = class_count

        self.model = models.resnet50(weights=False).to(self.device)
        self.model.fc = nn.Sequential(
                    nn.Linear(2048, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 5),
                    nn.Sigmoid()).to(self.device)
        self.model.load_state_dict(load(model_path, map_location=self.device))
        
        self.model.eval()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.data_transform = transforms.Compose([
                                                transforms.Resize(image_size),
                                                transforms.ToTensor(),
                                                self.normalize
                                            ])
        
    def change_device(self, device):
        self.device = device
        self.model.to(device)
        
    def classify(self, images, orchid_part, weights):
        
        predictions = []
        predictions_raw = []
        preds = []
        
        #weights = torch.from_numpy(weights).to("cuda")
        #print(weights)
        
        if len(images) > 0:
        
            for image in images:
                #x = TF.to_tensor(image).to(self.device)
                #x.unsqueeze_(0)
                x = self.data_transform(image).to(self.device)
                x = reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))

                output = self.model(x)
                
                predictions_raw.append(output)
                
                if orchid_part == "flower":
                    output[0][0] *= weights[0]
                    output[0][1] *= weights[1]
                    output[0][2] *= weights[2]
                    output[0][3] *= weights[3]
                    output[0][4] *= weights[4]
                
                if orchid_part == "leaf":
                    output[0][0] *= weights[5]
                    output[0][1] *= weights[6]
                    output[0][2] *= weights[7]
                    output[0][3] *= weights[8]
                    output[0][4] *= weights[9]
                    
                if orchid_part == "root":
                    output[0][0] *= weights[10]
                    output[0][1] *= weights[11]
                    output[0][2] *= weights[12]
                    output[0][3] *= weights[13]
                    output[0][4] *= weights[14]
                    
                if orchid_part == "stem":
                    output[0][0] *= weights[15]
                    output[0][1] *= weights[16]
                    output[0][2] *= weights[17]
                    output[0][3] *= weights[18]
                    output[0][4] *= weights[19]
                
                
                #output[0][2] *= .25
                #output[0][1] *= .1
                #preds.append(F.softmax(output, dim=1).cpu().data.numpy())
                predictions.append(output)
                #print(output) 
                
            
            pred = 0
            counter = 0
            for cl in predictions:
                pred += cl[0]
                counter += 1
            pred_avg = pred.cpu().data.numpy() / counter#len(pred.cpu().data.numpy())
            
            pred_raw = 0
            for cl in predictions:
                pred_raw += cl[0]
            pred_raw = pred_raw.cpu().data.numpy()
            
            #print("------------------------")
            return (pred_raw, pred_avg)#preds

        return ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0])