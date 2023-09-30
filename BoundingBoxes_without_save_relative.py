from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import LoadImages
import torch
from os import listdir, remove
from tqdm import tqdm

from PIL import Image
from glob import glob


class BoundingBoxes():
    def __init__(self, model_path, device) -> None:
        self.device = device
        self.predictor = attempt_load(model_path, map_location=self.device)
        self.stride = int(self.predictor.stride.max())
        self.imgsz = check_img_size(640, s=self.stride)
        
        self.predictor.eval()
        
        self.cl = [0, 1, 2, 3]
        self.classes = ["flower", "leaf", "root", "stem"]

    def split_image(self, image_path):
        
        dataset = LoadImages(image_path, img_size=self.imgsz, stride=self.stride)
        
        boxes = {"flower":[], 
                 "leaf":[], 
                 "root":[], 
                 "stem":[]}
        
        relatives = {"flower":[], 
                 "leaf":[], 
                 "root":[], 
                 "stem":[]}
        
        for ind, (path, im, im0s, vid_cap) in enumerate(dataset):
            try:
                #im = cv2.imread(filename)
                
                im = torch.from_numpy(im).to(self.device)
                im = im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if im.ndimension() == 3:
                    im = im.unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.predictor(im)[0]
                #print(predictor)

                pred = non_max_suppression(outputs, 0.3, 0.45, classes=self.cl)
                
                #print(im)

                if len(pred) != 0:
                    for i, det in enumerate(pred):
                        if len(det):
                            
                            #print(det)
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                            
                            i = 0
                                
                            for *xyxy, conf, cls in reversed(det):
                                
                                cls = int(cls)
                                classname = self.classes[cls]
                                
                                img = Image.open(path)
                                box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                                img2 = img.crop(box)
                                if cls == 0:
                                    img2 = img2.resize((32*2, 32*2))
                                elif cls == 1:
                                    img2 = img2.resize((64*2, 64*2))
                                elif cls == 2:
                                    img2 = img2.resize((32*2, 32*2))
                                elif cls == 3:
                                    img2 = img2.resize((16*2, 32*2))
                                
                                boxes[classname].append(img2)
                                relatives[classname].append(((int(xyxy[2]) - int(xyxy[0])) / 2 + int(xyxy[0]), (int(xyxy[3]) - int(xyxy[1])) / 2 + int(xyxy[1])))
                                
                                #saving_path = self.save_path + classname + "/"
                                
                                #try:
                                 #   img2.save(saving_path + classname + str(len(listdir(saving_path))) + f'-{conf:.2f}.jpg')
                                #except:
                                 #   img2.save(self.save_path + classname + str(len(listdir(self.save_path))) + f'-{conf:.2f}.jpg')

                                i += 1
            except Exception as e:
                pass
                #print(path)
                #print(e)
        
        return boxes, relatives