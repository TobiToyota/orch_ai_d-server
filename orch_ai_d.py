from torch import max, from_numpy, device, cuda
#from torch.nn import functional as F

from working_relatives import get_new_boxes

import numpy as np
from glob import glob
import json
from pathlib import Path
import os
import text_generator
#import time

classes = ["dry", "healthy", "mealybugs", "sunburn", "wet"]

types = ["Cymbidium", "Dendrobium", "Oncidium", "Phalaenopsis", "Vanda"]

with open("./database.json", "r", encoding="UTF-8") as f:
    orchid_type = json.loads(f.read())

weights = [0.7750201677551428, 0.7605443347010628, 1.466845478443798, 0.7660074123201902, 0.7060079564259714, 1.6312053623176408, 0.747878471062351, 2.5735485474732194, 2.13853914236755, 1.805220753178268, 1.4898408352773849, 0.6974312213394122, 0.5524323169097582, 1.0149075371660283, 1.4662394447469342, 0.526634286696205, 0.6051243506565994, 0.4775711906676842, 0.7468073867786461, 0.6572072934499351]

#weights = [0.0, 0.6255515213064848, 0.8572908505396275, 0.0, 0.0, 
#          1.2209097277703498, 0.6503448966109107, 1.9631505388262305, 2.3995733753933095, 1.9496851277480838, 
#          1.3225803902830133, 0.5976212103750256, 0.0, 0.0, 1.116477523935857]

#images = glob("./input/"  + '*.jpg')
#for img in images:
def orch_ai_d(img, bb, cl_flower, cl_leaf, cl_root, cl_stem, cl_orchid_type, devic, dir="./input"):
    
    #bb.change_device(devic)
    #cl_flower.change_device(devic)
    #cl_leaf.change_device(devic)
    #cl_root.change_device(devic)
    #cl_stem.change_device(devic)
    #cl_orchid_type.change_device(devic)
    
    boxes, relatives = bb.split_image(f"{dir}/{img}")
    #print(123)
    boxes = get_new_boxes(boxes, relatives, value=9, smallize=False)
    #print(3213)
    out_flower = cl_flower.classify(boxes["flower"], "flower", weights=weights)
    out_leaf = cl_leaf.classify(boxes["leaf"], "leaf", weights=weights)
    out_root = cl_root.classify(boxes["root"], "root", weights=weights)
    out_stem = cl_stem.classify(boxes["stem"], "stem", weights=weights)#([0, 0, 0, 0, 0], [0, 0, 0, 0, 0])#
    #print(123)
    outputs = [out_flower, out_leaf, out_root, out_stem]
    out = 0.0
    counter = 0
    
    for o in outputs:
        if type(o[1] == [0, 0, 0, 0, 0]) != bool:
            out += np.array(o[1])
            counter += 1

    if counter > 0:
        out_avg = out / counter
        val, prediction = max(from_numpy(out_avg), 0)
    
        img_path = str(Path(f"{dir}/{img}").absolute())
        
        out_orchid_types = cl_orchid_type.classify(img_path)
        
        _, type_prediction = max(from_numpy(out_orchid_types[1]), 0)
        
        p = "low p"
        
        if int(out_avg[prediction]) > .7:
            p = "high p"

        out_final = orchid_type[types[int(type_prediction)]][classes[int(prediction)]][p]
        
        out_final = text_generator.generate_text(out_final, prediction, val, type_prediction, out_avg)
        
        print("------------------------------------------------------------------------------")
        print(f"image: {img}, prediction: {classes[int(prediction)]}, orchid_type: {types[int(type_prediction)]}")
        print("--------------------------------------")
        print(f"out added: {out}")
        print(f"out in %: {out_avg * 100}")
        print("-------------------")
        print(f"out flower: {out_flower[1]}")
        print(f"out leaf: {out_leaf[1]}")
        print(f"out root: {out_root[1]}")
        #print(f"out stem: {out_stem[1]}")
        print(f"out orchid_type: {out_orchid_types[1]}")
        
        dir_new = f"{dir}/{types[int(type_prediction)]}/{classes[int(prediction)]}"
        
        os.rename(f"{dir}/{img}", f"{dir_new}/{len(os.listdir(dir_new))}-{img}")
        
    else:
        out_avg = np.array([0, 0, 0, 0, 0])
        out_orchid_types = [0, np.array([0, 0, 0, 0, 0])]
        out_final = "Ich konnte keine Orchidee in diesem Bild erkennen. Versuche es noch einmal!"

    
    print("--------------------------------------")
    print(out_final)
    print("--------------------------------------")
    
    
    
    return out_final#out_avg, out_orchid_types[1]

if __name__ == "__main__":
    from BoundingBoxes_without_save import BoundingBoxes
    from CL_sicknesses_RL import CL_sicknesses
    from CL_orchid_type import CL_orchid_type

    devic = device("cuda:0" if cuda.is_available() else "cpu")

    bb = BoundingBoxes("./weights/BoundingBoxes.pt", devic)

    cl_flower = CL_sicknesses("./weights/best-flower.h5", devic, image_size=(32*2, 32*2))
    cl_leaf = CL_sicknesses("./weights/best-leaf-30.h5", devic, image_size=(64*2, 64*2))
    cl_root = CL_sicknesses("./weights/best-root-257.h5", devic, image_size=(32*2, 32*2))
    cl_stem = CL_sicknesses("./weights/best-stem-1971.h5", devic, image_size=(16*2, 32*2))
    
    cl_orchid_type = CL_orchid_type("./weights/best-orchid_type-1.h5", devic, image_size=(224, 224))
    
    images = glob("./input/"  + '*.jpg')
    for img in images:
        orch_ai_d(img, bb, cl_flower, cl_leaf, cl_root, cl_stem, cl_orchid_type, devic)
