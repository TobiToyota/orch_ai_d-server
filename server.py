from time import time
from flask import Flask, request
from orch_ai_d import orch_ai_d
from BoundingBoxes_without_save_relative import BoundingBoxes
from CL_sicknesses_RL import CL_sicknesses
from CL_orchid_type import CL_orchid_type
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
devic = "cpu"

classes = ["dry", "healthy", "mealybugs", "sunburn", "wet"]

types = ["Cymbidium", "Dendrobium", "Oncidium", "Phalaenopsis", "Vanda"]

bb = BoundingBoxes("./weights/BoundingBoxes.pt", devic)

cl_flower = CL_sicknesses("./weights/best-flower.h5", devic, image_size=(32*2, 32*2))
cl_leaf = CL_sicknesses("./weights/best-leaf-30.h5", devic, image_size=(64*2, 64*2))
cl_root = CL_sicknesses("./weights/best-root-257.h5", devic, image_size=(32*2, 32*2))
cl_stem = CL_sicknesses("./weights/best-stem-1971.h5", devic, image_size=(16*2, 32*2))

cl_orchid_type = CL_orchid_type("./weights/best-orchid_type-1.h5", devic, image_size=(224, 224))


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.remote_addr, request.headers)
        file = request.files['file']
        dir_path = f"./images/{request.remote_addr}"
    
        try:
            os.mkdir(dir_path)
            for typ in types:
                os.mkdir(f"{dir_path}/{typ}")
                for sickness in classes:
                    os.mkdir(f"{dir_path}/{typ}/{sickness}")
                
        except:
            try:
                for typ in types:
                    os.mkdir(f"{dir_path}/{typ}")
                    for sickness in classes:
                        os.mkdir(f"{dir_path}/{typ}/{sickness}")
                    
            except:
                pass
        
        file_name = f"{request.remote_addr}-{time()}.jpg"    
        
        file.save(f"./images/{request.remote_addr}/{file_name}")
        output_text = orch_ai_d(file_name, bb, cl_flower, cl_leaf, cl_root, cl_stem, cl_orchid_type, devic, dir_path)
        return output_text


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7777)
    
    
    
    
