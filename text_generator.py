import json
import torch

with open("./database.json", "r", encoding="UTF-8") as f:
    orchid_type = json.loads(f.read())
    
    
classes = ["trocken ist ( {} ){}", "gesund ist ( {} ){}", "Wolllaeuse hat ( {} ){}", "einen Sonnenbrand hat ( {} ){}", "ueberwaessert ist ( {} ){}"]
grammar = [", ", " oder ", ""]

types = ["Cymbidium", "Dendrobium", "Oncidium", "Phalaenopsis", "Vanda"]

add_text = "\nEs koennte aber auch sein, dass deine Orchidee {}."

def generate_text(text, pred, val, type, out_avg):
    val = round(float(val), 2)
    out = list(out_avg)
    out_avg = list(out_avg)
    out.sort()
    out.reverse()
    for o in range(len(out)):
        out[o] = round(float(out[o]), 2)
        out_avg[o] = round(float(out_avg[o]), 2)
    out_full = out.copy()
    factor = 1.0
    old_val = float(val)

    #print(val, pred, out, out_full)
    if int(float(val) * 100) > 95:
        val = 0.95
        factor = val / old_val
        for o, ou in enumerate(out):
            out[o] *= factor
            out[o] = round(out[o], 2)
            
    #print(val, pred, out, out_full)
    
    for o, ou in enumerate(out[1:]):
        if val / 2 > ou:
            out.remove(ou)
#	    out_full.remove(round(ou * factor, 3))
            
    #for o, ou in enumerate(out_full[1:]):
    #    if old_val / 2 > ou:
    #        out_full.remove(ou)
    
    #print(val, pred, out, out_full)
    
    
    text = text.format(f"{int(val * 100)}%")
     
    out_full.remove(old_val)
    out.remove(val)
    
    #print(val, pred, out, out_full)
    
    
    if len(out) > 0:
        adding_text = ""
        for o, ou in enumerate(out):
            gr = 2
            if o+1 < len(out) - 1:
                gr = 0
            elif o+1 == len(out) - 1 and len(out) > 1:
                gr = 1
            indx = int(out_avg.index(out_full[o]))
            adding_text += classes[indx].format(f"{int(ou * 100)}%", grammar[gr])
            
        text += add_text.format(adding_text)
    
    return text
