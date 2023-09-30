#from tkinter import *
from math import sqrt, asin

import numpy as np

from bisect import bisect

def create_breakpoints(num_bp):
    return [(x+1)/num_bp for x in range(num_bp)]

def find_groups(boxes, small=True):
    properties = []
    
    for i, c in enumerate(boxes):
        length = sqrt(c[0]**2 + c[1]**2)
        rotation = np.rad2deg(asin(c[0] / length))
        properties.append([length, rotation])
        
    properties_right = []
    
    for i, c in enumerate(boxes):
        length = sqrt((640-c[0])**2 + c[1]**2)
        rotation = np.rad2deg(asin((640-c[0]) / length))
        properties_right.append([length, rotation])
    
    Xt = []
    longest_length = sqrt(640**2 + 640**2)
    
    length_avg = 0.0
    rotation_avg = 0.0
    
    x_avg = 0.0
    y_avg = 0.0
    
    length_breakpoints = []
    rot_breakpoints = []
    x_breakpoints = []
    y_breakpoints = []
    breakpoints = []
    
    num_bp_len = 8
    num_bp_rot = 8
    num_bp_x = 4
    num_bp_y = 4
    
    if small:
        for i, (l, r) in enumerate(properties):
            box_avg_len = 0.0
            box_avg_rot = 0.0
            for i2, (l2, r2) in enumerate(properties):
                box_avg_len += abs(l - l2) / len(properties)
                box_avg_rot += abs(r - r2) / len(properties)
            length_avg += box_avg_len / (len(properties) + len(properties_right))
            rotation_avg += box_avg_rot / (len(properties) + len(properties_right))
        
        for i, (l, r) in enumerate(properties_right):
            box_avg_len = 0.0
            box_avg_rot = 0.0
            for i2, (l2, r2) in enumerate(properties_right):
                box_avg_len += abs(l - l2) / len(properties_right)
                box_avg_rot += abs(r - r2) / len(properties_right)
            length_avg += box_avg_len / (len(properties_right) + len(properties))
            rotation_avg += box_avg_rot / (len(properties_right) + len(properties))
        
        for i, (x, y) in enumerate(boxes):
            box_x_avg = 0.0
            box_y_avg = 0.0
            for i2, (x2, y2) in enumerate(boxes):
                box_x_avg += abs(x - x2) / len(boxes)
                box_y_avg += abs(y - y2) / len(boxes)
            x_avg += box_x_avg / len(boxes)
            y_avg += box_y_avg / len(boxes)
        
        try:
            num_bp_len = int(longest_length / length_avg)
            if num_bp_len > 8:
                num_bp_len = 8
            if num_bp_len <= 0:
                num_bp_len = 1
        except:
            num_bp_len = 1
        
        try:
            num_bp_rot = int(90 / rotation_avg)
            if num_bp_rot > 8:
                num_bp_rot = 8
            if num_bp_rot <= 0:
                num_bp_rot = 1
        except:
            num_bp_rot = 1
        
        try:
            num_bp_x = int(640 / x_avg)
            if num_bp_x > 4:
                num_bp_x = 4
            if num_bp_x <= 0:
                num_bp_x = 1
        except:
            num_bp_x = 1
        
        try:
            num_bp_y = int(640 / y_avg)
            if num_bp_y > 4:
                num_bp_y = 4
            if num_bp_y <= 0:
                num_bp_y = 1
        except:
            num_bp_y = 1
                
    length_breakpoints = create_breakpoints(num_bp_len)
    rot_breakpoints = create_breakpoints(num_bp_rot)
    breakpoints = create_breakpoints(int((num_bp_len + num_bp_rot) / 2))
    x_breakpoints = create_breakpoints(num_bp_x)
    y_breakpoints = create_breakpoints(num_bp_y)
    
    #print("breakpoints: ", len(length_breakpoints), len(rot_breakpoints), len(breakpoints), len(y_breakpoints), len(x_breakpoints))
    
    for length, rotation in properties:
        Xt.append(np.array([bisect(length_breakpoints, length/longest_length), bisect(rot_breakpoints, rotation/90)]))
    
    Xt_right = []
    
    for length, rotation in properties_right:
        Xt_right.append(np.array([bisect(length_breakpoints, length/longest_length), bisect(rot_breakpoints, rotation/90)]))

    Xt_pos = []
    
    for x, y in boxes:
        Xt_pos.append(np.array([bisect(x_breakpoints, x/640), bisect(y_breakpoints, y/640)]))
    
    groups = []
    
    for i in range(len(boxes)):
        group = []
        for rb in range(len(boxes)):
            prob = 0
            if Xt[rb][0] == Xt[i][0]:
                prob += 1 / num_bp_len
            if Xt[rb][1] == Xt[i][1]:
                prob +=  1 / num_bp_rot
            if Xt_right[rb][0] == Xt_right[i][0]:
                prob += 1 / num_bp_len
            if Xt_right[rb][1] == Xt_right[i][1]:
                prob +=  1 / num_bp_rot
            if Xt_pos[rb][0] == Xt_pos[i][0]:
                prob +=  1 / num_bp_x
            if Xt_pos[rb][1] == Xt_pos[i][1]:
                prob +=  1 / num_bp_y
            if prob >= 0.5:
                group.append(rb)
        groups.append(group)
        
    same_groups = []

    for i, group in enumerate(groups):
        same_group = []
        for i, gr in enumerate(groups):
            same = False
            res = [x for x in gr if x not in group]
            res2 = [x for x in group if x not in gr]
            if (res != gr and len(res) != 0) or (res2 != group and len(res2) != 0) or gr == group:
                same = True
            if len(same_group) <= i:
                same_group.append(same)
            else:
                same_group[i] = same
        
        same_groups.append(same_group)
    
    for rb in range(len(boxes)):
        for i, group in enumerate(same_groups):
            same_group = group.copy()
            for y, gr in enumerate(same_groups):
                same = False
                res = []
                
                if group[y]:
                    for x, g in enumerate(gr):
                        if g:
                            same_group[x] = True
                            
            same_groups[i] = same_group
    
    group_count = 0
    group_count_groups = []
    
    for group in same_groups:
        if group not in group_count_groups:
            group_count += 1
            group_count_groups.append(group)

    #print(group_count_groups, group_count)
    
    return same_groups, group_count_groups, group_count

def find_nearest(numbers, pos=0):
    nums = numbers.copy()
    nums.sort()
    #closest = None
    #for number in numbers:
    #    if closest is None or abs(number - target) < abs(closest - target):
    #        closest = number
    return nums[pos] #closest

def filter_groups(relatives, same_groups, group_count_groups, kind=""):
    
    group_coords = [[] for _ in range(len(group_count_groups))]
    
    for i, c in enumerate(relatives):
        group_coords[group_count_groups.index(same_groups[i])].append(c)
        
        #print(group_coords, group_count_groups.index(same_groups[i]))
        
    group_center_coords = [[0, 0] for _ in range(len(group_count_groups))]
    
    for i, c in enumerate(relatives):
        group_center_coords[group_count_groups.index(same_groups[i])][0] += c[0] / len(group_coords[group_count_groups.index(same_groups[i])])
        group_center_coords[group_count_groups.index(same_groups[i])][1] += c[1] / len(group_coords[group_count_groups.index(same_groups[i])])
        
    #for i in range(len(group_count_groups)):
     #   group_center_coords[group_count_groups.index(same_groups[i])] = (group_center_coords[group_count_groups.index(same_groups[i])][0] / len(group_coords[i]), group_center_coords[group_count_groups.index(same_groups[i])][1] / len(group_coords[i]))
    
    avg_group_distances = [0 for _ in range(len(group_count_groups))]
    
    for i, gc in enumerate(group_coords):
        for y, c in enumerate(gc):
            #plt.plot(c[0], c[1], marker='v', color="red")
            box_avg = 0.0
            for c_2 in gc:
                box_avg += sqrt(abs(c[0] - c_2[0])**2 + abs(c[1] - c_2[1])**2)
            if len(gc) > 1:
                box_avg /= len(gc) - 1
            
            avg_group_distances[i] += box_avg / len(gc)
    
    distances_to_middle = [0 for _ in range(len(group_count_groups))]
    for i, c in enumerate(group_center_coords):
        distances_to_middle[i] = sqrt(abs(320 - c[0])**2 + abs(320 - c[1])**2)
    
    group_sizes = [len(group) for group in group_coords]
    
    
    for x in range(len(distances_to_middle)):
        main_group = distances_to_middle.index(find_nearest(distances_to_middle, x))
        #print(len(group_coords[main_group]), max(group_sizes))
        if len(group_coords[main_group]) > 1 or max(group_sizes) == 1:
            #print(123)
            break
        break
    
    valid_groups = []
    
    for i, agd in enumerate(avg_group_distances):
        if ((agd >= avg_group_distances[main_group] * 0.75 and agd <= avg_group_distances[main_group] * 1.25) or avg_group_distances[main_group] == 0) and (distances_to_middle[i] >= distances_to_middle[main_group] * 0.5 and distances_to_middle[i] <= distances_to_middle[main_group] * 1.5):
            valid_groups.append(i)
    
    #print(group_center_coords, avg_group_distances, main_group, distances_to_middle, valid_groups)
    
    valid_group_coords = []
    
    for i, vg in enumerate(valid_groups):
        valid_group_coords += group_coords[vg]
    
    return valid_group_coords, group_center_coords
    
def get_new_boxes(boxes, relatives, value, smallize=True):
    valid_groups_flower, valid_groups_leaf, valid_groups_root, valid_groups_stem = relatives["flower"], relatives["leaf"], relatives["root"], relatives["stem"]
    
    box_count = len(relatives["flower"]) + len(relatives["leaf"]) + len(relatives["root"]) + len(relatives["stem"])
    
    if len(relatives["flower"]) > 0 and box_count > 0:
        small = True
        if box_count >= value:
            small = False
        if (small and smallize) or not small:
            same_groups_flower, group_count_groups_flower, group_count_flower = find_groups(relatives["flower"], small)
            #print("flower")
            valid_groups_flower, center_coords_flower = filter_groups(relatives["flower"], same_groups_flower, group_count_groups_flower)
    
    if len(relatives["leaf"]) > 0 and box_count > 0:
        small = True
        if box_count >= value:
            small = False
        if (small and smallize) or not small:
            same_groups_leaf, group_count_groups_leaf, group_count_leaf = find_groups(relatives["leaf"], small)
            #print("leaf")
            valid_groups_leaf, center_coords_leaf = filter_groups(relatives["leaf"], same_groups_leaf, group_count_groups_leaf)
    
    if len(relatives["root"]) > 0 and box_count > 0:
        small = False
        if box_count >= value:
            small = True
        if (small and smallize) or not small:
            same_groups_root, group_count_groups_root, group_count_root = find_groups(relatives["root"], small)
            #print("root")
            valid_groups_root, center_coords_root = filter_groups(relatives["root"], same_groups_root, group_count_groups_root)
    
    if len(relatives["stem"]) > 0 and box_count > 0:
        small = False
        if box_count >= value:
            small = True
        if (small and smallize) or not small:
            same_groups_stem, group_count_groups_stem, group_count_stem = find_groups(relatives["stem"], small)
            #print("stem")
            valid_groups_stem, center_coords_stem = filter_groups(relatives["stem"], same_groups_stem, group_count_groups_stem)
    
    #group_coords = [valid_groups_flower, valid_groups_leaf, valid_groups_root, valid_groups_stem]
    #center_coords = [center_coords_flower, center_coords_leaf, center_coords_root, center_coords_stem]
    
    #compare_groups(group_coords)
    
    #for i, c in enumerate(relatives["flower"]):
     #   plt.plot(c[0], c[1], marker='v', color=colors[group_count_groups_flower.index(same_groups_flower[i])])
    
    #for i, group in enumerate(valid_groups_flower):
    new_boxes = {"flower":[], 
                 "leaf":[], 
                 "root":[], 
                 "stem":[]}
    
    for i, c in enumerate(valid_groups_flower):
        new_boxes["flower"].append(boxes["flower"][relatives["flower"].index(c)])
    
    for i, c in enumerate(valid_groups_leaf):
        new_boxes["leaf"].append(boxes["leaf"][relatives["leaf"].index(c)])
    
    for i, c in enumerate(valid_groups_root):
        new_boxes["root"].append(boxes["root"][relatives["root"].index(c)])

    for i, c in enumerate(valid_groups_stem):
        new_boxes["stem"].append(boxes["stem"][relatives["stem"].index(c)])
        
    return new_boxes


    
    value = 15
    
    new_boxes = get_new_boxes(boxes, relatives, value)
