from ast import iter_child_nodes
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
import json
import warnings  
warnings.filterwarnings("ignore") 

Path = '/home/austin/Test_ws/src/collect/src/Datasets'
ratio = 1
count = [0,0]
Filter_0 = 1
mm = 10000

def json2action(idx):
    with open(idx,"r") as F:
        data = json.load(F)
        if data['shapes'][0]['label'] == 'upper':
            x = 160
            y = 80
        else:
            x = 160
            y = 240

    return  x, y

def logger(path, File_name):
    name_list = os.listdir(path)
    name_list.sort()
    with h5py.File(File_name,'w') as f:
        for name in name_list:
            print(name)
            num = int(name.split('_')[1])
            if num > 35:
                F = 'failed'
            else:
                F = 'sucess'

            L = os.listdir(Path+'/'+name+'/rgb')
            L.sort()
            LL = os.listdir(Path+'/'+name+'/depth')
            LL.sort()

            LLL = os.listdir(Path+'/'+name+'/Qmap')
            LLL.sort()

            # print(L,LL,LLL)

            # state image
            color = cv2.imread(Path+'/'+name+'/rgb/'+L[0])
            color = color[:,:,[2,1,0]]
            color = cv2.resize(color, (224,224))

            depth = np.load(Path+'/'+name+'/depth/'+LL[0])
            depth[depth > mm] = 0
            qmap = np.load(Path+'/'+name+'/Qmap/'+LLL[0])

            x, y = json2action(Path+'/'+name+'/rgb/'+L[1])

            # next state
            color_n = cv2.imread(Path+'/'+name+'/rgb/'+L[2])
            color_n = color[:,:,[2,1,0]]
            color_n = cv2.resize(color_n, (224,224))

            depth_n = np.load(Path+'/'+name+'/depth/'+LL[1])
            depth_n[depth_n > mm] = 0

            g1=f.create_group("iter_"+str(num))
            if F == 'failed':
                g1["reward"] = np.array([0])
            else:
                g1["reward"] = np.array([5])

            g1["action"] = np.array([y, x])
            g1["qmap"] = qmap

            g2 = g1.create_group("state")
            g2.create_dataset('color', (224,224,3), data=color)
            g2.create_dataset('depth', (224,224), data=depth)

            g3 = g1.create_group("next_state")
            g3.create_dataset('color', (224,224,3), data=color_n)
            g3.create_dataset('depth', (224,224), data=depth_n)
            g3["empty"] = np.array([True])

    print('Done')



File_name = '/home/austin/Downloads/HERoS-Dataset/Logger_view.hdf5'
# logger(Path, File_name)

id = 'iter_7'
# # Check
f = h5py.File(File_name)
# print('Get ',len(f.keys()), ' transitions')
# print('Success : ',count[0], ' Fail : ', count[1])
# print('========================')

group = f[id]
for key in group.keys():
    print(key)
# print('========================')
print(group['state'])
print(group['action'])
print(group['reward'])
print(group['qmap'])
print(group['next_state'])
# print('========================')
# for key in group['next_state']:
#     print(key)

color = f[id+'/state/color'].value
depth = f[id+'/state/depth'].value
colorn = f[id+'/next_state/color'].value
depthn = f[id+'/next_state/depth'].value

# print('========================')
# print(group['next_state/empty'])
# em = group['next_state/empty']
# print(em.value)
# print(color.shape)
# print(depth.shape)
# action = group['action']
# reward = group['reward']
# theta = group['origin_theta']
# print(action.value)
# print(reward.value)
# print(theta.value)

# _, axarr = plt.subplots(2,2) 
# axarr[0][0].imshow(color)
# axarr[0][1].imshow(depth)
# axarr[1][0].imshow(colorn)
# axarr[1][1].imshow(depthn)
# plt.show()