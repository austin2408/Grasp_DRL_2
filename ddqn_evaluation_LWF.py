import cv2
from scipy import ndimage
import scipy.misc
import numpy as np
import torch
import os
import json
from math import *
import h5py
import matplotlib.pyplot as plt

from ddqn_LWF.model_new import reinforcement_net_new
from ddqn_LWF.utils_new import plot_figures, preprocessing, postProcessing
import warnings  
warnings.filterwarnings("ignore") 

net = reinforcement_net_new(use_cuda=True)

model_name = 'weight/behavior_LWF_10_1500_0.00.pth'
net.load_state_dict(torch.load(model_name))
# net.grasp_net.load_state_dict(torch.load('weight/grasp_net.pth'))
# net.value_net.load_state_dict(torch.load('weight/value_net.pth'))
net = net.cuda().eval()

# torch.save(net.grasp_net.state_dict(), os.path.join(
#             '/home/austin/Grasp_DRL_2/weight', "grasp_net.pth"))

# torch.save(net.value_net.state_dict(), os.path.join(
#             '/home/austin/Grasp_DRL_2/weight', "value_net.pth"))


L = os.listdir('/home/austin/Test_ws/src/collect/src/Datasets_view')
# L.sort()
# print(L)
up = [0,0]
fr = [0,0]
x = []
y = []
for name in L:
# for name in L[:1]:
# for name in L[345:355]:
    # print(name)
    print(100*(up[1]+fr[1])/len(L), end='\r')
    num = int(name.split('_')[1])
    A = os.listdir('/home/austin/Test_ws/src/collect/src/Datasets_view/'+name+'/rgb')
    A.sort()
    B = os.listdir('/home/austin/Test_ws/src/collect/src/Datasets_view/'+name+'/depth')
    B.sort()
    # print(B)
    # print(A)

    

    color = cv2.imread('/home/austin/Test_ws/src/collect/src/Datasets_view/'+name+'/rgb/'+A[0])
    # print(color.shape)
    color = cv2.resize(color, (224,224))
    depth = np.load('/home/austin/Test_ws/src/collect/src/Datasets_view/'+name+'/depth/'+B[0])
    depth[depth>10000] == 0


    color_tensor, depth_tensor, pad = preprocessing(color, depth)
    color_tensor = color_tensor.cuda()
    depth_tensor = depth_tensor.cuda()

    with torch.no_grad():
            prediction, prediction_new = net.forward(color_tensor, depth_tensor, is_volatile=True)

    _,_,aff, out, view = postProcessing(prediction, prediction_new, color, depth, color_tensor, pad, show=False)
    # print(view.shape)
    id = np.where(view == np.max(view))
    cv2.circle(view, (id[1][0], id[0][0]), 3, (0, 0, 0), 2)
    # print(id[1][0])
    x.append(id[0][0])
    y.append(id[1][0])
    with open('/home/austin/Test_ws/src/collect/src/Datasets_view/'+name+'/rgb/'+A[1],"r") as F:
        data = json.load(F)
        if data['shapes'][0]['label'] == 'upper':
            if num < 350:
                up[1] += 1
                if id[1][0] < 112:
                    up[0] += 1
            else:
                fr[1] += 1
                if id[1][0] > 112:
                    fr[0] += 1

        else:
            if num < 350:
                fr[1] += 1
                if id[1][0] > 112:
                    fr[0] += 1
            else:
                up[1] += 1
                if id[1][0] < 112:
                    up[0] += 1

    # fig = plt.figure(figsize=(10, 10))
    # fig.add_subplot(1, 3, 1)
    # plt.imshow(color[:,:,[2,1,0]])

    # fig.add_subplot(1, 3, 2)
    # plt.imshow(aff[:,:,[2,1,0]])
    # # plt.imshow(out[0][0])

    # fig.add_subplot(1, 3, 3)
    # plt.imshow(view)

    # plt.show()

    # os.mkdir('/home/austin/Test_ws/src/collect/src/Datasets_view/'+name+'/Qmap')
    # np.save('/home/austin/Test_ws/src/collect/src/Datasets_view/'+name+'/Qmap/qmap', out[0][0])
    # norm = np.linalg.norm(out[0][0])
    # out[0][0] = out[0][0]/norm
    # print(out[0][0])
    # plt.imshow(out[0][0])

    # plt.show()
print(up[0]/up[1])
# print(up[0],up[1])
print(fr[0]/fr[1])
plt.scatter(y, x)
# plt.savefig('V6_1000.png', dpi=300)
plt.show()

#IfiveOO
# vtwo0.17624521072796934
# 0.7782426778242678
# vthree0.29118773946360155
# 0.7280334728033473
# vfour0.20306513409961685
# 0.8326359832635983
# vfive0.5977011494252874
# 0.4225941422594142