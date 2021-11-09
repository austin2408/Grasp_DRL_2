from scipy import ndimage
import scipy.misc
import numpy as np
import torch
import os
from math import *
import h5py
import matplotlib.pyplot as plt

from ddqn.model import reinforcement_net
from ddqn.utils import plot_figures, preprocessing, postProcessing
import warnings  
warnings.filterwarnings("ignore") 

net = reinforcement_net(use_cuda=True)

model_name = '../weight/behavior_500_0.06.pth'
net.load_state_dict(torch.load(model_name))
net = net.cuda().eval()

hdf5_path = 'Logger05.hdf5'
f = h5py.File(hdf5_path, "r")
angle = [90, -45, 0, 45]
dis_count = 0
theta_count = 0
angle_label = []
angle_pred = []
count = 0
# file = open(path+'/Test_record2.txt', "a+")
for name in f.keys():
    group = f[name]
    action = group['action'][()]
    color = f[name+'/state/color'][()]
    depth = f[name+'/state/depth'][()]
    reward = group['reward'][()]
    if reward > 0:
        count += 1
        color_tensor, depth_tensor, pad = preprocessing(color, depth)
        color_tensor = color_tensor.cuda()
        depth_tensor = depth_tensor.cuda()
        
        with torch.no_grad():
            prediction = net.forward(color_tensor, depth_tensor, is_volatile=True)

        
        result,_,_ = postProcessing(prediction, color, depth, color_tensor, pad, show=False)
        theta = angle[int(action[0])]
        angle_pred.append(int(result[0]))
        angle_label.append(int(theta))
        print(name)
        print('Pred : ',result)
        print('Label : ',[theta, action[1], action[2]])
        plt.show()
        dis_error = abs(sqrt((result[1] - action[1])**2 + (result[2] - action[2])**2))
        if (int(result[0]) !=  int(theta)):
            theta_count += 1
        print('Dis_error : ',dis_error)
        if dis_error > 20:
            dis_count += 1

        print('===================================================')
    # file.write(name+'\n'+'Pred : '+str(result)+'\n'+'Label : '+str([theta, action.value[1], action.value[2]])+'\n'+'Dis_error : '+str(dis_error)+'\n')
    # file.write('==================================================='+'\n')

print('Accuracy of position : ',(1 - dis_count/count))
print('Accuracy of theta : ',(1 - theta_count/count))
# file.write(str((1 - dis_count/len(f.keys())))+'/'+str((1 - theta_count/len(f.keys()))))
# file.close()

matrix = np.zeros((4,4))
for i in range(len(angle_pred)):
    pred = angle.index(angle_pred[i])
    label = angle.index(angle_label[i])
    matrix[pred, label] += 1

print('     90 -45 0 45')
print(' 90', matrix[0])
print('-45', matrix[1])
print('  0', matrix[2])
print(' 45', matrix[3])
