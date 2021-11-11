import cv2
from scipy import ndimage
import scipy.misc
import numpy as np
import torch
import os
from math import *
import h5py
import matplotlib.pyplot as plt

from ddqn_LWF.model_new import reinforcement_net_new
from ddqn_LWF.utils_new import plot_figures, preprocessing, postProcessing
import warnings  
warnings.filterwarnings("ignore") 

net = reinforcement_net_new(use_cuda=True)

model_name = 'weight/behavior_1000_178715_LWF.pth'
# net.grasp_net.load_state_dict(torch.load('weight/grasp_net.pth'))
# net.value_net.load_state_dict(torch.load('weight/value_net.pth'))
net.load_state_dict(torch.load(model_name))
net = net.cuda().eval()

# torch.save(net.grasp_net.state_dict(), os.path.join(
#             '/home/austin/Grasp_DRL_2/weight', "grasp_net.pth"))

# torch.save(net.value_net.state_dict(), os.path.join(
#             '/home/austin/Grasp_DRL_2/weight', "value_net.pth"))

# hdf5_path = 'Logger05.hdf5'
# f = h5py.File(hdf5_path, "r")
# angle = [90, -45, 0, 45]
# dis_count = 0
# theta_count = 0
# angle_label = []
# angle_pred = []
# count = 0

L = os.listdir('/home/austin/Test_ws/src/collect/src/Datasets')
for name in L[:1]:
    print(name)
    A = os.listdir('/home/austin/Test_ws/src/collect/src/Datasets/'+name+'/rgb')
    A.sort()
    B = os.listdir('/home/austin/Test_ws/src/collect/src/Datasets/'+name+'/depth')
    B.sort()
    # print(B)
    # print(A)

    

    color = cv2.imread('/home/austin/Test_ws/src/collect/src/Datasets/'+name+'/rgb/'+A[0])
    # print(color.shape)
    color = cv2.resize(color, (224,224))
    depth = np.load('/home/austin/Test_ws/src/collect/src/Datasets/'+name+'/depth/'+B[0])
    depth[depth>10000] == 0
    print(depth.shape)

    # print(color.shape)
    # print(depth.shape)

    color_tensor, depth_tensor, pad = preprocessing(color, depth)
    color_tensor = color_tensor.cuda()
    depth_tensor = depth_tensor.cuda()

    with torch.no_grad():
            prediction, prediction_new = net.forward(color_tensor, depth_tensor, is_volatile=True)

    _,_,aff, out = postProcessing(prediction, color, depth, color_tensor, pad, show=False)
    for i in prediction_new[0][0][0][:1]:
        print(i)
    print(len(prediction_new))
    # print(prediction_new[0].shape)
    # print(len(prediction))
    # print(prediction[0].shape)
    print(torch.max(prediction_new[0]))

    plt.imshow(aff[:,:,[2,1,0]])
    # os.mkdir('/home/austin/Test_ws/src/collect/src/Datasets/'+name+'/Qmap')
    # np.save('/home/austin/Test_ws/src/collect/src/Datasets/'+name+'/Qmap/qmap', out[0][0])
    # norm = np.linalg.norm(out[0][0])
    # out[0][0] = out[0][0]/norm
    # print(out[0][0])
    # plt.imshow(out[0][0])
    plt.show()


# # file = open(path+'/Test_record2.txt', "a+")
# for name in f.keys():
#     group = f[name]
#     action = group['action'][()]
#     color = f[name+'/state/color'][()]
#     depth = f[name+'/state/depth'][()]
#     reward = group['reward'][()]
#     if reward > 0:
#         count += 1
#         color_tensor, depth_tensor, pad = preprocessing(color, depth)
#         color_tensor = color_tensor.cuda()
#         depth_tensor = depth_tensor.cuda()
        
#         with torch.no_grad():
#             prediction = net.forward(color_tensor, depth_tensor, is_volatile=True)

        
#         result,_,_ = postProcessing(prediction, color, depth, color_tensor, pad, show=False)
#         theta = angle[int(action[0])]
#         angle_pred.append(int(result[0]))
#         angle_label.append(int(theta))
#         print(name)
#         print('Pred : ',result)
#         print('Label : ',[theta, action[1], action[2]])
#         plt.show()
#         dis_error = abs(sqrt((result[1] - action[1])**2 + (result[2] - action[2])**2))
#         if (int(result[0]) !=  int(theta)):
#             theta_count += 1
#         print('Dis_error : ',dis_error)
#         if dis_error > 20:
#             dis_count += 1

#         print('===================================================')
#     # file.write(name+'\n'+'Pred : '+str(result)+'\n'+'Label : '+str([theta, action.value[1], action.value[2]])+'\n'+'Dis_error : '+str(dis_error)+'\n')
#     # file.write('==================================================='+'\n')

# print('Accuracy of position : ',(1 - dis_count/count))
# print('Accuracy of theta : ',(1 - theta_count/count))
# # file.write(str((1 - dis_count/len(f.keys())))+'/'+str((1 - theta_count/len(f.keys()))))
# # file.close()

# matrix = np.zeros((4,4))
# for i in range(len(angle_pred)):
#     pred = angle.index(angle_pred[i])
#     label = angle.index(angle_label[i])
#     matrix[pred, label] += 1

# print('     90 -45 0 45')
# print(' 90', matrix[0])
# print('-45', matrix[1])
# print('  0', matrix[2])
# print(' 45', matrix[3])
