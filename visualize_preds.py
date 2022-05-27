import time
import torch
from models import PoseExpNet
from datasets.validation_folders import ValidationSetWithPose
import argparse
from imageio import imread
from path import Path
import os
import numpy as np
# import cv2
import custom_transforms
from inverse_warp import pose_vec2mat
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


plt.ion()


# @torch.no_grad()
def main():
    pred_path = './09-clemend.npy'
    data = np.load(pred_path)
    # [len(imgs), seq_len, 3, 4]
    odom_list = [data[0]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-100,100)
    ax.set_ylim(-100, 100)

    plt.ion()
    plt.show()
    verts = [(0,0)]
    codes = [Path.MOVETO]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='white', lw=2)
    ax.add_patch(patch)
    
    for i in range(1, len(data), 4):
        last_odom = odom_list[-1] # [seq_len, 3, 4]
        transition_odom = last_odom[-1] # [3, 4]
        r = transition_odom[:, :3] # [3, 3]
        t = transition_odom[:, -1:] # [3, 1]
        if np.linalg.det(r) - 1 > 1e-2:
            print('Not special orthogonal')
        sub_odom = data[i] # [seq_len, 3, 4]
        # Multiple with R
        odom = r @ sub_odom # [seq_len, 3, 4]
        # Translation
        odom[:, :, -1:] = odom[:, :, -1:] + t  # [seq_len, 3, 4]
        odom_list.append(odom[1:, :, :])
        # For visualize
        xs = odom[1:, 0, -1] # [seq_len-1, 1]
        zs = odom[1:, 2, -1] # [seq_len-1, 1]
        print(xs.shape)
        verts  = verts + [v for v in zip(xs, zs)]
        # verts.append((x, z))
        codes  = codes + [Path.LINETO for i in range(4)]
    
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='white', lw=2)
        ax.add_patch(patch)
        plt.draw()
        plt.show(block=False)
        plt.pause(0.05)

    odom_arr = np.concatenate(odom_list, 0)
    print(odom_arr.shape)
    pred_x = odom_arr[:, 0, -1]
    pred_y = odom_arr[:, 2, -1]
    print(pred_x.shape)
    print(pred_y.shape)
    # plt.plot(pred_x, pred_y, 'r-')
    # plt.gca().set_aspect('equal', adjustable='box')



if __name__ == '__main__':
    main()