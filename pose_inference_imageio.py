import time
import torch
from models import PoseExpNet
import argparse
from imageio import imread
from path import Path as FilePath
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

import argparse

parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    # pred_path = './09-clemend.npy'
    # data = np.load(pred_path)
    # # [len(imgs), seq_len, 3, 4]
    folder = FilePath('./kitti09/09_2')
    img_files = sorted(folder.files('*.jpg'))
    weights = torch.load(args.pretrained_posenet, map_location=device)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    odom_list = []
    input_list = []

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
    
    # print(img_files)
    for file in img_files:
        input_list.append(file)
        print(len(input_list))
        # print(file)
        if len(input_list) < seq_length:
            continue
        raw_imgs = [imread(input_list[i]).astype(np.float32) for i in range(len(input_list))]
        imgs = [np.transpose(img, (2,0,1)) for img in raw_imgs]
        ref_imgs = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.5)/0.5).to(device)
            if i == len(imgs)//2:
                tgt_img = img
            else:
                ref_imgs.append(img)

        input_list.pop(0)
        print(len(input_list))
        _, poses = pose_net(tgt_img, ref_imgs)

        poses = poses.cpu()[0]
        poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])

        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]
        odom_list.append(final_poses)

        last_odom = odom_list[-1] # [seq_len, 3, 4]
        transition_odom = last_odom[-1] # [3, 4]
        r = transition_odom[:, :3] # [3, 3]
        t = transition_odom[:, -1:] # [3, 1]
        if np.linalg.det(r) - 1 > 1e-2:
            print('Not special orthogonal')
        sub_odom = final_poses # [seq_len, 3, 4]
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