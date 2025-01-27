#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from scipy.spatial import KDTree
import open3d as o3d

# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def chunk_occlusion(pointcloud, num_to_remove):
    # Transpose to (N, 3) for easier spatial operations
    pointcloud = pointcloud.T
    
    # Build a KDTree for efficient nearest-neighbor search
    tree = KDTree(pointcloud)
    
    # Randomly select a base point
    base_index = np.random.randint(0, pointcloud.shape[0])
    base_point = pointcloud[base_index]
    
    # Adjust radius iteratively until exactly `num_to_remove` points are selected
    radius = 0.1  # Start with a small radius
    while True:
        indices = tree.query_ball_point(base_point, radius)
        if len(indices) >= num_to_remove:
            # Select exactly `num_to_remove` points
            indices = np.random.choice(indices, size=num_to_remove, replace=False)
            break
        radius += 0.01  # Increase radius gradually

    # Mask out the selected indices
    mask = np.ones(pointcloud.shape[0], dtype=bool)
    mask[indices] = False
    pointcloud_occluded = pointcloud[mask]
    
    # Transpose back to (3, N)
    return pointcloud_occluded.T

def downsample_pointcloud(pointcloud, target_size=1024):
    current_size = pointcloud.shape[1]
    if current_size > target_size:
        selected_indices = np.random.choice(current_size, target_size, replace=False)
        pointcloud = pointcloud[:, selected_indices]
    return pointcloud

def save_pointcloud_to_ply(pointcloud, filename, color="red"):
    # Convert point cloud to Open3D format
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pointcloud.T)  # Open3D expects (N, 3)

    # Assign colors based on the color parameter
    if color == "red":
        pc.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (pointcloud.shape[1], 1)))  # Red
    elif color == "green":
        pc.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (pointcloud.shape[1], 1)))  # Green
    else:
        raise ValueError("Color not supported. Please choose 'red' or 'green'.")

    # Save to .ply file
    o3d.io.write_point_cloud(filename, pc)
    print(f"Saved point cloud to {filename} with color {color}")

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, occlusion_ratio=0.0, unseen=False, factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.occlusion_ratio = occlusion_ratio
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        if self.occlusion_ratio != 0.0:
            num_points_to_remove = int(pointcloud2.shape[1] * self.occlusion_ratio)
            #save_pointcloud_to_ply(pointcloud2, "pointcloud_before_occlusion.ply", "red")
            pointcloud2 = chunk_occlusion(pointcloud2, num_points_to_remove)
            #save_pointcloud_to_ply(pointcloud2, "pointcloud_after_occlusion.ply", "green")
            pointcloud1 = downsample_pointcloud(pointcloud1, target_size=pointcloud2.shape[1])

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break
