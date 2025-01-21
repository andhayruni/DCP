#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


#######################
### ModelNet40 data ###
#######################


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


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
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

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]




####################
### ScanNet data ###
####################



# Imports and variables to download ScanNet public data release

import urllib.request
import tempfile
import json
import pymeshlab

import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = 'http://kaldir.vc.in.tum.de/scannet/'
TOS_URL = BASE_URL + 'ScanNet_TOS.pdf'
FILETYPES = ['.aggregation.json', '.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply', '_vh_clean.segs.json', '_vh_clean.aggregation.json', '_vh_clean_2.labels.ply', '_2d-instance.zip', '_2d-instance-filt.zip', '_2d-label.zip', '_2d-label-filt.zip']
FILETYPES_TEST = ['.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.ply']
PREPROCESSED_FRAMES_FILE = ['scannet_frames_25k.zip', '5.6GB']
TEST_FRAMES_FILE = ['scannet_frames_test.zip', '610MB']
LABEL_MAP_FILES = ['scannetv2-labels.combined.tsv', 'scannet-labels.combined.tsv']
DATA_EFFICIENT_FILES = ['limited-reconstruction-scenes.zip', 'limited-annotation-points.zip', 'limited-bboxes.zip', '1.7MB']
GRIT_FILES = ['ScanNet-GRIT.zip']
RELEASES = ['v2/scans', 'v1/scans']
RELEASES_TASKS = ['v2/tasks', 'v1/tasks']
RELEASES_NAMES = ['v2', 'v1']
RELEASE = RELEASES[0]
RELEASE_TASKS = RELEASES_TASKS[0]
RELEASE_NAME = RELEASES_NAMES[0]
LABEL_MAP_FILE = LABEL_MAP_FILES[0]
RELEASE_SIZE = '1.2TB'
V1_IDX = 1

# Functions to download ScanNet public data release

def get_release_scans(release_file):
    scan_lines = urllib.request.urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_id = scan_line.decode('utf8').rstrip('\n')
        scans.append(scan_id)
    return scans

def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    print('\t' + url + ' > ' + out_file)
    fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
    f = os.fdopen(fh, 'w')
    f.close()
    urllib.request.urlretrieve(url, out_file_tmp)
    os.rename(out_file_tmp, out_file)

def download_scan(scan_id, out_dir, file_types, use_v1_sens, skip_existing=False):
    print('Downloading ScanNet ' + RELEASE_NAME + ' scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        v1_sens = use_v1_sens and ft == '.sens'
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + scan_id + ft if not v1_sens else BASE_URL + RELEASES[V1_IDX] + '/' + scan_id + '/' + scan_id + ft
        out_file = out_dir + '/' + scan_id + ft
        if skip_existing and os.path.isfile(out_file):
            continue
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)

def download_SN(num_points):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    splits = ['train','test']

    num_train = 224
    num_test = 41

    # Find from where to start / continue the downloading
    split_start = 0
    scan_number = 0
    scan_start = 0
    object_id = 0
    if os.path.exists(DATA_DIR):
        if os.path.exists(os.path.join(DATA_DIR,'data.json')):
            with open(os.path.join(DATA_DIR,'data.json'), 'r') as f:
                json_data = json.load(f)
            scan_number = json_data['scan_number']
            scan_start = scan_number
            object_id = json_data['size']
            # If all the scans have already been downloaded
            if scan_number == num_train+num_test:
                return
            # If we the train data has already been completely downloaded but not the test data 
            if scan_number >= num_train:
                split_start = 1
        else:
            json_data = {}
            json_data['train'] = {}
            json_data['train']['object_group'] = []
            json_data['test'] = {}
            json_data['test']['object_group'] = []
    else:
        os.mkdir(DATA_DIR)
        json_data = {}
        json_data['train'] = {}
        json_data['train']['object_group'] = []
        json_data['test'] = {}
        json_data['test']['object_group'] = []

    release_file = BASE_URL + RELEASE + '.txt'
    release_scans = get_release_scans(release_file)
    # File types to be downloaded
    scan_file_types = ['_vh_clean.ply','_vh_clean.aggregation.json','_vh_clean.segs.json']
    for split in splits[split_start:]:

        if split == 'train':
            scan_ids = release_scans[scan_start:num_train]
        else:
            scan_ids = release_scans[max(num_train,scan_start):num_train+num_test]
        
        for scan_id in scan_ids:
            # Download the ScanNet data
            out_dir = os.path.join(DATA_DIR, scan_id)
            download_scan(scan_id,out_dir,scan_file_types,True)

            ### Extract each object's point clouds from the mesh

            # load the ScanNet data
            ply_file = os.path.join(out_dir, scan_id+'_vh_clean.ply')
            json_agg_file = os.path.join(out_dir, scan_id+'_vh_clean.aggregation.json')
            json_segs_file = os.path.join(out_dir, scan_id+'_vh_clean.segs.json')
            with open(json_agg_file,'r') as f:
                aggregation_data = json.load(f)
            with open(json_segs_file, 'r') as f:
                segs_data = json.load(f)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(ply_file)
            mesh = ms.current_mesh()

            # Extract the vertices from the mesh
            vertices = mesh.vertex_matrix()
            vertex_colors = mesh.vertex_color_matrix()

            # Extract each instance's segments and the vertices' segment IDs
            seg_groups = aggregation_data['segGroups']  # Informations des segments
            seg_indices = np.array(segs_data['segIndices'])

            # Create a mask for each object and save the exctracted point cloud as PLY
            for instance in seg_groups:
                instance_id = instance['id']
                instance_label = instance['label']
                instance_segments = np.array(instance['segments'])

                mask = np.isin(seg_indices,instance_segments)
                instance_points = vertices[mask]
                instance_colors = vertex_colors[mask]

                if instance_points.shape[0] < num_points:
                    continue

                ms_subset = pymeshlab.Mesh(vertex_matrix=instance_points,v_color_matrix=instance_colors)
                ms.clear()
                ms.add_mesh(ms_subset)
                object_path = os.path.join(out_dir,f'instance_{instance_id}_{instance_label}.ply')
                ms.save_current_mesh(object_path)

                json_data[split]['object_group'].append(
                    {
                    'id': object_id,
                    'label': instance_label,
                    'path': object_path
                    }
                )
                object_id+=1

            # Remove the unused ScanNet data
            os.system('rm %s' % (ply_file))
            os.system('rm %s' % (json_agg_file))
            os.system('rm %s' % (json_segs_file))

            scan_number += 1

            # Save the json data at each processed scan
            json_data['size'] = object_id
            json_data['scan_number'] = scan_number
            with open(os.path.join(DATA_DIR,'data.json'), 'w') as f:
                json.dump(json_data, f, indent=4)


def load_data_SN(partition,num_points):
    download_SN(num_points)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    with open(os.path.join(DATA_DIR,'data.json'), 'r') as f:
        data = json.load(f)
    
    dataset = data[partition]
    objects = dataset['object_group']

    all_data = []
    all_label = []

    for object in objects:
        label = object['label']

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(object['path'])
        mesh = ms.current_mesh()
        vertices = mesh.vertex_matrix()

        indices = np.random.choice(vertices.shape[0], num_points, replace=False)
        vertices = vertices[indices]

        # Center and fit the point cloud in the unit sphere
        vertices = vertices - np.mean(vertices, axis=0)
        max_distance = np.max(np.linalg.norm(vertices, axis=1))
        vertices = vertices / max_distance

        all_label.append(label)
        all_data.append(vertices)

    return np.array(all_data), np.array(all_label)

class ScanNet(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4):
        self.data, self.label = load_data_SN(partition,num_points)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
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
    train = ScanNet(1024)
    test = ScanNet(1024, 'test')
    for data in train:
        print(len(data))
        break
