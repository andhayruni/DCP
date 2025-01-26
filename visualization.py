import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from data import ModelNet40
from data import ScanNet
from model import DCP
import pymeshlab
import argparse




def get_rot_trans(model_path, dataset_name, pointcloud_id, args):

    # load the weights of the model
    net = DCP(args).cuda()
    net.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

    # load the dataset
    if dataset_name == 'modelnet40':
        dataset = ModelNet40(num_points=1024, partition='test', gaussian_noise=False, unseen=False, factor=4)
    elif dataset_name == 'scannet':
        dataset = ScanNet(num_points=1024, partition='test', gaussian_noise=False, unseen=False, factor=4)
    else:
        raise Exception("dataset not implemented")
    src, tgt = dataset.__getitem__(pointcloud_id)[:2]

    # Convert the point clouds to tensors
    src_tensor = torch.from_numpy(src).unsqueeze(0).cuda()
    tgt_tensor = torch.from_numpy(tgt).unsqueeze(0).cuda()

    # Get the output of the model
    with torch.no_grad():
        R_ab, t_ab, R_ba, t_ba = net(src_tensor,tgt_tensor)

    # Convert to numpy arrays
    R_ab , t_ab , R_ba , t_ba = R_ab[0].cpu().numpy() , t_ab[0].cpu().numpy() , R_ba[0].cpu().numpy() , t_ba[0].cpu().numpy()

    # Apply the rotations and the translations
    rotation_ab = Rotation.from_matrix(R_ab)
    rotation_ba = Rotation.from_matrix(R_ba)

    src_to_tgt = rotation_ab.apply(src.T) + t_ab
    tgt_to_src = rotation_ba.apply(tgt.T) + t_ba



    return [src.T, tgt.T, src_to_tgt, tgt_to_src] , [R_ab, t_ab , R_ba, t_ba]



def visualization(model_path, dataset_name, pointcloud_id, args):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VISU_DIR = os.path.join(BASE_DIR, 'visualization')
    if not os.path.exists(VISU_DIR):
        os.mkdir(VISU_DIR)

    point_clouds , rigid_transformations = get_rot_trans(model_path,dataset_name,pointcloud_id, args)

    ms = pymeshlab.MeshSet()

    for i in range(2):
        ms.clear()
        mesh = pymeshlab.Mesh(vertex_matrix=point_clouds[i].astype(np.float64))
        ms.add_mesh(mesh)
        ms.save_current_mesh(os.path.join(VISU_DIR, f'ptcloud_{pointcloud_id}_{i}.ply'))
    
    for i in range(2):
        ms.clear()
        mesh = pymeshlab.Mesh(vertex_matrix=point_clouds[i+2].astype(np.float64))
        ms.add_mesh(mesh)
        ms.save_current_mesh(os.path.join(VISU_DIR, f'transformed_ptcloud_{pointcloud_id}_{i}.ply'))

def main():
    parser = argparse.ArgumentParser(description='DCP Visualization')

    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40','scannet'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--id', type=int, default=0, metavar='N',
                        help='id of the object to visualize')
    
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if not( args.model_path ):
        if not( args.exp_name ):
            raise Exception("no model specified")
        else:
            path = os.path.join(BASE_DIR,'checkpoints',args.exp_name,'models','model.best.t7')
    else:
        path = args.model_path
    
    visualization(path, args.dataset, args.id, args)
    

if __name__ == '__main__':
    main()