import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
# from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval
from graspnetAPI.graspnet_eval_zibo import GraspGroup, GraspNetEval  # change the library code to save eval results

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
# from utils.collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/bot/980A6F5E0A6F38801/datasets/graspnet')
parser.add_argument('--checkpoint_path', help='Model checkpoint path',
                    default='/data/zibo/logs/log_kn_v5/np15000_dim512_graspness1e-1_M1024_bs2_lr5e-4_viewres_dataaug_epoch05.tar')
parser.add_argument('--dump_dir', help='Dump dir to save outputs',
                    default='/data/zibo/logs/log_kn_v5/dump_epoch05')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference():
    test_dataset = GraspNetDataset(cfgs.dataset_root, split='mini_test', camera=cfgs.camera, num_points=cfgs.num_point,
                                   voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False)
    print('Test dataset length: ', len(test_dataset))
    scene_list = test_dataset.scene_list()
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
    print('Test dataloader length: ', len(test_dataloader))
    # Init the model
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    batch_interval = 100
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(test_dataloader):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()

            gg = GraspGroup(preds)
            # collision detection
            # if cfgs.collision_thresh > 0:
            #     cloud, _ = test.get_data(data_idx, return_raw_cloud=True)
            #     mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
            #     collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            #     gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if (batch_idx + 1) % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx + 1, (toc - tic) / batch_interval))
            tic = time.time()


def evaluate(dump_dir):  # changed the graspnetAPI code, it can directly save evaluating results
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test_seen')
    # ge.eval_seen(dump_folder=dump_dir, proc=6)
    ge.eval_scene(scene_id=100, dump_folder=dump_dir)


if __name__ == '__main__':
    # python test.py  --infer --eval
    if cfgs.infer:
        inference()
    # python test.py --dump_dir logs/dump_kn_decouple_quat/
    if cfgs.eval:
        evaluate(cfgs.dump_dir)
