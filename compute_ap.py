import numpy as np
import os

acc_all = []
# ap_scenes_path = '/data/zibo/cvpr22/logs/log_kn_graspnessrevised_pointnet2_v15_cs/' \
#                  'dump_epoch06_fps/ap_scenes'
ap_scenes_path = '/home/zibo/graspness/logs/log_kn_v9/dump_epoch06/ap_scenes'
print('For ', ap_scenes_path)
for index in range(100, 130):
    acc_scene = []
    for i in range(0, 256):
        path = os.path.join(ap_scenes_path, 'scene_' + str(index).zfill(4), str(i).zfill(4) + '.npy')
        acc_c = np.load(path)
        acc_scene.append(acc_c)
    acc_all.append(acc_scene)

acc_all = np.array(acc_all) * 100.
# 90 scenes * 256 images * 50 top_k * 6 len(list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
print('acc shape: ', acc_all.shape)
ap_all = np.mean(acc_all)
# ap_seen_all = np.mean(acc_all[0:30])

# ap_unseen_all = np.mean(acc_all[30:60])
# ap_novel_all = np.mean(acc_all[60:90])
print('AP: ', ap_all)
# print('AP Seen: ', ap_seen_all)
# print('AP Unseen: ', ap_unseen_all)
# print('AP Novel: ', ap_novel_all)

ap_all_2 = np.mean(acc_all[:, :, :, 0])
ap_all_4 = np.mean(acc_all[:, :, :, 1])
ap_all_8 = np.mean(acc_all[:, :, :, 3])
ap_all_10 = np.mean(acc_all[:, :, :, 4])
ap_all_12 = np.mean(acc_all[:, :, :, 5])

#
# ap_seen_2 = np.mean(acc_all[0:30, :, :, 0])
# ap_seen_4 = np.mean(acc_all[0:30, :, :, 1])
# ap_seen_8 = np.mean(acc_all[0:30, :, :, 3])
#
# ap_unseen_2 = np.mean(acc_all[30:60, :, :, 0])
# ap_unseen_4 = np.mean(acc_all[30:60, :, :, 1])
# ap_unseen_8 = np.mean(acc_all[30:60, :, :, 3])
#
#
# ap_novel_2 = np.mean(acc_all[60:90, :, :, 0])
# ap_novel_4 = np.mean(acc_all[60:90, :, :, 1])
# ap_novel_8 = np.mean(acc_all[60:90, :, :, 3 ])
#
print('\nAP all 0.2: ', ap_all_2)
print('AP all 0.4: ', ap_all_4)
print('AP all 0.8: ', ap_all_8)
print('AP all 1.0: ', ap_all_10)
print('AP all 1.2: ', ap_all_12)
#
# print('\nAP Seen 0.2: ', ap_seen_2)
# print('AP Seen 0.4: ', ap_seen_4)
# print('AP Seen 0.8: ', ap_seen_8)
#
# print('\nAP Unseen 0.2: ', ap_unseen_2)
# print('AP Unseen 0.4: ', ap_unseen_4)
# print('AP Unseen 0.8: ', ap_unseen_8)
#
# print('\nAP Novel 0.2: ', ap_novel_2)
# print('AP Novel 0.4: ', ap_novel_4)
# print('AP Novel 0.8: ', ap_novel_8)
