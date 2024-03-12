import cv2
import os
import json
import numpy as np
from PIL import Image
import imgviz
import open3d as o3d
from open3d.examples.geometry.point_cloud_outlier_removal_statistical import display_inlier_outlier

from point import PointCloudHelper

# imgs_path = r"E:\Processed\dress-long-sleeve\dress-long-sleeve_001_20230731_134137\000673513312\color"
# for listx in os.listdir(imgs_path):
#     print(listx)
# file_num = len([listx for listx in os.listdir(imgs_path)])
# img = cv2.imread(os.path.join(imgs_path,'0.png'))  #读取第一张图片
# imgInfo = img.shape
# size = (imgInfo[1],imgInfo[0])  #获取图片宽高度信息
# print(file_num)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# videoWrite = cv2.VideoWriter('test.mp4',fourcc,10,size,True)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））
#
# for i in range(1, file_num):
#     fileName = str(i*3)+'.png'    #循环读取所有的图片
#     img = cv2.imread(os.path.join(imgs_path,fileName))
#     print(img.shape)
#     videoWrite.write(img)# 将图片写入所创建的视频对象
#
# videoWrite.release()
# print('end!')
def imgs2video(root_path=r"C:\Users\robotflow\Desktop\SAM\data0916\dress-long-sleeve_007_20230731_143236"):
    camera_info_json_path = os.path.join(root_path, 'cameras_info.jsonl')
    camera_info_json = json.load(open(camera_info_json_path, 'r'))
    print(camera_info_json.keys())
    for listx in os.listdir(root_path):
        if listx.split("_")[0] == "SaveClip":
            json_file_name = ""
            for listx2 in os.listdir(os.path.join(root_path, listx)):
                if listx2.split("_")[0] == "HandPose":
                    json_file_name = listx2
                    break
            # print(json_file_name)
            valid_json_path = os.path.join(root_path, listx, json_file_name)
            valid_json = json.load(open(valid_json_path, 'r'))
            valid_idexs = list(valid_json.keys())
            print(valid_idexs)
            file_num = len(valid_idexs)
            img = cv2.imread(os.path.join(root_path, list(camera_info_json.keys())[0], "color", valid_idexs[0] + '.png'))
            size = (img.shape[1], img.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            for camera_name in camera_info_json.keys():
                # if(os.path.exists())
                videoWriter = cv2.VideoWriter(os.path.join(root_path, listx, camera_name + ".mp4"), fourcc, 10, size, True)
                for idex in valid_idexs:
                    img = np.zeros((size[1], size[0], 3), np.uint8)
                    if os.path.exists(os.path.join(root_path, camera_name, "color", idex + '.png')):
                        img = cv2.imread(os.path.join(root_path, camera_name, "color", idex + '.png'))
                    videoWriter.write(img)
                videoWriter.release()


def mask2pc_test(masks_dir=r"C:\Users\robotflow\Desktop\SAM\Segment-and-Track-Anything-1.5\tracking_results\000673513312\000673513312_masks",
                 valid_hand_pose_json_path=r"C:\Users\robotflow\Desktop\SAM\data0916\dress-long-sleeve_007_20230731_143236\SaveClip_156_702\HandPose_156_702.json"):
    masks_paths = os.listdir(masks_dir)
    if not os.path.exists(os.path.join(masks_dir, "test")):
        os.mkdir(os.path.join(masks_dir, "test"))
    print(masks_paths)
    valid_json = json.load(open(valid_hand_pose_json_path, 'r'))
    valid_idexs = sorted(list(valid_json.keys()))
    for i, mask_name in enumerate(masks_paths):
        if not mask_name.endswith(".png"):
            continue
        # if not mask_name=="00100.png":
        #    continue
        label = np.asarray(Image.open(os.path.join(masks_dir, mask_name)), dtype=np.int32)
        # img = cv2.imread(os.path.join(masks_dir, mask_name))
        print(label.shape)
        print(label[250:300, 450:500])
        label = label == 1
        lbl_pil = Image.fromarray(label.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(os.path.join(masks_dir, "test", mask_name))

        # break


def mask2pc(clip_root=r"C:\Users\robotflow\Desktop\SAM\data0916\dress-long-sleeve_007_20230731_143236\SaveClip_156_702"):
    if not os.path.exists(os.path.join(clip_root, "pcds")):
        os.mkdir(os.path.join(clip_root, "pcds"))

    root_path = os.path.dirname(clip_root)
    camera_info_json_path = os.path.join(root_path, 'cameras_info.jsonl')
    camera_info_json = json.load(open(camera_info_json_path, 'r'))
    camera_extrinsics = {name: camera_info_json[name]['extrinsic'] for name in camera_info_json.keys()}
    for name in camera_info_json.keys():
        camera_info_json[name]["K"][0][0] = camera_info_json[name]["fxy"][0]
        camera_info_json[name]["K"][1][1] = camera_info_json[name]["fxy"][1]
        camera_info_json[name]["K"][0][2] = camera_info_json[name]["cxy"][0]
        camera_info_json[name]["K"][1][2] = camera_info_json[name]["cxy"][1]

    json_file_name = ""
    for listx in os.listdir(clip_root):
        if listx.split("_")[0] == "HandPose":
            json_file_name = listx
            break
    if json_file_name == "":
        print("HandPose***.json not found")
        return
    valid_json_path = os.path.join(clip_root, json_file_name)
    valid_json = json.load(open(valid_json_path, 'r'))
    valid_idexs = sorted(list(valid_json.keys()))

    camera_masks = dict()
    for camera_name in camera_info_json.keys():
        if not os.path.exists(os.path.join(clip_root, camera_name + "_masks")):
            print(camera_name + "'s masks dir not found")
            return
        img_filenames = []

        for file in os.listdir(os.path.join(clip_root, camera_name + "_masks")):
            if not file.endswith('.png'): continue
            img_filenames.append(file)
        if not len(valid_idexs) == len(img_filenames):
            if len(valid_idexs) == len(img_filenames)+1:
                img_filenames = ["00001.png"]+img_filenames
            else:
                print(len(valid_idexs), len(img_filenames))
                return
        camera_masks[camera_name] = img_filenames
    print("valid clip and masks")

    for i, idex in enumerate(valid_idexs):
        print("processing frame: ", idex)
        pcds = []
        pcd_all = o3d.geometry.PointCloud()
        for camera_name in camera_masks.keys():
            img = cv2.imread(os.path.join(root_path, camera_name, "color", valid_idexs[i] + '.png'))
            depth = cv2.imread(os.path.join(root_path, camera_name, "depth", valid_idexs[i] + '.png'), cv2.IMREAD_UNCHANGED)
            mask = np.asarray(Image.open(os.path.join(clip_root, camera_name + "_masks", camera_masks[camera_name][i])), dtype=np.int32)
            mask = mask == 1
            depth = np.where(mask, depth, 0)
            Camera_Tranform = np.eye(4)
            Camera_Tranform[:3, :3] = np.array(camera_info_json[camera_name]["extrinsic"]["R"])
            Camera_Tranform[:3, 3] = np.array(camera_info_json[camera_name]["extrinsic"]["T"])
            pc = PointCloudHelper.rgbd2pc(img,
                                          depth,
                                          (3840 // 4, 2160 // 4, camera_info_json[camera_name]["K"]),
                                          transform=Camera_Tranform,
                                          enable_denoise=False)

            pcds.append(pc)
            pcd_all += pc
        cl, ind = pcd_all.remove_statistical_outlier(nb_neighbors=50,
                                                            std_ratio=2.0)
        output = pcd_all.select_by_index(ind)
        #o3d.visualization.draw_geometries([output])
        #display_inlier_outlier(pcd_all, ind)

        o3d.io.write_point_cloud(os.path.join(clip_root, "pcds", valid_idexs[i] + '.ply'), output)
        #break


if __name__ == "__main__":
    imgs2video(r"\\tsclient\C\Users\robotflow\Desktop\SAM\data0916\dress-long-sleeve_013_20230731_142555")
    #mask2pc(r"C:\Users\robotflow\Desktop\SAM\data0916\dress-long-sleeve_007_20230731_143236\SaveClip_156_702")

