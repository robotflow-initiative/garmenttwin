import math
import threading
import time

import cv2
import cv2.aruco as aruco
from pupil_apriltags import Detector
import os
import json
import jsonlines
import numpy as np
from typing import List

from tqdm import tqdm

from StaticPos import Board, Apriltag

import open3d as o3d
import transforms3d as t3d

import keyboard

from point import PointCloudHelper


def add_geometry(pic_res, geoms: List[o3d.geometry.Geometry]):
    camera = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    T_matrix = np.eye(4)
    T_matrix[:3, :3] = pic_res["camera=(R_matrix)@board+(T_vec)"][0].T
    T_matrix[:3, 3] = (pic_res["camera=(R_matrix)@board+(T_vec)"][0].T @ -pic_res["camera=(R_matrix)@board+(T_vec)"][1]).flatten()
    camera.transform(T_matrix)
    geoms.append(camera)

    for R, T in pic_res["camera=(R_matrix)@hands+(T_vec)"].values():
        R_total = pic_res["camera=(R_matrix)@board+(T_vec)"][0].T @ R
        T_total = pic_res["camera=(R_matrix)@board+(T_vec)"][0].T @ (T - pic_res["camera=(R_matrix)@board+(T_vec)"][1])

        T_matrix = np.eye(4)
        T_matrix[:3, :3] = R_total
        T_matrix[:3, 3] = T_total.flatten()

        hand = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        hand.transform(T_matrix)

        geoms.append(hand)
        geoms.append(TD.apriltag.get_cloud_points(R_total, T_total))


class TagDetector:
    def __init__(self, dataset_path: str, camera_names: List[str] = None, apriltag_ids: List[int] = None):
        if camera_names is None:
            camera_names = ["000700713312", "000729313312"]
        self.cameras_names = camera_names
        if apriltag_ids is None:
            apriltag_ids = [11, 16]
        self.apriltag_ids = apriltag_ids

        self.dataset_path = dataset_path
        if not os.path.exists(dataset_path):
            print("dataset_path not exists")
            raise NotADirectoryError

        if not os.path.exists(os.path.join(dataset_path, "kinect/calibration.json")):
            print("calibration.json not exists")
            raise FileNotFoundError
        cameras = json.load(open(os.path.join(dataset_path, "kinect/calibration.json")))["cameras"]
        self.camera_poses = json.load(open(os.path.join(dataset_path, "kinect/calibration.json")))["camera_poses"]
        self.cameras = dict()
        for name in self.cameras_names:
            camera = dict()
            camera["K"] = np.array(cameras[name]["K"])
            camera["dist"] = np.array(cameras[name]["dist"]).flatten()
            self.cameras[name] = camera

        self.aruco_dict_board = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
        self.board = Board()

        self.apriltag = Apriltag()

        if not os.path.exists(os.path.join(dataset_path, "PostProcessDebug")):
            os.mkdir(os.path.join(dataset_path, "PostProcessDebug"))
        self.root_dir = os.path.join(dataset_path, "PostProcessDebug")

        self.each_pic_res = []
        self.hand_pos = []
        for _ in os.listdir(os.path.join(dataset_path, "kinect/" + self.cameras_names[0] + "/color")):
            self.each_pic_res.append(dict())
            self.hand_pos.append(dict())

    def detect_one(self, camera_name: str, pic_path: str, index: int):
        if not os.path.exists(os.path.join(self.root_dir, camera_name)):
            os.mkdir(os.path.join(self.root_dir, camera_name))
        if not os.path.exists(os.path.join(self.root_dir, camera_name, "board")):
            os.mkdir(os.path.join(self.root_dir, camera_name, "board"))
        if not os.path.exists(os.path.join(self.root_dir, camera_name, "apriltag")):
            os.mkdir(os.path.join(self.root_dir, camera_name, "apriltag"))
        if not os.path.exists(os.path.join(self.root_dir, camera_name, "color")):
            os.mkdir(os.path.join(self.root_dir, camera_name, "color"))
        if not os.path.exists(os.path.join(self.root_dir, camera_name, "depth")):
            os.mkdir(os.path.join(self.root_dir, camera_name, "depth"))

        pic_res = dict()

        # --board--
        frame_raw = cv2.imread(pic_path)
        frame = cv2.undistort(frame_raw, self.cameras[camera_name]["K"], self.cameras[camera_name]["dist"])
        cv2.imwrite(os.path.join(self.root_dir, camera_name, "color", f"{index}.png"), frame)
        depth_raw = cv2.imread(pic_path.replace("color", "depth").replace("jpg", "png"), cv2.IMREAD_UNCHANGED)
        depth = cv2.undistort(depth_raw, self.cameras[camera_name]["K"], self.cameras[camera_name]["dist"])
        depth = cv2.blur(depth, (7, 7))
        cv2.imwrite(os.path.join(self.root_dir, camera_name, "depth", f"{index}.png"), depth)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict_board, parameters=parameters)
        aruco.drawDetectedMarkers(frame, corners, ids)
        ids = np.array(ids).flatten()
        corners = np.array(corners).reshape(-1, 4, 2)
        ids_mask = np.isin(ids, self.board.ids)
        obj_pos, _ = self.board.get_grids(ids[ids_mask])
        # debug
        if obj_pos is None:
            print("wrong with", _)
            cv2.imwrite(os.path.join(self.root_dir, "aruco5x5.jpg"), frame)
            cv2.imshow("frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        res = cv2.solvePnP(obj_pos.reshape(-1, 3),
                           corners[ids_mask].reshape(-1, 2),
                           self.cameras[camera_name]["K"],
                           self.cameras[camera_name]["dist"][0])

        pic_res["camera=(R_matrix)@board+(T_vec)"] = (np.matrix(cv2.Rodrigues(res[1])[0]).tolist(), np.array(res[2]).reshape(3, 1).tolist())
        position = np.array(pic_res["camera=(R_matrix)@board+(T_vec)"][0]).T @ (-np.array(pic_res["camera=(R_matrix)@board+(T_vec)"][1]))
        pic_res["camera_in_board_coord_position"] = position.tolist()

        cv2.imwrite(os.path.join(self.root_dir, camera_name, "board", os.path.basename(pic_path)), frame)
        # --board--

        # --apriltag--
        frame = cv2.undistort(frame_raw, self.cameras[camera_name]["K"], self.cameras[camera_name]["dist"])
        at_detector = Detector(families='tag25h9')
        tags = at_detector.detect(gray)
        pic_res["camera=(R_matrix)@hands+(T_vec)"] = dict()
        # if len(list(tags)) < 2:
        #     print(index, "--------------------------")
        for tag in tags:
            cv2.circle(frame, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2)  # left-top
            cv2.circle(frame, tuple(tag.corners[1].astype(int)), 4, (0, 255, 0), 2)  # right-top
            cv2.circle(frame, tuple(tag.corners[2].astype(int)), 4, (0, 0, 255), 2)  # right-bottom
            cv2.circle(frame, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2)  # left-bottom

        for tag_id in self.apriltag_ids:
            if tag_id not in self.hand_pos[index].keys():
                self.hand_pos[index][tag_id] = []
        for tag in tags:
            if tag.tag_id in self.apriltag_ids:
                res = cv2.solvePnP(self.apriltag.grids.reshape(-1, 3),
                                   np.array(tag.corners).reshape(-1, 2),
                                   self.cameras[camera_name]["K"],
                                   self.cameras[camera_name]["dist"][0])
                pic_res["camera=(R_matrix)@hands+(T_vec)"][tag.tag_id] = (np.matrix(cv2.Rodrigues(res[1])[0]).tolist(), np.array(res[2]).reshape(3, 1).tolist())
                R_total = np.array(pic_res["camera=(R_matrix)@board+(T_vec)"][0]).T @ np.array(pic_res["camera=(R_matrix)@hands+(T_vec)"][tag.tag_id][0])
                T_total = np.array(pic_res["camera=(R_matrix)@board+(T_vec)"][0]).T @ (np.array(pic_res["camera=(R_matrix)@hands+(T_vec)"][tag.tag_id][1])
                                                                                       - np.array(pic_res["camera=(R_matrix)@board+(T_vec)"][1]))
                # pos = np.concatenate([T_total.reshape(-1, 1), t3d.quaternions.mat2quat(R_total).reshape(-1, 1)], axis=0).flatten().tolist()[0]
                pos = {"T": T_total.tolist(), "R": R_total.tolist()}
                self.hand_pos[index][tag.tag_id].append(pos)

        cv2.imwrite(os.path.join(self.root_dir, camera_name, "apriltag", os.path.basename(pic_path)), frame)
        # --apriltag--

        self.each_pic_res[index][camera_name] = pic_res
        return pic_res

    def detect_all(self):
        if os.path.exists(os.path.join(self.root_dir, "res.jsonl")):
            self.each_pic_res = json.load(open(os.path.join(self.root_dir, "res.jsonl"), "r"))
            # print(self.each_pic_res)
            return

        pic_path_list = np.concatenate([np.sort(np.array([int(filename.replace('.jpg', ''))
                                                          for filename in os.listdir(os.path.join(self.dataset_path, "kinect/" + camera_name + "/color"))], dtype=int)).reshape(1, -1)
                                        for camera_name in self.cameras_names], axis=0).T
        pic_path_list = [(*item, index) for index, item in enumerate(pic_path_list)]
        pbar = tqdm(pic_path_list)
        Threads = []
        for pic_path_pair in pbar:
            pbar.set_description("processing{:>13}".format(f"{pic_path_pair[0]}.jpg"))
            for i, item in enumerate(pic_path_pair[:-1]):
                t = threading.Thread(target=self.detect_one, args=(self.cameras_names[i],
                                                                   os.path.join(self.dataset_path, "kinect/" + self.cameras_names[i] + "/color", f"{item}.jpg"),
                                                                   pic_path_pair[-1]))
                t.start()
                Threads.append(t)
            for t in Threads:
                t.join()
                # self.detect_one(self.cameras_names[i],
                #                 os.path.join(self.dataset_path, "kinect/" + self.cameras_names[i] + "/color", f"{item}.jpg"),
                #                 pic_path_pair[-1])

        with jsonlines.open(os.path.join(self.root_dir, "res.jsonl"), "w") as f:
            f.write(self.each_pic_res)
        with jsonlines.open(os.path.join(self.root_dir, "hand_pos.jsonl"), "w") as f:
            f.write(self.hand_pos)

    def refine_camera_poses_diff(self):
        camera_T_diff = np.eye(4)
        diff_R = np.array(self.camera_poses[self.cameras_names[1] + "_to_" + self.cameras_names[0]]["R"])
        diff_T = np.array(self.camera_poses[self.cameras_names[1] + "_to_" + self.cameras_names[0]]["T"]).reshape(3, 1)
        camera_T_diff[:3, :3] = diff_R.T
        camera_T_diff[:3, 3] = (diff_R.T @ -diff_T).flatten()
        for i in range(len(self.each_pic_res)):
            color_undisort0 = cv2.imread(os.path.join(self.root_dir, self.cameras_names[0], "color", f"{i}.png"))
            depth_undisort0 = cv2.imread(os.path.join(self.root_dir, self.cameras_names[0], "depth", f"{i}.png"), cv2.IMREAD_UNCHANGED)
            color_undisort1 = cv2.imread(os.path.join(self.root_dir, self.cameras_names[1], "color", f"{i}.png"))
            depth_undisort1 = cv2.imread(os.path.join(self.root_dir, self.cameras_names[1], "depth", f"{i}.png"), cv2.IMREAD_UNCHANGED)
            pcd0 = PointCloudHelper.rgbd2pc(color_undisort0,
                                            depth_undisort0,
                                            (3840, 2160, self.cameras[self.cameras_names[0]]["K"]),
                                            depth_near=0,
                                            transform=np.eye(4),
                                            enable_denoise=False)

            pcd1 = PointCloudHelper.rgbd2pc(color_undisort1,
                                            depth_undisort1,
                                            (3840, 2160, self.cameras[self.cameras_names[1]]["K"]),
                                            depth_near=0,
                                            transform=np.eye(4),
                                            enable_denoise=False)

            o3d.visualization.draw_geometries([pcd1.transform(camera_T_diff)])

            res = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.1, camera_T_diff, o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation
            print(camera_T_diff)
            print(res)
            o3d.visualization.draw_geometries([pcd0, pcd1.transform(np.linalg.inv(camera_T_diff)).transform(res)])
            T_matrix = np.array(res)
            camera_T_diff = dict()
            camera_T_diff["T"] = T_matrix[:3, 3].flatten().tolist()
            camera_T_diff["Q"] = t3d.quaternions.mat2quat(T_matrix[:3, :3]).tolist()
            camera_T_diff["R"] = T_matrix[:3, :3].tolist()
            with jsonlines.open(os.path.join(self.root_dir, "camera_T_diff.jsonl"), "w") as fc:
                fc.write(camera_T_diff)
            return

    def filter(self):

        camera_T_diff = dict()  # json.load(open(os.path.join(self.root_dir, "camera_T_diff.jsonl"), "r"))
        diff_R = np.array(self.camera_poses[self.cameras_names[1] + "_to_" + self.cameras_names[0]]["R"])
        diff_T = np.array(self.camera_poses[self.cameras_names[1] + "_to_" + self.cameras_names[0]]["T"]).reshape(3, 1)
        camera_T_diff["T"] = (diff_R.T @ -diff_T).flatten().tolist()
        camera_T_diff["Q"] = t3d.quaternions.mat2quat(diff_R.T).tolist()
        camera_T_diff["E"] = (np.array(t3d.euler.quat2euler(np.array(camera_T_diff["Q"]), "syxz")) * 180 / np.pi).tolist()
        # [-unity.z -unity.x unity.y] "szxy"
        # [unity.y unity.x -unity.z] "syxz"
        camera_T_diff["R"] = diff_R.T.tolist()

        with jsonlines.open(os.path.join(self.root_dir, "camera_T_diff.jsonl"), "w") as fc:
            fc.write(camera_T_diff)
        # for i in range(len(self.each_pic_res)):
        #     t1 = np.array(self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@board+(T_vec)"][1]).flatten()
        #     t2 = np.array(self.each_pic_res[i][self.cameras_names[1]]["camera=(R_matrix)@board+(T_vec)"][1]).flatten()
        #     R1 = np.array(self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@board+(T_vec)"][0])
        #     R2 = np.array(self.each_pic_res[i][self.cameras_names[1]]["camera=(R_matrix)@board+(T_vec)"][0])
        #
        #     R = R1 @ R2.T
        #     t = t1 - R1 @ R2.T @ t2
        #     q = t3d.quaternions.mat2quat(R)
        #
        #     camera_T_diff.append({"T": t.tolist(), "Q": q.tolist(), "R": R.tolist()})

        mask2 = set()
        right_data = []
        for i in range(len(self.each_pic_res)):
            if str(self.apriltag_ids[1]) in self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@hands+(T_vec)"].keys():
                t = np.array(self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@hands+(T_vec)"][str(self.apriltag_ids[1])][1]).flatten()
                M = np.array(self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@hands+(T_vec)"][str(self.apriltag_ids[1])][0])
                q = t3d.quaternions.mat2quat(M)
            elif str(self.apriltag_ids[1]) in self.each_pic_res[i][self.cameras_names[1]]["camera=(R_matrix)@hands+(T_vec)"].keys():
                mask2.add(i)
                _t = np.array(self.each_pic_res[i][self.cameras_names[1]]["camera=(R_matrix)@hands+(T_vec)"][str(self.apriltag_ids[1])][1]).reshape(3, 1)
                _M = np.array(self.each_pic_res[i][self.cameras_names[1]]["camera=(R_matrix)@hands+(T_vec)"][str(self.apriltag_ids[1])][0])
                t = (np.array(camera_T_diff["R"]) @ _t + np.array(camera_T_diff["T"]).reshape(3, 1)).flatten()
                M = np.array(camera_T_diff["R"]) @ _M
                q = t3d.quaternions.mat2quat(M)
            else:
                mask2.add(i)
                t = np.zeros(3)
                M = np.zeros([3, 3])
                q = np.zeros(4)
            right_data.append({"T": t.tolist(), "Q": q.tolist(), "R": M.tolist()})
        for i, data in enumerate(right_data):
            if np.sum(data["T"]) < 1e-6:
                for j in range(i + 1, len(right_data)):
                    if np.sum(right_data[j]["T"]) > 1e-6:
                        right_data[i] = right_data[j]
                        break
        with jsonlines.open(os.path.join(self.root_dir, "right_data.jsonl"), "w") as f3:
            f3.write(right_data)

        left_data = []
        for i in range(len(self.each_pic_res)):
            if str(self.apriltag_ids[0]) in self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@hands+(T_vec)"].keys():
                t = np.array(self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@hands+(T_vec)"][str(self.apriltag_ids[0])][1]).flatten()
                M = np.array(self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@hands+(T_vec)"][str(self.apriltag_ids[0])][0])
                q = t3d.quaternions.mat2quat(M)
            elif str(self.apriltag_ids[0]) in self.each_pic_res[i][self.cameras_names[1]]["camera=(R_matrix)@hands+(T_vec)"].keys():
                mask2.add(i)
                _t = np.array(self.each_pic_res[i][self.cameras_names[1]]["camera=(R_matrix)@hands+(T_vec)"][str(self.apriltag_ids[0])][1]).reshape(3, 1)
                _M = np.array(self.each_pic_res[i][self.cameras_names[1]]["camera=(R_matrix)@hands+(T_vec)"][str(self.apriltag_ids[0])][0])
                t = (np.array(camera_T_diff["R"]) @ _t + np.array(camera_T_diff["T"]).reshape(3, 1)).flatten()
                M = np.array(camera_T_diff["R"]) @ _M
                q = t3d.quaternions.mat2quat(M)
            else:
                mask2.add(i)
                t = np.zeros(3)
                M = np.zeros([3, 3])
                q = np.zeros(4)
            left_data.append({"T": t.tolist(), "Q": q.tolist(), "R": M.tolist()})
        for i, data in enumerate(left_data):
            if np.sum(data["T"]) < 1e-6:
                for j in range(i + 1, len(left_data)):
                    if np.sum(left_data[j]["T"]) > 1e-6:
                        left_data[i] = left_data[j]
                        break
        with jsonlines.open(os.path.join(self.root_dir, "left_data.jsonl"), "w") as f4:
            f4.write(left_data)

        mask_acc = set()
        accelertion_thresh = 1
        for data_list in [left_data, right_data]:
            t_list = [data["T"] for data in data_list]
            t_list = [t_list[0]] + t_list + [t_list[-1]]
            for i, transform in enumerate(t_list):
                if i - 1 in mask_acc: continue
                if i == 0: continue
                if i == len(t_list) - 1: continue
                if i - 1 not in mask_acc:
                    a = np.sum(np.power(np.array(t_list[i + 1]) + np.array(t_list[i - 1]) - 2 * np.array(transform), 2)) / 0.033 ** 2
                    if a > accelertion_thresh:
                        mask_acc.add(i - 1)

        a_list = []
        import matplotlib.pyplot as plt
        mask_ang = set()
        angular_thresh = 100
        for data_list in [left_data, right_data]:
            R_list = [data["R"] for data in data_list]
            R_list = [R_list[0]] + R_list + [R_list[-1]]
            for i, R in enumerate(R_list):
                if i - 1 in mask_ang: continue
                if i == 0: continue
                if i == len(R_list) - 1: continue
                if i - 1 not in mask_acc:
                    r1 = np.array(R_list[i - 1])
                    r2 = np.array(R)
                    r3 = np.array(R_list[i + 1])

                    omega1 = math.acos(max(min(((r1.T @ r2).trace() - 1) / 2, 1), -1))
                    omega2 = math.acos(max(min(((r2.T @ r3).trace() - 1) / 2, 1), -1))
                    a = (omega2 - omega1) / 0.033 ** 2
                    a_list.append(a)
                    if a > angular_thresh:
                        mask_ang.add(i - 1)
            # plt.plot(range(len(a_list)), a_list)
            # plt.ylim([-100,100])
            # plt.show()

        mask_list = list(mask_acc.union(mask2).union(mask_ang))
        mask_list.sort()
        with jsonlines.open(os.path.join(self.root_dir, "index_mask.jsonl"), "w") as f:
            f.write(mask_list)

    def visualize_hand_pos(self, json_path: str):
        hand_pos = json.load(open(json_path, "r"))

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        geoms = []
        coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geoms.append(coordinate)
        geoms.append(self.board.cloud_points)

        [vis.add_geometry(source) for source in geoms]

        tracker_meshes = {
            apriltag_id: o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0., 0., 0.])) for apriltag_id in self.apriltag_ids
        }
        tracker_transforms = {
            apriltag_id: np.eye(4) for apriltag_id in self.apriltag_ids
        }

        [vis.add_geometry(source) for source in tracker_meshes.values()]

        for i, frame in enumerate(hand_pos):
            for apriltag in frame.keys():
                if len(frame[apriltag]) == 0:
                    continue
                possible_pos = frame[apriltag]
                t = np.mean([item["T"] for item in possible_pos], axis=0)
                R = possible_pos[0]["R"]
                Tmatix = np.eye(4)
                Tmatix[:3, :3] = R
                Tmatix[:3, 3] = t.flatten()

                apriltag = int(apriltag)
                # if apriltag is not 11: continue
                T_diff = Tmatix @ np.linalg.inv(tracker_transforms[apriltag])
                tracker_transforms[apriltag] = T_diff @ tracker_transforms[apriltag]
                tracker_meshes[apriltag].transform(T_diff)

                vis.update_geometry(tracker_meshes[apriltag])
            if not vis.poll_events():
                break
            vis.update_renderer()
            print(i)
            # keyboard.wait('space')

    def visualize_static(self, index: int, json_path: str, camera_name: str):
        geoms = []
        coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geoms.append(coordinate)
        geoms.append(self.board.cloud_points)

        Camera_Tranform = np.eye(4)
        Camera_Tranform[:3, :3] = np.array(self.each_pic_res[index][self.cameras_names[0]]["camera=(R_matrix)@board+(T_vec)"][0]).T
        Camera_Tranform[:3, 3] = (np.array(self.each_pic_res[index][self.cameras_names[0]]["camera=(R_matrix)@board+(T_vec)"][0]).T \
                                  @ -np.array(self.each_pic_res[index][self.cameras_names[0]]["camera=(R_matrix)@board+(T_vec)"][1]).reshape(3, 1)).flatten()

        pc = PointCloudHelper.rgbd2pc(cv2.imread(os.path.join(self.root_dir, self.cameras_names[1], "color", f"{index}.png")),
                                      cv2.imread(os.path.join(self.root_dir, self.cameras_names[1], "depth", f"{index}.png"), cv2.IMREAD_UNCHANGED),
                                      (3840, 2160, self.cameras[self.cameras_names[1]]["K"]),
                                      transform=Camera_Tranform,
                                      enable_denoise=False)
        pc2 = PointCloudHelper.rgbd2pc(cv2.imread(os.path.join(self.root_dir, camera_name, "color", f"{index}.png")),
                                       cv2.imread(os.path.join(self.root_dir, camera_name, "depth", f"{index}.png"), cv2.IMREAD_UNCHANGED),
                                       (3840, 2160, self.cameras[camera_name]["K"]),
                                       transform=Camera_Tranform,
                                       enable_denoise=False)

        geoms.append(pc)

        tracker_meshes = {
            apriltag_id: o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0., 0., 0.])) for apriltag_id in self.apriltag_ids
        }
        hand_pos = json.load(open(json_path, "r"))
        frame = hand_pos[index]

        for apriltag in frame.keys():
            if len(frame[apriltag]) == 0:
                continue
            possible_pos = frame[apriltag]
            t = np.mean([item["T"] for item in possible_pos], axis=0)
            R = possible_pos[0]["R"]
            Tmatix = np.eye(4)
            Tmatix[:3, :3] = R
            Tmatix[:3, 3] = t.flatten()

            apriltag = int(apriltag)
            # if apriltag is not 11: continue

            tracker_meshes[apriltag].transform(Tmatix)

        geoms.extend(tracker_meshes.values())

        o3d.visualization.draw_geometries(geoms)

    def visualize_after(self, json_path: str):
        hand_pos = json.load(open(json_path, "r"))

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        geoms = []
        coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geoms.append(coordinate)
        geoms.append(self.board.cloud_points)

        [vis.add_geometry(source) for source in geoms]

        tracker_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0., 0., 0.]))
        tracker_transform = np.eye(4)

        vis.add_geometry(tracker_mesh)

        for i, frame in enumerate(hand_pos):
            if i > 150: break
            t = np.array(frame["T"]).flatten()
            R = t3d.quaternions.quat2mat(np.array(frame["Q"]))
            Tmatix = np.eye(4)
            Tmatix[:3, :3] = R
            Tmatix[:3, 3] = t.flatten()

            t2 = np.array(self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@board+(T_vec)"][1]).flatten()
            r2 = np.array(self.each_pic_res[i][self.cameras_names[0]]["camera=(R_matrix)@board+(T_vec)"][0])
            Tmatix2 = np.eye(4)
            Tmatix2[:3, :3] = r2
            Tmatix2[:3, 3] = t2.flatten()

            Tmatix = Tmatix2.T @ Tmatix

            T_diff = Tmatix @ np.linalg.inv(tracker_transform)
            tracker_transform = T_diff @ tracker_transform
            tracker_mesh.transform(T_diff)

            vis.update_geometry(tracker_mesh)
            if not vis.poll_events():
                break
            vis.update_renderer()
            print(i)


if __name__ == '__main__':
    TD = TagDetector(dataset_path=r"C:\Users\robotflow\Desktop\azure-kinect-apiserver\azure_kinect_data\2023-7-21-board")

    TD.detect_all()
    TD.refine_camera_poses_diff()
    # TD.visualize_hand_pos(r"C:\Users\robotflow\Desktop\kva-system\azure_kinect_data\7-14-2\PostProcessDebug\hand_pos.jsonl")
    # TD.visualize_static(0 , r"C:\Users\robotflow\Desktop\azure-kinect-apiserver\azure_kinect_data\2023-7-21-board\PostProcessDebug\hand_pos.jsonl", "000700713312")#"000700713312""000729313312"
    # TD.filter()
    # TD.visualize_after(r"C:\Users\robotflow\Desktop\kva-system\azure_kinect_data\7-14-2\PostProcessDebug\right_data.jsonl")

    # pic_res = TD.detect_one(camera_name="000700713312", pic_path=r"C:\Users\robotflow\Desktop\kva-system\azure_kinect_data\7-14-2\kinect\000700713312\color\10411822.jpg", index=0)
    # geoms = []
    #
    # coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # geoms.append(coordinate)
    # geoms.append(TD.board.cloud_points)
    #
    # add_geometry(pic_res, geoms)
    #
    # o3d.visualization.draw_geometries(geoms)

    # rgbd = cv2.imread(r"C:\Users\robotflow\Desktop\kva-system\azure_kinect_data\7-14-2\PostProcessDebug\000700713312\depth\0.png", cv2.IMREAD_UNCHANGED)
    # print(rgbd.shape)
    # #rgbd = rgbd/10*255
    # cv2.imshow("rgbd", rgbd)
    # cv2.waitKey(0)
