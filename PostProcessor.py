import os
import json
import threading

from tqdm import tqdm
import cv2
import jsonlines
import numpy as np
import transforms3d as t3d
from pupil_apriltags import Detector

from StaticPos import Apriltag
from point import PointCloudHelper
from thirdparty import pykinect
import open3d as o3d

class PostProcessor:
    def __init__(self, dataset_path, target_path):
        if not os.path.exists(dataset_path):
            print("dataset_path not exists")
            raise NotADirectoryError

        self.dataset_path = dataset_path
        self.main_camera_name = "000673513312"
        self.sub_camera_names = ["000700713312", "000729313312"]
        self.apriltag_ids = [11, 16]
        self.mapping = {self.apriltag_ids[0]: "left_hand", self.apriltag_ids[1]: "right_hand"}
        self.apriltag = Apriltag()
        self.downsample_ratio = 4
        self.time_downsample = 3

        # if not os.path.exists(os.path.join(dataset_path, "PostProcessDebug")):
        #     os.mkdir(os.path.join(dataset_path, "PostProcessDebug"))
        # self.root_dir = os.path.join(dataset_path, "PostProcessDebug")

        if not os.path.exists(os.path.join(target_path, os.path.basename(dataset_path))):
            os.mkdir(os.path.join(target_path, os.path.basename(dataset_path)))
        self.root_dir = os.path.join(target_path, os.path.basename(dataset_path))

        self.load_camera_info()

        camera_names = [self.main_camera_name] + self.sub_camera_names
        self.each_pic_res = dict()
        # for _ in range(2000):
        #     self.each_pic_res.append(dict())
        #     for name in camera_names:
        #         self.each_pic_res[-1][name] = dict()

    def load_camera_info(self):
        if not os.path.exists(os.path.join(self.dataset_path, "kinect/calibration.json")):
            print("calibration.json not exists")
            raise FileNotFoundError

        calibration = json.load(open(os.path.join(self.dataset_path, "kinect/calibration.json")))
        cameras = calibration["cameras"]
        self.camera_poses = calibration["camera_poses"]
        self.cameras = dict()
        for name in [self.main_camera_name] + self.sub_camera_names:
            camera = dict()
            camera["K"] = cameras[name]["K"]
            camera["dist"] = cameras[name]["dist"]
            camera["fxy"] = [cameras[name]["K"][0][0] / self.downsample_ratio, cameras[name]["K"][1][1] / self.downsample_ratio]
            camera["cxy"] = [cameras[name]["K"][0][2] / self.downsample_ratio, cameras[name]["K"][1][2] / self.downsample_ratio]
            camera["extrinsic"] = dict()
            if name in self.sub_camera_names:
                _R = np.array(self.camera_poses[name + "_to_" + self.main_camera_name]["R"]).T
                _T = np.array(self.camera_poses[name + "_to_" + self.main_camera_name]["T"]).reshape(3, 1)
                camera["extrinsic"]["T"] = (_R @ - _T).flatten().tolist()
                camera["extrinsic"]["Q"] = t3d.quaternions.mat2quat(_R).tolist()
                camera["extrinsic"]["R"] = _R.tolist()
            elif name == self.main_camera_name:
                _R = np.array(self.camera_poses[self.main_camera_name]["R"]).T
                _T = np.array(self.camera_poses[self.main_camera_name]["T"]).reshape(3, 1)
            else:
                raise ValueError
            camera["extrinsic"]["T"] = (_R @ - _T).flatten().tolist()
            camera["extrinsic"]["Q"] = t3d.quaternions.mat2quat(_R).tolist()
            camera["extrinsic"]["R"] = _R.tolist()
            self.cameras[name] = camera

        self.realworld_rot_trans = calibration["realworld_rot_trans"]

        with jsonlines.open(os.path.join(self.root_dir, "cameras_info.jsonl"), "w") as fc:
            fc.write(self.cameras)
        fc.close()

        with jsonlines.open(os.path.join(self.root_dir, "realworld_rot_trans.jsonl"), "w") as fc:
            fc.write(self.realworld_rot_trans)
        fc.close()

    def process_all(self):
        if os.path.exists(os.path.join(self.root_dir, "res.jsonl")):
            self.each_pic_res = json.load(open(os.path.join(self.root_dir, "res.jsonl"), "r"))
            return

        camera_names = [self.main_camera_name] + self.sub_camera_names
        pic_path_list = np.concatenate([np.sort(np.array([int(filename.replace('.jpg', ''))
                                                          for filename in os.listdir(os.path.join(self.dataset_path, "kinect/" + camera_name + "/color"))], dtype=int)).reshape(1, -1)
                                        for camera_name in camera_names], axis=0).T
        pic_path_list = [(*item, index) for index, item in enumerate(pic_path_list)]
        pbar = tqdm(pic_path_list)
        Threads = []
        for pic_path_pair in pbar:
            pbar.set_description("processing{:>13}".format(f"{pic_path_pair[0]}.jpg"))
            for i, item in enumerate(pic_path_pair[:-1]):
                t = threading.Thread(target=self.detect_one, args=(camera_names[i],
                                                                   os.path.join(self.dataset_path, "kinect/" + camera_names[i] + "/color", f"{item}.jpg"),
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
        f.close()

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
        hand_pos = {"T": [], "Q": [], "R": []}
        pic_res["left_hand"] = hand_pos
        pic_res["right_hand"] = hand_pos
        mapping = {self.apriltag_ids[0]: "left_hand", self.apriltag_ids[1]: "right_hand"}
        '''
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
        '''

        frame_raw = cv2.imread(pic_path)
        frame = cv2.undistort(frame_raw, self.cameras[camera_name]["K"], self.cameras[camera_name]["dist"])
        img_size = frame.shape[:2]
        frame_low = cv2.resize(frame, (img_size[1] // 2, img_size[0] // 2))
        cv2.imwrite(os.path.join(self.root_dir, camera_name, "color", f"{index}.png"), frame_low)

        depth_raw = cv2.imread(pic_path.replace("color", "depth").replace("jpg", "png"), cv2.IMREAD_UNCHANGED)
        depth = cv2.undistort(depth_raw, self.cameras[camera_name]["K"], self.cameras[camera_name]["dist"])
        depth_low = cv2.resize(depth, (img_size[1] // 2, img_size[0] // 2), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(self.root_dir, camera_name, "depth", f"{index}.png"), depth_low)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --apriltag--
        at_detector = Detector(families='tag25h9')
        tags = at_detector.detect(gray)

        '''
        for tag in tags:
            cv2.circle(frame, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2)  # left-top
            cv2.circle(frame, tuple(tag.corners[1].astype(int)), 4, (0, 255, 0), 2)  # right-top
            cv2.circle(frame, tuple(tag.corners[2].astype(int)), 4, (0, 0, 255), 2)  # right-bottom
            cv2.circle(frame, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2)  # left-bottom
        cv2.imwrite(os.path.join(self.root_dir, camera_name, "apriltag", os.path.basename(pic_path)), frame)
        '''

        for tag in tags:
            if tag.tag_id in self.apriltag_ids:
                res = cv2.solvePnP(self.apriltag.grids.reshape(-1, 3),
                                   np.array(tag.corners).reshape(-1, 2),
                                   self.cameras[camera_name]["K"],
                                   self.cameras[camera_name]["dist"][0])
                if res[0]:
                    _R = self.cameras[camera_name]["extrinsic"]["R"] @ np.matrix(cv2.Rodrigues(res[1])[0])
                    _T = np.array(res[2]).reshape(3, 1)

                    pic_res[mapping[tag.tag_id]]["T"] = (self.cameras[camera_name]["extrinsic"]["R"] @ _T + self.cameras[camera_name]["extrinsic"]["T"]).flatten().tolist()
                    pic_res[mapping[tag.tag_id]]["Q"] = t3d.quaternions.mat2quat(_R).tolist()
                    pic_res[mapping[tag.tag_id]]["R"] = _R.tolist()
        # --apriltag--

        self.each_pic_res[index][camera_name] = pic_res
        return pic_res

    def detect_one(self, camera_name: str, rgb_img, depth_img, index):
        if index % self.time_downsample != 0:
            return None

        if not os.path.exists(os.path.join(self.root_dir, camera_name)):
            os.mkdir(os.path.join(self.root_dir, camera_name))
        if not os.path.exists(os.path.join(self.root_dir, camera_name, "color")):
            os.mkdir(os.path.join(self.root_dir, camera_name, "color"))
        if not os.path.exists(os.path.join(self.root_dir, camera_name, "depth")):
            os.mkdir(os.path.join(self.root_dir, camera_name, "depth"))

        pic_res = dict()
        hand_pos_left = {"T": [], "Q": [], "R": []}
        hand_pos_right = {"T": [], "Q": [], "R": []}
        pic_res["left_hand"] = hand_pos_left
        pic_res["right_hand"] = hand_pos_right

        camera_K = np.array(self.cameras[camera_name]["K"])
        camera_dist = np.array(self.cameras[camera_name]["dist"])

        img_size = rgb_img.shape[:2]
        frame = rgb_img  # cv2.undistort(rgb_img, camera_K, camera_dist)
        frame_undistort = cv2.undistort(frame, camera_K, camera_dist)
        frame_low = cv2.resize(frame_undistort, (img_size[1] // self.downsample_ratio, img_size[0] // self.downsample_ratio))
        cv2.imwrite(os.path.join(self.root_dir, camera_name, "color", f"{index}.png"), frame_low)

        depth = depth_img  # cv2.undistort(depth_img, camera_K, camera_dist)
        depth_undistort = cv2.undistort(depth, camera_K, camera_dist)
        depth_low = cv2.resize(depth_undistort, (img_size[1] // self.downsample_ratio, img_size[0] // self.downsample_ratio), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(self.root_dir, camera_name, "depth", f"{index}.png"), depth_low)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --apriltag--
        at_detector = Detector(families='tag25h9')
        tags = at_detector.detect(gray)

        for tag in tags:
            if tag.tag_id in self.apriltag_ids:
                res = cv2.solvePnP(self.apriltag.grids.reshape(-1, 3),
                                   np.array(tag.corners).reshape(-1, 2),
                                   camera_K,
                                   camera_dist)
                if res[0]:
                    _R = np.array(self.cameras[camera_name]["extrinsic"]["R"]) @ np.matrix(cv2.Rodrigues(res[1])[0])
                    _T = np.array(res[2]).reshape(3, 1)

                    pic_res[self.mapping[tag.tag_id]]["T"] = (np.array(self.cameras[camera_name]["extrinsic"]["R"])
                                                              @ _T + np.array(self.cameras[camera_name]["extrinsic"]["T"]).reshape(3, 1)).flatten().tolist()
                    pic_res[self.mapping[tag.tag_id]]["Q"] = t3d.quaternions.mat2quat(_R).tolist()
                    pic_res[self.mapping[tag.tag_id]]["R"] = _R.tolist()

        # --apriltag--

        lock = threading.RLock()
        lock.acquire()
        if index not in self.each_pic_res.keys():
            self.each_pic_res[index] = dict()

        self.each_pic_res[index][camera_name] = pic_res
        lock.release()
        return pic_res

    def save(self):
        with open(os.path.join(self.root_dir, "res.json"), "w") as f:
            json.dump(self.each_pic_res, f, indent=4)


if __name__ == '__main__':
    PP = PostProcessor(dataset_path=r"E:\kva-system\azure_kinect_data\dress-long-sleeve_059_20230723_170240")
