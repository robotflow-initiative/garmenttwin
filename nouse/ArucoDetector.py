import os.path
import cv2
import cv2.aruco as aruco
from azure_kinect_apiserver.common import AzureKinectDataset
import numpy as np
from StaticPos import Board
import json
import tqdm
from typing import *
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from TrackerDataset import TrackerDataset, HandAruco


class ArucoResult:
    def __init__(self, left_hand_aruco_id: int = 0, right_hand_aruco_id: int = 1):
        self.ts = 0
        self.basename = 0

        self.board_pnp = False
        self.rvec_from_board_pnp = np.zeros(3).tolist()
        self.tvec_from_board_pnp = np.zeros(3).tolist()
        self.camera_in_board_coord_position = np.zeros(3).tolist()

        self.hand_corners = dict()

        self.hand_obj_corners = dict()  # vive tracker 坐标系下
        self.hand_pnp = dict()
        self.rvec_from_hand_pnp = dict()
        self.tvec_from_hand_pnp = dict()
        self.hand_in_board_coord_position = dict()

        self.left_hand_aruco_id = left_hand_aruco_id
        self.right_hand_aruco_id = right_hand_aruco_id

    def add_hand_corners(self, aruco_id: np.ndarray, corner: np.ndarray):
        # aruco_id.shape: (N, 1)
        # corner.shape: (N, 4, 2)
        aruco_id = aruco_id.reshape(-1, 1)
        corner = corner.reshape(-1, 4, 2)
        assert aruco_id.shape[0] == corner.shape[0]
        for i in range(aruco_id.shape[0]):
            if aruco_id[i][0] == self.left_hand_aruco_id or aruco_id[i][0] == self.right_hand_aruco_id:
                self.hand_corners[aruco_id[i][0]] = corner[i]


class ArucoDetector:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.board = Board()
        self.tracker_data = TrackerDataset(r"C:\Users\robotflow\Desktop\kva-system\tracker_data\test22_20230708_201421.jsonl")

        self.aruco_id2hand = {0: HandAruco("LHR-6A736404"), 1: HandAruco("LHR-656F5409", exchange=[1, 2, 3, 0])}

        self.each_pic_res = dict()

        self.aruco_dict_board = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
        self.aruco_dict_hand = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

        if not os.path.exists(os.path.join(dataset_path, "PostProcessDebug")):
            os.mkdir(os.path.join(dataset_path, "PostProcessDebug"))
        self.root_dir = os.path.join(dataset_path, "PostProcessDebug")

        if not os.path.exists(dataset_path):
            print("dataset_path not exists")
            raise NotADirectoryError
        if not os.path.exists(os.path.join(dataset_path, "kinect/calibration.json")):
            print("calibration.json not exists")
            raise FileNotFoundError
        if not os.path.exists(os.path.join(dataset_path, "kinect/meta.json")):
            print("meta.json not exists")
            raise FileNotFoundError

        cameras = json.load(open(os.path.join(dataset_path, "kinect/calibration.json")))["cameras"]
        self.camera_poses = json.load(open(os.path.join(dataset_path, "kinect/calibration.json")))["camera_poses"]

        self.cameras_names = list(cameras.keys())
        self.cameras = dict()
        for name in self.cameras_names:
            camera = dict()
            camera["K"] = np.array(cameras[name]["K"])
            camera["dist"] = np.array(cameras[name]["dist"]).flatten()
            self.cameras[name] = camera
            self.each_pic_res[name] = dict()

        self.system_timestamp_offset = json.load(open(os.path.join(dataset_path, "kinect/meta.json")))["system_timestamp_offset"]

    def run_one_pic(self, camera_name: str, pic_path: str):
        pic_res = dict()
        pic_res["basename"] = int(os.path.basename(pic_path).split(".")[0])
        pic_res["ts"] = int(os.path.basename(pic_path).split(".")[0]) * 1e-6 + self.system_timestamp_offset + 28800

        # --board--
        frame = cv2.imread(pic_path)
        frame = cv2.undistort(frame, self.cameras[camera_name]["K"], self.cameras[camera_name]["dist"])
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

        pic_res["camera=(R_matrix)@board+(T_vec)"] = (np.matrix(cv2.Rodrigues(res[1])[0]), np.array(res[2]).reshape(3, 1))
        position = pic_res["camera=(R_matrix)@board+(T_vec)"][0].T @ (-pic_res["camera=(R_matrix)@board+(T_vec)"][1])
        pic_res["camera_in_board_coord_position"] = position
        # --board--

        # --hand--
        corners_hand, ids_hand, rejectedImgPoints_hand = aruco.detectMarkers(gray, self.aruco_dict_hand, parameters=parameters)
        aruco.drawDetectedMarkers(frame, corners_hand, ids_hand)

        # --save--
        if not os.path.exists(os.path.join(self.root_dir, camera_name)):
            os.mkdir(os.path.join(self.root_dir, camera_name))
        cv2.imwrite(os.path.join(self.root_dir, camera_name, os.path.basename(pic_path)), frame)
        # --save--

        ids_hand = np.array(ids_hand).flatten()
        corners_hand = np.array(corners_hand).reshape(-1, 4, 2)
        ids_hand_mask = np.isin(ids_hand, list(self.aruco_id2hand.keys()))
        ids_hand = ids_hand[ids_hand_mask]
        corners_hand = corners_hand[ids_hand_mask]

        pic_res["hand_result"] = dict()
        print(ids_hand)
        for index, aruco_id in enumerate(ids_hand):
            hand_aruco = self.aruco_id2hand[aruco_id]

            pack, _ = self.tracker_data[(pic_res["ts"] * 10 ** 9, hand_aruco, "linear")]
            if pack is None:
                print(_)
                continue
            hand_obj_corners = pack["pts"].reshape(-1, 3)

            hres = cv2.solvePnP(hand_obj_corners[:4],
                                corners_hand[index].reshape(4, 2),
                                self.cameras[camera_name]["K"],
                                self.cameras[camera_name]["dist"][0],
                                flags=cv2.SOLVEPNP_P3P)
            print(hres[0])

            pic_res["hand_result"][aruco_id] = dict()
            pic_res["hand_result"][aruco_id]["vive_tracker_pts"] = pack["pts"]
            pic_res["hand_result"][aruco_id]["camera=(R_matrix)@vive+(T_vec)"] = (np.matrix(cv2.Rodrigues(hres[1])[0]), np.array(hres[2]).reshape(3, 1))
            h2c = pic_res["camera=(R_matrix)@board+(T_vec)"][0].T @ pic_res["hand_result"][aruco_id]["camera=(R_matrix)@vive+(T_vec)"][1]
            pic_res["hand_result"][aruco_id]["vive_in_board_coord_position"] = position + h2c
        # --hand--

        self.each_pic_res[camera_name][pic_res["basename"]] = pic_res
        return pic_res

    def test_camera_move(self, camera_name):
        if not os.path.exists(os.path.join(self.dataset_path, "kinect/" + camera_name + "/color")):
            print("'kinect/" + camera_name + "/color' not exists")
            raise NotADirectoryError

        pbar = tqdm.tqdm(os.listdir(os.path.join(self.dataset_path, "kinect/" + camera_name + "/color")))
        Threads = []
        for pic_path in pbar:
            pbar.set_description("processing{:>13}".format(pic_path))
            t = threading.Thread(target=self.run_one_pic, args=(camera_name, os.path.join(self.dataset_path, "kinect/" + camera_name + "/color", pic_path)))
            t.start()
            Threads.append(t)
        for t in Threads:
            t.join()

        # for pic_path in pbar:
        #     pbar.set_description("processing{:>13}".format(pic_path))
        #     self.run_one_pic(camera_name, os.path.join(self.dataset_path, "kinect/" + camera_name + "/color", pic_path), camera_poss)

        np.save(os.path.join(self.root_dir, camera_name + "_camera_poss.npy"), self.each_pic_res[camera_name])
        return self.each_pic_res[camera_name]

    def test_all_camera(self):
        fig = plt.figure()

        ax = Axes3D(fig)
        fig.add_axes(ax)

        ax.scatter(self.board.aruco_grids[..., 0].flat, self.board.aruco_grids[..., 1].flat, self.board.aruco_grids[..., 2].flat, c="r")

        res = dict()
        for camera_name in self.cameras_names:
            # if camera_name != "000729313312":
            #     continue
            if not os.path.exists(os.path.join(self.root_dir, camera_name + "_camera_poss.npy")):
                res[camera_name] = self.test_camera_move(camera_name)["tvec"]
                ax.scatter(res[camera_name][..., 0], res[camera_name][..., 1], res[camera_name][..., 2], c="b")
            else:
                self.each_pic_res[camera_name] = np.load(camera_name + "_camera_poss.npy", allow_pickle=True).item()
                # print(res[camera_name]["tvec"])
                ax.scatter(res[camera_name]["tvec"][..., 0], res[camera_name]["tvec"][..., 1], res[camera_name]["tvec"][..., 2], c="b")

        range = 2.5
        ax.set_xlim3d(-1, -1 + range)
        ax.set_ylim3d(1 - range, 1)
        ax.set_zlim3d(-0.5, -0.1 + range)
        plt.show()


if __name__ == '__main__':
    a = ArucoDetector(r"C:\Users\robotflow\Desktop\kva-system\azure_kinect_data\test22_20230708_201421")
    # 000673513312
    # 000700713312

    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter(a.board.aruco_grids[..., 0].flat, a.board.aruco_grids[..., 1].flat, a.board.aruco_grids[..., 2].flat, c="r")

    pic_res = a.run_one_pic("000700713312", r"C:\Users\robotflow\Desktop\kva-system\azure_kinect_data\test22_20230708_201421\kinect\000700713312\color\968500.jpg")
    aruco_id = 1
    cibp = np.array(pic_res["camera_in_board_coord_position"])
    vibp = np.array(pic_res["hand_result"][aruco_id]["vive_in_board_coord_position"])
    ax.scatter(cibp[0], cibp[1], cibp[2], c="g")
    ax.scatter(vibp[0], vibp[1], vibp[2], c="y")
    plot_res = pic_res["camera=(R_matrix)@board+(T_vec)"][0].T @ \
               pic_res["hand_result"][aruco_id]["camera=(R_matrix)@vive+(T_vec)"][0] @ \
               pic_res["hand_result"][aruco_id]["vive_tracker_pts"] + vibp
    ax.scatter(plot_res[0, :], plot_res[1, :], plot_res[2, :], c="b")

    # pic_res = a.run_one_pic("000673513312", r"C:\Users\robotflow\Desktop\kva-system\azure_kinect_data\test22_20230708_201421\kinect\000673513312\color\967133.jpg")
    # aruco_id = 0
    # cibp = np.array(pic_res["camera_in_board_coord_position"])
    # vibp = np.array(pic_res["hand_result"][aruco_id]["vive_in_board_coord_position"])
    # ax.scatter(cibp[0], cibp[1], cibp[2], c="g")
    # ax.scatter(vibp[0], vibp[1], vibp[2], c="y")
    # plot_res = pic_res["camera=(R_matrix)@board+(T_vec)"][0].T @ \
    #            pic_res["hand_result"][aruco_id]["camera=(R_matrix)@vive+(T_vec)"][0] @ \
    #            pic_res["hand_result"][aruco_id]["vive_tracker_pts"] + vibp
    # ax.scatter(plot_res[0, :], plot_res[1, :], plot_res[2, :], c="b")

    range = 2.5
    ax.set_xlim3d(-1, -1 + range)
    ax.set_ylim3d(1 - range, 1)
    ax.set_zlim3d(-0.5, -0.1 + range)
    plt.show()
