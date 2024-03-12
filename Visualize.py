import os
import json
import cv2
import numpy as np
import open3d as o3d

from point import PointCloudHelper


class Visualizer:
    def __init__(self, path):
        self.path = path
        self.cameras_info = json.load(open(os.path.join(self.path, "cameras_info.jsonl"), "r"))
        self.cameras_name = self.cameras_info.keys()

        for name in self.cameras_name:
            self.cameras_info[name]["K"][0][0] = self.cameras_info[name]["fxy"][0]
            self.cameras_info[name]["K"][1][1] = self.cameras_info[name]["fxy"][1]
            self.cameras_info[name]["K"][0][2] = self.cameras_info[name]["cxy"][0]
            self.cameras_info[name]["K"][1][2] = self.cameras_info[name]["cxy"][1]

        self.show_pcd()

    def show_pcd(self):
        index = 3 * 100
        geoms = []
        for name in self.cameras_name:
            rgb = cv2.imread(os.path.join(self.path, name, "color", f"{index}.png"))
            depth = cv2.imread(os.path.join(self.path, name, "depth", f"{index}.png"), cv2.IMREAD_UNCHANGED)
            Camera_Tranform = np.eye(4)
            Camera_Tranform[:3, :3] = np.array(self.cameras_info[name]["extrinsic"]["R"])
            Camera_Tranform[:3, 3] = np.array(self.cameras_info[name]["extrinsic"]["T"])
            pc = PointCloudHelper.rgbd2pc(rgb,
                                          depth,
                                          (3840 // 4, 2160 // 4, self.cameras_info[name]["K"]),
                                          transform=Camera_Tranform,
                                          enable_denoise=False)
            geoms.append(pc)
        o3d.visualization.draw_geometries(geoms)

        self.pivot= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.pivot.transform(np.linalg.inv(Camera_Tranform))

if __name__ == '__main__':
    v = Visualizer(r"E:\Processed\pants-long\pants-long_036_20230727_151338")
