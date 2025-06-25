import os
import threading
import time
from typing import Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from groundingdino.util.inference import annotate, load_image, load_model, predict
from mcp.server.fastmcp import FastMCP
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

robot_host = "127.0.0.1"
robot_port = "5000"
speed = 20
radius = 0
connect = 0
block = 1


class Camera:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Camera, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if Camera._initialized:
            print("[Warning] Camera already initialized. Skipping...")
            return

        try:
            self.connect_device = []
            self._init_camera()
            Camera._initialized = True
            self.first_call = True
        except Exception as e:
            Camera._initialized = False
            raise RuntimeError(f"[ERROR] Camera initialization failed: {e}")

    def _init_camera(self):
        ctx = rs.context()
        if not ctx.devices:
            raise RuntimeError("No RealSense device found.")

        for d in ctx.devices:
            dev_name = d.get_info(rs.camera_info.name)
            dev_sn = d.get_info(rs.camera_info.serial_number)
            print(f"[INFO] Found device: {dev_name} {dev_sn}")
            self.connect_device.append(dev_sn)

        if len(self.connect_device) != 1:
            raise RuntimeError(
                f"[ERROR] Expected 1 device, but found {len(self.connect_device)}"
            )

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.connect_device[0])
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = (
            self.profile.get_device().first_depth_sensor().get_depth_scale()
        )
        print("[INFO] Camera initialized successfully")

    def record_wrist_frame(self, output_dir: str = "./output") -> Tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        if self.first_call:
            for _ in range(45):
                self.pipeline.wait_for_frames()
            self.first_call = False
            print("[INFO] Warm-up frames completed")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to capture frames.")

        color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_image = (
            np.asanyarray(depth_frame.get_data(), dtype=np.float32)
            * self.depth_scale
            * 1000
        )

        color_path = os.path.join(output_dir, "wrist_obs.png")
        depth_path = os.path.join(output_dir, "wrist_obs_depth.npy")

        cv2.imwrite(color_path, color_image)
        np.save(depth_path, depth_image)

        print(f"[INFO] Saved color image: {color_path}")
        print(f"[INFO] Saved depth data: {depth_path}")
        return color_path, depth_path

    def shutdown(self):
        if hasattr(self, "pipeline"):
            self.pipeline.stop()
        Camera._initialized = False
        print("[INFO] Camera shut down")

    def __del__(self):
        self.shutdown()


class Dino:
    def __init__(
        self,
        config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        weights_path="GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device=None,
    ):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = load_model(config_path, weights_path)
        print(f"[INFO] DINO model loaded on {self.device}")

    def predict(self, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        image_source, image = load_image(image_path)
        image = image.to(self.device)
        boxes, logits, phrases = predict(
            self.model,
            image,
            text_prompt,
            box_threshold,
            text_threshold,
            device=self.device,
        )
        h, w, _ = image_source.shape
        boxes_point = boxes * torch.tensor([w, h, w, h])
        self.save_annotated_image(
            image_source, boxes, logits, phrases, "./output/annotated_image.png"
        )
        return boxes_point, logits

    def save_annotated_image(self, image_source, boxes, logits, phrases, output_path):
        annotated = annotate(image_source, boxes, logits, phrases)
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Annotated image saved: {output_path}")

    @staticmethod
    def uv_to_xyz(uv, depth_file, extrinsics):
        depth = np.load(depth_file)
        z = depth[uv[1], uv[0]] * 0.001
        fx, fy, cx, cy = 600.42, 600.68, 328.08, 238.68
        x = (uv[0] - cx) * z / fx
        y = (uv[1] - cy) * z / fy
        cam_point = np.array([x, y, z, 1.0]).reshape(4, 1)
        base_point = extrinsics @ cam_point
        return base_point[:3].flatten()

    @staticmethod
    def transform_camera2base(gripper_pose):
        R = np.array(
            [[-0.034, -0.999, 0], [0.999, -0.034, -0.006], [0.006, 0, 1]],
            dtype=np.float32,
        )
        t = np.array([[0.09], [-0.03], [0.02]], dtype=np.float32)

        def euler_to_rot(rx, ry, rz):
            Rx = np.array(
                [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
            )
            Ry = np.array(
                [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
            )
            Rz = np.array(
                [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
            )
            return Rz @ Ry @ Rx

        T = np.eye(4)
        T[:3, :3] = euler_to_rot(*gripper_pose[3:])
        T[:3, 3] = gripper_pose[:3]
        T_gripper = T

        T_cam = np.eye(4)
        T_cam[:3, :3] = R
        T_cam[:3, 3] = t.flatten()

        return T_gripper @ T_cam


class RealmanArm:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, host, port, speed, radius, connect, block):
        if self._initialized:
            return
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.robot.rm_create_robot_arm(host, port)
        self.v, self.r = speed, radius
        self.connect, self.block = connect, block
        print(f"[INFO] Robot handle: {self.handle.id}")
        self._initialized = True

    def observe(self):
        self.robot.rm_movej(
            [2, 84, -74, -12, -100, 4], self.v, self.r, self.connect, self.block
        )
        print("[INFO] Arm moved to observe pose")

    def grasp(self, pose):
        self.robot.rm_set_gripper_release(500, True, 10)
        self.robot.rm_movej_p(pose, self.v, self.r, self.connect, self.block)
        pose[0] += 0.025
        self.robot.rm_movej_p(pose, self.v, self.r, self.connect, self.block)
        self.robot.rm_set_gripper_position(5, False, 5)
        time.sleep(1)
        pose[0] -= 0.10
        flag = self.robot.rm_movej_p(pose, self.v, self.r, self.connect, self.block)
        print("[INFO] Arm grasp sequence complete")
        return flag


# Initialize FastMCP server
mcp = FastMCP("robots")


@mcp.tool()
def grasp_object(object: str):
    """
    Perform object grasping using Realman robotic arm, wrist-mounted camera, and Grounding DINO model.

    Args:
        object (str): The name of the target object to grasp (e.g., "apple", "bottle").

    Returns:
        str: A message indicating whether the grasping operation succeeded or failed.
    """
    
    dino = Dino()
    camera = Camera()
    arm = RealmanArm(robot_host, robot_port, speed, radius, connect, block)

    arm.observe()
    color_path, depth_path = camera.record_wrist_frame()
    boxes, _ = dino.predict(color_path, object)
    if not boxes:
        return f"{object} not found."

    uv = [int(x) for x in boxes[0][:2].numpy()]
    gripper_pose = arm.robot.rm_get_current_arm_state()[1]["pose"]
    extrinsics = dino.transform_camera2base(gripper_pose)
    xyz = dino.uv_to_xyz(uv, depth_path, extrinsics)

    grasp_pose = xyz.tolist() + [0, 0, 3.1]
    grasp_pose[0] -= 0.04
    grasp_pose[1] += 0.06
    grasp_pose[2] -= 0.1

    success = arm.grasp(grasp_pose)
    camera.shutdown()
    return f"'{object}' has been {'successfully' if success == 'ok' else 'failed'}  grasped"


if __name__ == "__main__":
    mcp.run(transport="stdio")
