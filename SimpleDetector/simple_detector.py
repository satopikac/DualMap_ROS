#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimpleDetector — 精简版 RGBD 语义检测工具
==========================================

功能概述:
  1. 订阅 ROS1 的 RGB、Depth、Odom 话题，实时缓存最新帧
  2. 用户在终端按 Enter 手动触发当前帧的语义识别
  3. 使用 YOLO 检测 → SAM 分割 → 深度图点云 → DBSCAN 去噪 → 3D AABB
  4. 检测结果累积追加到 output/semantic_map.json

依赖:
  - ROS1 (rospy, cv_bridge, message_filters, sensor_msgs, nav_msgs)
  - ultralytics (YOLO, SAM)
  - open3d, torch, numpy, scipy, opencv-python, PyYAML

用法:
  source /opt/ros/noetic/setup.bash   # 或你的 ROS 工作空间
  python simple_detector.py [--config config.yaml]
"""

import argparse
import json
import logging
import os
import sys
import threading
import time

from collections import deque
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SimpleDetector")


# ============================================================
# 工具函数：点云处理（来源于 DualMap_ROS/utils/pcd_utils.py）
# ============================================================

def mask_depth_to_points(depth, image, cam_K, masks, device="cuda"):
    """
    使用深度图、RGB 图、相机内参和检测 mask 生成每个物体的 3D 点云（相机坐标系）。

    参数:
        depth  : (H, W) torch.Tensor，深度图（米）
        image  : (H, W, 3) torch.Tensor，RGB 图像 [0,1]
        cam_K  : (3, 3) torch.Tensor，相机内参矩阵
        masks  : (N, H, W) torch.Tensor，N 个物体的二值 mask
        device : 计算设备

    返回:
        points : (N, H, W, 3) 每个 mask 对应的 3D 坐标
        colors : (N, H, W, 3) 每个 mask 对应的 RGB 颜色
    """
    N, H, W = masks.shape
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]

    # 生成像素网格坐标
    y, x = torch.meshgrid(
        torch.arange(0, H, device=device),
        torch.arange(0, W, device=device),
        indexing="ij",
    )

    # 每个 mask 乘以深度，只保留 mask 内的深度值
    z = depth.repeat(N, 1, 1) * masks  # (N, H, W)
    valid = (z > 0).float()

    # 反投影：像素坐标 → 相机坐标
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    points = torch.stack((x, y, z), dim=-1) * valid.unsqueeze(-1)  # (N, H, W, 3)

    # RGB 颜色
    rgb = image.repeat(N, 1, 1, 1) * masks.unsqueeze(-1)
    colors = rgb * valid.unsqueeze(-1)

    return points, colors


def refine_points_with_clustering(points, colors, eps=0.05, min_points=10):
    """
    使用 DBSCAN 聚类去噪，保留最大簇的点云。

    参数:
        points     : (M, 3) torch.Tensor 或 np.ndarray，点坐标
        colors     : (M, 3) torch.Tensor 或 np.ndarray，点颜色
        eps        : DBSCAN 邻域半径
        min_points : DBSCAN 最小点数

    返回:
        refined_points : (K, 3) np.ndarray
        refined_colors : (K, 3) np.ndarray
    """
    # 转为 numpy
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()

    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # 创建 Open3D 点云并执行 DBSCAN
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    # 去除噪声标签 -1
    mask_valid = labels != -1
    labels = labels[mask_valid]
    points = points[mask_valid]
    colors = colors[mask_valid]

    if len(labels) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # 保留最大簇
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_label = unique_labels[np.argmax(counts)]
    mask = labels == max_label

    return points[mask], colors[mask]


# ============================================================
# 工具函数：姿态与深度处理（来源于 runner_ros_base.py）
# ============================================================

def build_pose_matrix(translation, quaternion):
    """
    从平移向量和四元数构造 4x4 齐次变换矩阵。

    参数:
        translation : (3,) 平移 [x, y, z]
        quaternion  : (4,) 四元数 [x, y, z, w]（scipy 格式）

    返回:
        (4, 4) np.ndarray 齐次变换矩阵
    """
    from scipy.spatial.transform import Rotation as R
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


def create_world_transform(roll_deg, pitch_deg, yaw_deg):
    """
    根据欧拉角创建世界坐标系旋转补偿矩阵。

    参数:
        roll_deg, pitch_deg, yaw_deg : 绕 X/Y/Z 轴的旋转角度（度）

    返回:
        (4, 4) np.ndarray 旋转变换矩阵
    """
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    return T


def process_depth_image(depth_img, depth_factor):
    """
    将深度图转换为 float32 米制，形状 (H, W, 1)。

    支持:
      - uint16: 通常为毫米（depth_factor=1000.0）
      - float32/float64: 通常为米（depth_factor=1.0）

    参数:
        depth_img    : 原始深度图
        depth_factor : 除法因子，将原始值转换为米

    返回:
        (H, W, 1) np.float32 深度图
    """
    depth_img = depth_img.astype(np.float32) / depth_factor
    return np.expand_dims(depth_img, axis=-1)


# ============================================================
# 类别列表加载
# ============================================================

def load_classes(classes_path, bg_classes, skip_bg):
    """
    从文本文件加载类别列表。

    参数:
        classes_path : 类别文件路径，每行一个类别名
        bg_classes   : 背景类别列表（如 ["wall", "floor", "ceiling"]）
        skip_bg      : 是否跳过背景类别

    返回:
        classes : List[str] 类别名称列表
    """
    with open(classes_path, "r") as f:
        all_classes = [cls.strip() for cls in f.readlines() if cls.strip()]

    if skip_bg:
        classes = [c for c in all_classes if c not in bg_classes]
    else:
        classes = all_classes

    logger.info(f"加载了 {len(classes)} 个类别（skip_bg={skip_bg}）")
    return classes


# ============================================================
# 检测流水线：YOLO + SAM → 点云 → bbox
# ============================================================

def detect_and_segment(yolo_model, sam_model, rgb_image, classes):
    """
    对单张 RGB 图像执行 YOLO 检测 + SAM 分割。

    参数:
        yolo_model : ultralytics.YOLO 模型实例
        sam_model  : ultralytics.SAM 模型实例
        rgb_image  : (H, W, 3) np.uint8 RGB 图像
        classes    : List[str] 类别名称列表

    返回:
        detections : dict，包含以下键：
            - "xyxy"       : (N, 4) 检测框坐标
            - "class_id"   : (N,) 类别索引
            - "confidence" : (N,) 置信度
            - "masks"      : (N, H, W) 分割掩码
            如果没有检测结果则返回 None
    """
    # --- YOLO 检测 ---
    yolo_results = yolo_model.predict(rgb_image, verbose=False)
    if len(yolo_results) == 0 or yolo_results[0].boxes is None:
        return None

    boxes = yolo_results[0].boxes
    if len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()        # (N, 4)
    class_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
    confidences = boxes.conf.cpu().numpy()  # (N,)

    # --- SAM 分割：用 YOLO 的 bbox 作为 prompt ---
    sam_results = sam_model.predict(rgb_image, bboxes=xyxy, verbose=False)
    if len(sam_results) == 0 or sam_results[0].masks is None:
        return None

    masks = sam_results[0].masks.data.cpu().numpy()  # (N, H, W)

    # 确保数量一致（SAM 可能返回不同数量的 mask）
    n = min(len(xyxy), len(masks))
    return {
        "xyxy": xyxy[:n],
        "class_id": class_ids[:n],
        "confidence": confidences[:n],
        "masks": masks[:n],
    }


def filter_detections(detections, small_mask_threshold):
    """
    过滤掉过小的检测结果。

    参数:
        detections           : detect_and_segment() 的返回值
        small_mask_threshold : mask 最小像素面积

    返回:
        过滤后的 detections dict（可能为 None）
    """
    if detections is None:
        return None

    masks = detections["masks"]
    # 计算每个 mask 的像素面积
    areas = masks.sum(axis=(1, 2))
    keep = areas >= small_mask_threshold

    if not np.any(keep):
        return None

    return {
        "xyxy": detections["xyxy"][keep],
        "class_id": detections["class_id"][keep],
        "confidence": detections["confidence"][keep],
        "masks": detections["masks"][keep],
    }


def process_frame(detections, rgb, depth, intrinsics, pose, cfg):
    """
    对过滤后的检测结果，生成每个物体的 3D 点云和包围盒。

    参数:
        detections : 过滤后的检测 dict
        rgb        : (H, W, 3) np.uint8 RGB 图像
        depth      : (H, W, 1) np.float32 深度图（米）
        intrinsics : (3, 3) np.ndarray 相机内参
        pose       : (4, 4) np.ndarray 相机在世界坐标系下的位姿
        cfg        : 配置字典

    返回:
        objects : List[dict]，每个元素包含一个物体的 JSON 条目
    """
    device = cfg["device"]
    masks = detections["masks"]
    N = len(masks)

    # --- 准备 tensor ---
    depth_tensor = torch.from_numpy(depth).to(device).float().squeeze()       # (H, W)
    masks_tensor = torch.from_numpy(masks).to(device).float()                 # (N, H, W)
    intrinsic_tensor = torch.from_numpy(intrinsics).to(device).float()        # (3, 3)
    image_tensor = torch.from_numpy(rgb).to(device).float() / 255.0           # (H, W, 3)

    # --- 批量生成点云（相机坐标系）---
    points_tensor, colors_tensor = mask_depth_to_points(
        depth_tensor, image_tensor, intrinsic_tensor, masks_tensor, device
    )

    objects = []
    for i in range(N):
        mask_points = points_tensor[i]   # (H, W, 3)
        mask_colors = colors_tensor[i]   # (H, W, 3)

        # 筛选有效点（Z > 0）
        valid_mask = mask_points[:, :, 2] > 0
        if torch.sum(valid_mask) < cfg["min_points"]:
            continue

        valid_points = mask_points[valid_mask]   # (M, 3)
        valid_colors = mask_colors[valid_mask]   # (M, 3)

        # 随机降采样
        sample_ratio = cfg["pcd_sample_ratio"]
        num_pts = valid_points.shape[0]
        if sample_ratio < 1.0:
            sample_count = max(1, int(num_pts * sample_ratio))
            indices = torch.randperm(num_pts, device=device)[:sample_count]
            valid_points = valid_points[indices]
            valid_colors = valid_colors[indices]

        # DBSCAN 去噪
        refined_pts, refined_cols = refine_points_with_clustering(
            valid_points, valid_colors,
            eps=cfg["dbscan_eps"],
            min_points=cfg["dbscan_min_points"],
        )

        if refined_pts.shape[0] < cfg["min_points"]:
            continue

        # 创建 Open3D 点云并变换到世界坐标系
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(refined_pts)
        pcd.colors = o3d.utility.Vector3dVector(refined_cols)
        pcd.transform(pose)

        # --- 3D AABB 包围盒 ---
        points_arr = np.asarray(pcd.points)
        if points_arr.shape[0] == 0:
            continue
        bbox_3d = pcd.get_axis_aligned_bounding_box()
        bbox_center_3d = np.asarray(bbox_3d.get_center()).tolist()
        bbox_extent_3d = np.asarray(bbox_3d.get_extent()).tolist()
        bbox_min_3d = np.asarray(bbox_3d.get_min_bound()).tolist()
        bbox_max_3d = np.asarray(bbox_3d.get_max_bound()).tolist()

        # --- 2D 俯视图包围盒（将 3D 点云投影到 XY 平面）---
        # 与 DualMap_ROS 的 voxel_downsample_2d() 一致：
        # 取点云的 X/Y 坐标，丢弃 Z，计算 2D AABB
        points_2d = points_arr[:, :2]  # 只保留 X, Y
        bbox_center_2d = points_2d.mean(axis=0).tolist()  # 近似中心
        min_2d = points_2d.min(axis=0)
        max_2d = points_2d.max(axis=0)
        bbox_center_2d = ((min_2d + max_2d) / 2).tolist()
        bbox_extent_2d = (max_2d - min_2d).tolist()

        # 构建 JSON 条目
        class_name = detections["_classes"][detections["class_id"][i]]
        entry = {
            "class_name": class_name,
            "bbox_2d": {
                "center": bbox_center_2d,
                "extent": bbox_extent_2d,
            },
            "bbox_3d": {
                "center": bbox_center_3d,
                "extent": bbox_extent_3d,
                "min_bound": bbox_min_3d,
                "max_bound": bbox_max_3d,
            },
        }
        objects.append(entry)

    return objects


# ============================================================
# JSON 保存（累积追加）
# ============================================================

def save_semantic_map(objects, json_path):
    """
    将新检测到的物体追加到语义地图 JSON 文件。

    参数:
        objects   : List[dict] 新检测的物体条目
        json_path : JSON 文件路径

    文件格式:
        {
            "metadata": { "total_objects": N, "last_updated": "..." },
            "objects": [ ... ]
        }
    """
    # 读取已有数据
    existing_objects = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "objects" in data:
                existing_objects = data["objects"]
        except (json.JSONDecodeError, KeyError):
            logger.warning("已有 JSON 文件损坏，将重新创建")

    # 追加新物体
    all_objects = existing_objects + objects

    output = {
        "metadata": {
            "total_objects": len(all_objects),
            "last_updated": datetime.now().isoformat(),
        },
        "objects": all_objects,
    }

    # 原子写入
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, json_path)

    logger.info(f"已保存 {len(objects)} 个新物体，累计 {len(all_objects)} 个 → {json_path}")


# ============================================================
# ROS1 数据订阅与同步
# ============================================================

class ROSReceiver:
    """
    ROS1 数据接收器：订阅 RGB、Depth、Odom 话题，时间同步后缓存最新帧。

    用法:
        receiver = ROSReceiver(cfg)
        frame = receiver.get_latest_frame()  # 返回最新帧或 None
    """

    def __init__(self, cfg):
        import rospy
        from cv_bridge import CvBridge
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image

        self.cfg = cfg
        self.bridge = CvBridge()
        self.frame_buffer = deque(maxlen=1)  # 只保留最新一帧
        self.intrinsics = None
        self.lock = threading.Lock()
        self.received_first_frame = False  # 是否已收到第一帧同步数据

        # 加载内参（优先使用配置文件）
        if "intrinsic" in cfg and cfg["intrinsic"]:
            intr = cfg["intrinsic"]
            self.intrinsics = np.array([
                [intr["fx"], 0, intr["cx"]],
                [0, intr["fy"], intr["cy"]],
                [0, 0, 1],
            ])
            logger.info(f"从配置加载相机内参: fx={intr['fx']}, fy={intr['fy']}")

        # 加载外参
        if "extrinsics" in cfg and cfg["extrinsics"]:
            self.extrinsics = np.array(cfg["extrinsics"])
        else:
            self.extrinsics = np.eye(4)

        # 世界坐标变换
        self.world_transform = create_world_transform(
            cfg.get("world_roll", 0),
            cfg.get("world_pitch", 0),
            cfg.get("world_yaw", 0),
        )

        topics = cfg["ros_topics"]

        # 创建订阅者
        if cfg.get("use_compressed_topic", False):
            rgb_sub = Subscriber(topics["rgb"], CompressedImage)
            depth_sub = Subscriber(topics["depth"], CompressedImage)
        else:
            rgb_sub = Subscriber(topics["rgb"], Image)
            depth_sub = Subscriber(topics["depth"], Image)

        odom_sub = Subscriber(topics["odom"], Odometry)

        # 时间同步
        sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, odom_sub],
            queue_size=10,
            slop=cfg.get("sync_threshold", 0.1),
        )
        sync.registerCallback(self._synced_callback)

        # 备用：从 camera_info 获取内参
        if self.intrinsics is None:
            rospy.Subscriber(topics["camera_info"], CameraInfo, self._camera_info_cb)

        logger.info("ROS 话题订阅完成，等待数据...")

    def _synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """RGB + Depth + Odom 同步回调。"""
        try:
            timestamp = rgb_msg.header.stamp.to_sec()

            # 解码 RGB
            if self.cfg.get("use_compressed_topic", False):
                np_arr = np.frombuffer(bytes(rgb_msg.data), np.uint8)
                rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            else:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

            # 解码 Depth
            if self.cfg.get("use_compressed_topic", False):
                depth_data = np.frombuffer(bytes(depth_msg.data)[12:], np.uint8)
                depth = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)
            else:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            # 处理深度图
            depth = process_depth_image(depth, self.cfg.get("depth_factor", 1000.0))

            # 构建位姿矩阵
            pos = odom_msg.pose.pose.position
            ori = odom_msg.pose.pose.orientation
            translation = np.array([pos.x, pos.y, pos.z])
            quaternion = np.array([ori.x, ori.y, ori.z, ori.w])
            pose = build_pose_matrix(translation, quaternion)

            # 应用外参和世界变换
            pose = self.world_transform @ (pose @ self.extrinsics)

            with self.lock:
                self.frame_buffer.append({
                    "rgb": rgb,
                    "depth": depth,
                    "pose": pose,
                    "intrinsics": self.intrinsics,
                    "timestamp": timestamp,
                })
                if not self.received_first_frame:
                    self.received_first_frame = True
                    logger.info("已收到第一帧同步数据，话题连接正常")

        except Exception as e:
            logger.error(f"数据回调异常: {e}")

    def _camera_info_cb(self, msg):
        """从 CameraInfo 话题获取内参（备用）。"""
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.K).reshape(3, 3)
            logger.info("从 camera_info 话题获取到相机内参")

    def get_latest_frame(self):
        """
        获取最新一帧数据。

        返回:
            dict: 包含 rgb, depth, pose, intrinsics, timestamp
            None: 如果缓冲区为空或内参未就绪
        """
        with self.lock:
            if not self.frame_buffer:
                return None
            frame = self.frame_buffer[-1]
            if frame["intrinsics"] is None:
                return None
            return frame


# ============================================================
# 主程序
# ============================================================

def main():
    # --- 解析参数 ---
    parser = argparse.ArgumentParser(description="SimpleDetector: 手动触发的 RGBD 语义检测工具")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # --- 加载配置 ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"配置文件已加载: {args.config}")

    # --- 创建输出目录 ---
    output_dir = cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "semantic_map.json")

    # --- 加载类别列表 ---
    classes = load_classes(
        cfg["classes_path"],
        cfg.get("bg_classes", []),
        cfg.get("skip_bg", False),
    )

    # --- 加载模型 ---
    logger.info("正在加载 YOLO 模型...")
    from ultralytics import SAM, YOLO
    yolo_model = YOLO(cfg["yolo_model"])
    yolo_model.set_classes(classes)
    logger.info(f"YOLO 模型加载完成: {cfg['yolo_model']}")

    logger.info("正在加载 SAM 模型...")
    sam_model = SAM(cfg["sam_model"])
    logger.info(f"SAM 模型加载完成: {cfg['sam_model']}")

    # --- 初始化 ROS ---
    import rospy
    rospy.init_node("simple_detector", anonymous=True)
    receiver = ROSReceiver(cfg)

    # --- 启动 ROS spin 线程 ---
    spin_thread = threading.Thread(target=lambda: rospy.spin(), daemon=True)
    spin_thread.start()

    # --- 等待话题数据就绪 ---
    topics = cfg["ros_topics"]
    logger.info("=" * 60)
    logger.info("等待 ROS 话题数据...")
    logger.info(f"  RGB:   {topics['rgb']}")
    logger.info(f"  Depth: {topics['depth']}")
    logger.info(f"  Odom:  {topics['odom']}")
    logger.info("=" * 60)

    while not rospy.is_shutdown() and not receiver.received_first_frame:
        time.sleep(0.5)
        sys.stdout.write("\r[等待中] 尚未收到同步的 RGB + Depth + Odom 数据...")
        sys.stdout.flush()

    if rospy.is_shutdown():
        logger.info("ROS 已关闭，退出")
        return

    # 清除等待提示行
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()
    logger.info("话题数据就绪，可以开始检测")

    # --- 主循环：等待用户输入触发检测 ---
    frame_count = 0
    logger.info("=" * 60)
    logger.info("SimpleDetector 已启动")
    logger.info("按 Enter 触发当前帧的语义检测，输入 q 退出")
    logger.info("=" * 60)

    try:
        while not rospy.is_shutdown():
            user_input = input("\n>>> 按 Enter 触发检测（q 退出）: ").strip().lower()
            if user_input == "q":
                logger.info("用户请求退出")
                break

            # 获取最新帧
            frame = receiver.get_latest_frame()
            if frame is None:
                logger.warning("当前没有可用的帧数据（等待 ROS 话题数据...）")
                continue

            frame_count += 1
            logger.info(f"--- 第 {frame_count} 次检测 (timestamp={frame['timestamp']:.3f}) ---")

            # Step 1: YOLO + SAM 检测分割
            t0 = time.time()
            detections = detect_and_segment(yolo_model, sam_model, frame["rgb"], classes)
            t1 = time.time()

            if detections is None:
                logger.info(f"未检测到任何物体（耗时 {t1-t0:.2f}s）")
                continue

            # 过滤小 mask
            detections = filter_detections(detections, cfg.get("small_mask_threshold", 204))
            if detections is None:
                logger.info(f"过滤后无有效检测（耗时 {t1-t0:.2f}s）")
                continue

            # 附加类别名称映射
            detections["_classes"] = classes
            logger.info(f"检测到 {len(detections['masks'])} 个物体（YOLO+SAM 耗时 {t1-t0:.2f}s）")

            # Step 2: 生成点云 + bbox
            t2 = time.time()
            objects = process_frame(
                detections, frame["rgb"], frame["depth"],
                frame["intrinsics"], frame["pose"], cfg,
            )
            t3 = time.time()
            logger.info(f"生成 {len(objects)} 个有效 3D 物体（点云处理耗时 {t3-t2:.2f}s）")

            if not objects:
                continue

            # 打印检测摘要
            for obj in objects:
                c3 = obj["bbox_3d"]["center"]
                c2 = obj["bbox_2d"]["center"]
                logger.info(
                    f"  [{obj['class_name']}] "
                    f"3D=({c3[0]:.2f}, {c3[1]:.2f}, {c3[2]:.2f}) "
                    f"2D=({c2[0]:.2f}, {c2[1]:.2f})"
                )

            # Step 3: 保存到 JSON
            save_semantic_map(objects, json_path)

    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在退出...")
    except EOFError:
        logger.info("输入流结束，退出")

    logger.info("SimpleDetector 已退出")


if __name__ == "__main__":
    main()
