# DualMap 环境感知建图保存管线 — 完整实现分析

本文档从第一个字节进入系统到最后一个文件写入磁盘，逐函数、逐行地追踪 DualMap 的完整数据管线。

---

## 目录

1. [管线总览](#1-管线总览)
2. [阶段一：ROS 数据接入](#2-阶段一ros-数据接入)
3. [阶段二：关键帧选择](#3-阶段二关键帧选择)
4. [阶段三：物体检测（Detector）](#4-阶段三物体检测detector)
5. [阶段四：观测生成（Observation）](#5-阶段四观测生成observation)
6. [阶段五：局部地图管理（LocalMapManager）](#6-阶段五局部地图管理localmapmanager)
7. [阶段六：全局地图管理（GlobalMapManager）](#7-阶段六全局地图管理globalmapmanager)
8. [阶段七：持久化保存](#8-阶段七持久化保存)
9. [并行线程模型](#9-并行线程模型)
10. [附录：关键数据结构字段速查](#10-附录关键数据结构字段速查)

---

## 1. 管线总览

```
┌─────────────────────────────────────────────────────────────────────┐
│  ROS 话题                                                           │
│  /camera/rgb  /camera/depth  /camera/pose                          │
└───────┬──────────┬──────────────┬───────────────────────────────────┘
        │          │              │
        ▼          ▼              ▼
┌─────────────────────────────────────────┐
│ ① ROS 数据接入 (RunnerROS1/2)          │
│   时间同步 → 图像解码 → 位姿矩阵构建   │
│   → 坐标变换 → DataInput 封装           │
└───────────────────┬─────────────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ ② 关键帧选择 (Dualmap.check_keyframe)    │
│   时间阈值 / 平移阈值 / 旋转阈值         │
└───────────────────┬───────────────────────┘
                    ▼
┌───────────────────────────────────────────────────────────────┐
│ ③ 物体检测 (Detector.process_detections)                      │
│                                                               │
│   ┌──────────┐  并行  ┌───────────┐                          │
│   │ YOLO     │◄──────►│ FastSAM   │                          │
│   │ 已知类别 │        │ 未知类别  │                           │
│   └────┬─────┘        └─────┬─────┘                          │
│        ▼                    │                                 │
│   ┌──────────┐              │                                 │
│   │ SAM      │   合并过滤 ◄─┘                                │
│   │ 精细掩膜 │──────┐                                        │
│   └──────────┘      ▼                                        │
│              ┌────────────────┐   ┌──────────────────┐       │
│              │ 掩膜×深度→3D点 │   │ CLIP 特征提取    │       │
│              │ DBSCAN 去噪    │   │ 图像+文本加权    │       │
│              └───────┬────────┘   └────────┬─────────┘       │
│                      └──────┬──────────────┘                 │
│                             ▼                                │
│                    移动性分类 (CLIP 原型)                      │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ ④ 观测生成 (Detector.calculate_observations)                  │
│   每个检测 → LocalObservation(pcd, bbox, clip_ft, class_id,  │
│                               conf, is_low_mobility, ...)    │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ ⑤ 局部地图 (LocalMapManager.process_observations)             │
│                                                               │
│   匹配: Tracker(空间重叠 + CLIP余弦相似度)                     │
│     ▼                                                         │
│   更新: 合并点云, 贝叶斯类别概率, 分裂检测                      │
│     ▼                                                         │
│   状态机: UPDATING → PENDING → STABLE                         │
│     ▼                    ▼                                    │
│   稳定 + 低移动性 → create_global_observation()               │
│     ▼                                                         │
│   On-Relation 检测: 杯子在桌子上 → 打包关联物体                │
│     ▼                                                         │
│   GlobalObservation (含 pcd_2d 俯视投影)                      │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ ⑥ 全局地图 (GlobalMapManager.process_observations)            │
│                                                               │
│   匹配: 2D bbox 交集比率 > 0.8                                │
│     ▼                                                         │
│   新物体 → 添加 GlobalObject                                  │
│   已有物体 → 合并点云/特征/关联物体                             │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ ⑦ 持久化保存                                                  │
│                                                               │
│   实时: semantic_map.json + spatial_graph.json (每次更新)      │
│   结束: *.pkl (物体) + layout.pcd + wall.pcd                  │
│         + 路径 JSON + 计时 CSV                                │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段一：ROS 数据接入

### 2.1 入口与版本检测

**文件**: `applications/runner_ros.py`

```python
def detect_ros_version():
    if importlib.util.find_spec("rclpy") is not None:
        return "ros2"
    elif importlib.util.find_spec("rospy") is not None:
        return "ros1"
```

通过检测 Python 模块是否可导入来判断 ROS 版本，然后分发到对应的 Runner。

### 2.2 话题订阅与时间同步

**文件**: `applications/utils/runner_ros1.py` (ROS1) / `runner_ros2.py` (ROS2)

系统订阅三个话题：

| 话题  | 消息类型                         | 内容                         |
| ----- | -------------------------------- | ---------------------------- |
| rgb   | `Image` 或 `CompressedImage` | RGB 彩色图像                 |
| depth | `Image` 或 `CompressedImage` | 深度图                       |
| odom  | `Odometry`                     | 位姿（平移+四元数）/nav_msgs |

使用 `ApproximateTimeSynchronizer` 进行三路时间同步：

```python
self.sync = ApproximateTimeSynchronizer(
    [self.rgb_sub, self.depth_sub, self.odom_sub],
    queue_size=10,
    slop=self.cfg.sync_threshold,  # 默认 0.1 秒
)
self.sync.registerCallback(self.synced_callback)
```

**含义**: 三个话题的时间戳偏差在 0.1 秒内才触发回调。`queue_size=10` 意味着每个话题缓存最新 10 条消息用于匹配。

### 2.3 同步回调：图像解码与位姿构建

**文件**: `runner_ros1.py:synced_callback` (以 ROS1 为例，ROS2 逻辑一致)

```python
def synced_callback(self, rgb_msg, depth_msg, odom_msg):
    timestamp = rgb_msg.header.stamp.to_sec()

    # 图像解码
    if self.cfg.use_compressed_topic:
        rgb_img = self.decompress_image(rgb_msg.data, is_depth=False)
        depth_img = self.decompress_image(depth_msg.data, is_depth=True)
    else:
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
```

**图像解码细节** (`runner_ros_base.py:decompress_image`):

- 非压缩模式：直接用 `cv_bridge` 将 ROS Image 转为 numpy 数组
- 压缩模式：RGB 用 `cv2.imdecode(np_arr, IMREAD_COLOR)` + BGR→RGB 转换；深度用 `cv2.imdecode(data[12:], IMREAD_UNCHANGED)` (跳过 12 字节 compressedDepth 头)

**深度图处理** (`runner_ros_base.py:process_depth_image`):

```python
def process_depth_image(self, depth_img, depth_factor):
    if depth_img.dtype == np.uint16:
        depth_img = depth_img.astype(np.float32) / depth_factor   # 毫米→米
    elif depth_img.dtype in [np.float32, np.float64]:
        depth_img = depth_img.astype(np.float32) / depth_factor   # 已是米或需要其他因子
    depth_img = np.expand_dims(depth_img, axis=-1)  # (H,W) → (H,W,1)
    return depth_img
```

D435i 输出 `uint16` 毫米深度，`depth_factor=1000.0` 转换为米。

**位姿矩阵构建** (`runner_ros_base.py:build_pose_matrix`):

```python
def build_pose_matrix(self, translation, quaternion):
    rotation_matrix = R.from_quat(quaternion).as_matrix()  # scipy: [x,y,z,w]
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    return transformation_matrix
```

从 Odometry 消息中提取平移 `[x,y,z]` 和四元数 `[x,y,z,w]`，构建 4×4 齐次变换矩阵。

### 2.4 坐标变换与数据封装

**文件**: `runner_ros_base.py:push_data`

```python
def push_data(self, rgb_img, depth_img, pose, timestamp):
    transformed_pose = self.create_world_transform() @ (pose @ self.extrinsics)
    data_input = DataInput(
        idx=self.kf_idx,
        time_stamp=timestamp,
        color=rgb_img,            # np.ndarray (H,W,3) uint8
        depth=depth_img,          # np.ndarray (H,W,1) float32, 米
        color_name=str(timestamp),
        intrinsics=self.intrinsics,  # np.ndarray (3,3)
        pose=transformed_pose,       # np.ndarray (4,4) SE(3)
    )
    self.synced_data_queue.append(data_input)
```

**坐标变换链**:

```
world_pose = world_transform × (odom_pose × extrinsics)
```

- `extrinsics`: 相机安装外参，camera_optical_frame → robot_base，4×4 矩阵。如果 odom 直接给的是相机位姿则为单位矩阵。
- `odom_pose`: 来自 Odometry 话题的位姿，robot_base → odom_frame。
- `world_transform`: 从配置文件 `world_roll/pitch/yaw` 构建的旋转矩阵，用于将 odom 坐标系对齐到世界坐标系。通常为单位矩阵（标准 ROS 约定下 Z 轴朝上）。

`create_world_transform` 构建 Rz(yaw) × Ry(pitch) × Rx(roll) 的旋转矩阵并嵌入 4×4 齐次矩阵。

`synced_data_queue` 是一个 `deque(maxlen=1)` — 只保留最新一帧，旧的自动丢弃。

### 2.5 DataInput 数据结构

**文件**: `utils/types.py`

```python
@dataclass
class DataInput:
    idx: int = 0                    # 关键帧编号
    time_stamp: float = 0.0         # ROS 时间戳 (秒)
    color: np.ndarray               # RGB 图像 (H, W, 3), uint8
    depth: np.ndarray               # 深度图 (H, W, 1), float32, 单位米
    color_name: str = ""            # 帧标识名
    intrinsics: np.ndarray          # 相机内参 (3, 3), [fx,0,cx; 0,fy,cy; 0,0,1]
    pose: np.ndarray                # 世界坐标系位姿 (4, 4), SE(3)
```

---

## 3. 阶段二：关键帧选择

**文件**: `applications/utils/runner_ros_base.py:run_once` → `dualmap/core.py:check_keyframe`

主循环以 `ros_rate`（默认 15Hz）频率调用 `run_once`：

```python
def run_once(self, current_time_fn):
    if not self.synced_data_queue:
        return
    data_input = self.synced_data_queue[-1]
    # ...
    if not self.dualmap.check_keyframe(data_input.time_stamp, data_input.pose):
        return   # 不是关键帧，跳过
    # 是关键帧 → 进入处理
    data_input.idx = self.dualmap.get_keyframe_idx()
    if self.cfg.use_parallel:
        self.dualmap.parallel_process(data_input)
    else:
        self.dualmap.sequential_process(data_input)
```

**关键帧判定逻辑** (`core.py:check_keyframe`)，满足任一条件即为关键帧：

1. **平移阈值**: 与上一关键帧的平移距离 ≥ `pose_threshold`（默认 0.1m）

   ```python
   translation_diff = np.linalg.norm(curr_pose[:3, 3] - self.last_keyframe_pose[:3, 3])
   if translation_diff >= self.pose_threshold:
       is_keyframe = True
   ```
2. **旋转阈值**: 与上一关键帧的旋转角度差 ≥ `rotation_threshold`（默认 3°）

   ```python
   rotation_diff = curr_rotation.inv() * last_rotation
   angle_diff = rotation_diff.magnitude() * (180 / np.pi)
   if angle_diff >= self.rotation_threshold:
       is_keyframe = True
   ```
3. **时间阈值**: 距上一关键帧的时间间隔 ≥ `time_threshold`（默认 0.5s）

   ```python
   if abs(time_stamp - self.last_keyframe_time) >= self.time_threshold:
       is_keyframe = True
   ```

**效果**: 机器人静止不动时，约每 0.5 秒取一帧；移动时，约每 0.1 米或每 3° 旋转取一帧。这过滤掉了大量冗余帧，只保留有信息增量的关键帧进入后续管线。

---

## 4. 阶段三：物体检测（Detector）

**文件**: `utils/object_detector.py:process_detections`

这是整个管线中计算最密集的阶段。

### 4.1 总体流程

```python
def process_detections(self):
    color = self.curr_data.color.astype(np.uint8)

    # ---- YOLO + SAM 与 FastSAM 并行 ----
    if self.cfg.use_fastsam:
        fastsam_thread = threading.Thread(target=self.process_fastsam, args=(color,))
        fastsam_thread.start()

    self.process_yolo_and_sam(color)      # 主线程: YOLO → SAM

    if self.cfg.use_fastsam:
        fastsam_thread.join()             # 等待 FastSAM 完成

    # ---- 过滤 ----
    self.filter.update_detections(self.curr_detections, color)
    filtered_detections = self.filter.run_filter()  # 小掩膜过滤等

    # ---- 合并 FastSAM 补充检测 ----
    if self.cfg.use_fastsam and self.fastsam_detections:
        filtered_detections = self.add_extra_detections_from_fastsam(
            color, self.fastsam_detections, filtered_detections
        )

    # ---- 3D 点云 与 CLIP 特征 并行 ----
    cluster_thread = threading.Thread(target=self.process_masks, args=(filtered_detections.mask,))
    cluster_thread.start()

    image_crops, image_feats, text_feats = self.compute_clip_features_batched(
        color, filtered_detections, ...)

    cluster_thread.join()

    self.curr_results = {
        "xyxy": filtered_detections.xyxy,
        "confidence": filtered_detections.confidence,
        "class_id": filtered_detections.class_id,
        "masks": filtered_detections.mask,
        "image_feats": image_feats,
        "text_feats": text_feats,
    }
```

### 4.2 YOLO: 已知类别检测

**函数**: `process_yolo_results`

```python
results = self.yolo.predict(color, conf=0.2, verbose=False)
```

- 输入: RGB 图像 (H×W×3)
- 输出: 每个检测的 `confidence`（置信度）、`class_id`（类别索引）、`xyxy`（边界框 [x_min, y_min, x_max, y_max]）
- 模型: `yolov8l-world.pt` — YOLO-World 开放词汇版本
- 类别设置: 通过 `self.yolo.set_classes(class_list)` 设定，类别来自 `config/class_list/gpt_indoor_general.txt` 之类的文件（约 80 个室内物体类别）
- 置信度阈值: 0.2（较低，尽量多召回）

### 4.3 SAM: 精细分割

**函数**: `process_yolo_and_sam` 中的 SAM 部分

```python
sam_out = self.sam.predict(color, bboxes=xyxy, verbose=False)
masks_tensor = sam_out[0].masks.data   # (N, H, W) boolean
masks_np = masks_tensor.cpu().numpy()
```

- 输入: RGB 图像 + YOLO 输出的边界框列表
- 输出: 每个边界框对应的像素级掩膜 (N, H, W)
- 模型: `mobile_sam.pt` — MobileSAM 轻量版

**SAM 的必要性**: YOLO 只给矩形边界框，框内包含大量背景。SAM 给出精确的像素掩膜，后续用 `mask × depth` 才能正确提取仅属于该物体的 3D 点云，避免背景污染。

### 4.4 FastSAM: 未知物体补充

**函数**: `process_fastsam` + `filter_fs_detections_by_curr` + `add_extra_detections_from_fastsam`

FastSAM 的作用是检测 YOLO 类别列表中没有的物体。

```python
# FastSAM 独立推理（在独立线程中）
results = self.fastsam(color, device="cuda", retina_masks=True,
                       imgsz=1024, conf=0.80, iou=0.9)
# 所有检测的 class_id 设为 unknown_class_id
detection_class_id_np = np.full_like(detection_class_id_np, self.unknown_class_id)
```

FastSAM 以类别无关方式分割整张图中所有"有边界的物体"，不区分类别。其高置信度阈值 (0.80) 确保只保留清晰的分割结果。

**关键: 去重过滤** — FastSAM 会检测到 YOLO 已经检测到的物体，需要去除重叠：

```python
def filter_fs_detections_by_curr(self, fs_detections, curr_detections, ...):
    # GPU 上批量计算所有 FastSAM 掩膜与 YOLO 掩膜的 IoU
    fs_masks_flat = fs_masks.view(num_fs, -1).to(torch.float32)
    curr_masks_flat = curr_masks.view(num_curr, -1).to(torch.float32)
    intersection = torch.matmul(fs_masks_flat, curr_masks_flat.T)
    # ...计算 IoU 和 overlap ratio...

    for i in range(num_fs):
        overlap = (iou_matrix[i] > 0.5) | (overlap_ratio_fs[i] > 0.6) | (overlap_ratio_curr[i] > 0.6)
        if overlap.any():
            keep_mask[i] = False   # 与 YOLO 重叠 → 丢弃
```

**效果**: 假设场景中有一个"书"（不在 YOLO 类别列表中），YOLO 检测不到它，但 FastSAM 能分割出它的掩膜。这个掩膜不与任何 YOLO 掩膜重叠，因此被保留。最终合并：

```
已知类别 (YOLO+SAM) + 未知物体 (FastSAM 补充) = 完整检测集
```

### 4.5 3D 点云生成

**函数**: `process_masks`

对每个检测到的物体掩膜，将掩膜区域的深度像素反投影为 3D 点云：

```python
def process_masks(self, masks):
    N, _, _ = masks.shape
    depth_tensor = torch.from_numpy(self.curr_data.depth).to(device).float().squeeze()
    masks_tensor = torch.from_numpy(masks).to(device).float()
    intrinsic_tensor = torch.from_numpy(self.curr_data.intrinsics).to(device).float()
    image_rgb_tensor = torch.from_numpy(self.curr_data.color).to(device).float() / 255.0

    # 批量计算所有掩膜的 3D 点
    points_tensor, colors_tensor = mask_depth_to_points(
        depth_tensor, image_rgb_tensor, intrinsic_tensor, masks_tensor, device)
```

**`mask_depth_to_points`** (`pcd_utils.py`) 的核心计算:

```python
def mask_depth_to_points(depth, image, cam_K, masks, device):
    N, H, W = masks.shape
    fx, fy, cx, cy = cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2]

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

    z = depth.repeat(N, 1, 1) * masks          # 掩膜区域的深度值
    valid = (z > 0).float()                     # 有效深度

    x = (x - cx) * z / fx                      # 像素坐标 → 相机坐标系 X
    y = (y - cy) * z / fy                      # 像素坐标 → 相机坐标系 Y

    points = torch.stack((x, y, z), dim=-1) * valid.unsqueeze(-1)   # (N, H, W, 3)
    colors = image.repeat(N, 1, 1, 1) * masks.unsqueeze(-1)         # 取对应颜色
    return points, colors
```

**原理**: 针孔相机模型的反投影。对于掩膜内每个像素 `(u, v)`，已知深度 `d`：

```
X = (u - cx) × d / fx
Y = (v - cy) × d / fy
Z = d
```

得到相机坐标系下的 3D 点 `(X, Y, Z)`。

随后对每个物体的点云做降采样和 DBSCAN 聚类去噪：

```python
for i in range(N):
    valid_points = mask_points[valid_points_mask]
    # 随机降采样（sample_ratio=0.04，保留 4%）
    sample_count = int(num_points * sample_ratio)
    sample_indices = torch.randperm(num_points)[:sample_count]
    downsampled_points = valid_points[sample_indices]

    # DBSCAN 聚类去噪，只保留最大簇
    refined_points, refined_colors = refine_points_with_clustering(
        downsampled_points, downsampled_colors, eps=0.1, min_points=10)
```

**`refine_points_with_clustering`** 使用 Open3D 的 DBSCAN：

1. 聚类 (`eps=0.1m`, `min_points=10`)
2. 丢弃噪声点 (label=-1)
3. 只保留最大簇的点

**效果**: 假设一个"椅子"的掩膜稍微覆盖了后面的墙壁，那些远离椅子主体的离散点（与椅子主体距离 > 0.1m）会被 DBSCAN 归为不同簇或噪声，从而被丢弃。

### 4.6 CLIP 特征提取

**函数**: `compute_clip_features_batched`

对每个检测到的物体，提取两种 CLIP 特征：

```python
def compute_clip_features_batched(self, image, detections, clip_model, ...):
    for idx in range(len(detections.xyxy)):
        # 裁剪物体区域（加 20px padding）
        cropped_image = image.crop((x_min - pad, y_min - pad, x_max + pad, y_max + pad))
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)

        # 获取类别名
        class_id = detections.class_id[idx]
        text_tokens.append(classes[class_id])   # 如 "chair"

    # 批量推理
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = clip_model.encode_text(text_tokens_batch)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 对 FastSAM 检测的未知物体，文本特征用所有类别的均值替代
    if self.cfg.use_avg_feat_for_unknown:
        for idx, class_id in enumerate(detections.class_id):
            if class_id == self.unknown_class_id:
                text_feats[idx] = self.class_feats_mean
```

**两种特征的含义**:

- `image_feats`: 物体裁剪图像经 CLIP 视觉编码器得到的 512 维向量 — 描述"这个物体看起来像什么"
- `text_feats`: 类别名经 CLIP 文本编码器得到的 512 维向量 — 描述"这个类别名在语义空间中的位置"

**加权融合** (`get_weighted_feature`):

```python
weighted_feature = 0.7 * image_feat + 0.3 * text_feat
weighted_feature /= np.linalg.norm(weighted_feature)
```

为什么加权而不只用 image_feat？因为 image_feat 容易受遮挡、光照、视角影响，而 text_feat 提供了稳定的类别锚点。70:30 的比例让视觉信息为主、文本为辅。

### 4.7 移动性分类

**函数**: `is_low_mobility`

使用 CLIP 文本原型与物体 CLIP 特征的余弦相似度进行分类：

```python
def is_low_mobility(self, clip_feat):
    sim = cosine_similarity(clip_feat.reshape(1,-1), self.proto_feats)
    sim_lm = np.max(sim[:num_lm])        # 与低移动性原型（table, sofa, ...）的最大相似度
    sim_hm = np.max(sim[num_lm:num_lm+num_hm])  # 与高移动性原型（person, cat, ...）的最大相似度
    sim_lm_des = np.max(sim[num_lm+num_hm:])     # 与低移动性描述的最大相似度

    if sim_lm > sim_hm + 0.05:    return True   # 明显更像低移动性
    elif sim_lm + 0.05 < sim_hm:  return False  # 明显更像高移动性
    else:                          return sim_lm_des > 0.45  # 接近时用描述判定
```

**意义**: 低移动性物体（家具、电器）是持久的环境结构，应该进入全局语义地图。高移动性物体（人、宠物）是动态的，不应出现在持久地图中。

---

## 5. 阶段四：观测生成（Observation）

**文件**: `object_detector.py:calculate_observations`

将检测结果转化为结构化的 `LocalObservation` 对象列表。

```python
def calculate_observations(self):
    N, _, _ = self.curr_results["masks"].shape

    for i in range(N):
        if self.masked_points[i] is None:
            continue           # 点云为空的检测跳过

        # 创建 Open3D 点云并变换到世界坐标系
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.masked_points[i])
        pcd.colors = o3d.utility.Vector3dVector(self.masked_colors[i])
        pcd.transform(self.curr_data.pose)   # 相机坐标系 → 世界坐标系

        # 计算包围盒
        bbox = safe_create_bbox(pcd)     # 轴对齐包围盒 AABB

        # 计算物体到相机的距离（用于贝叶斯权重）
        distance = self.get_distance(bbox, self.curr_data.pose)

        # 封装为 LocalObservation
        curr_obs = LocalObservation()
        curr_obs.idx = self.curr_data.idx           # 关键帧编号
        curr_obs.class_id = self.curr_results["class_id"][i]
        curr_obs.conf = self.curr_results["confidence"][i]
        curr_obs.clip_ft = self.get_weighted_feature(idx=i)   # 加权 CLIP 特征
        curr_obs.pcd = pcd                          # 世界坐标系点云
        curr_obs.bbox = bbox                        # 世界坐标系 AABB
        curr_obs.distance = distance
        curr_obs.is_low_mobility = self.is_low_mobility(curr_obs.clip_ft)

        self.curr_observations.append(curr_obs)
```

**重要**: `pcd.transform(self.curr_data.pose)` 将相机坐标系下的点云变换到世界坐标系。这样不同帧检测到的同一物体的点云在世界坐标系下可以直接合并对齐。

### LocalObservation 数据结构

```python
@dataclass
class LocalObservation(Observation):
    idx: int              # 关键帧编号
    class_id: int         # 类别 ID
    pcd: PointCloud       # 世界坐标系 3D 点云
    bbox: AABB            # 世界坐标系轴对齐包围盒
    clip_ft: ndarray      # 512D CLIP 加权特征
    conf: float           # YOLO 置信度
    distance: float       # 物体到相机的距离
    mask: ndarray         # 像素掩膜 (H,W)
    xyxy: ndarray         # 2D 边界框
    is_low_mobility: bool # 低/高移动性
    matched_obj_uid: UUID # 匹配到的已有物体 UID（初始为 None）
    matched_obj_idx: int  # 匹配到的已有物体索引（初始为 -1）
```

---

## 6. 阶段五：局部地图管理（LocalMapManager）

**文件**: `utils/local_map_manager.py`

局部地图是一个基于滑动窗口的临时物体跟踪层。

### 6.1 匹配：当前观测 vs 已有局部物体

**函数**: `process_observations`

```python
def process_observations(self, curr_observations):
    if not self.is_initialized:
        self.init_from_observation(curr_observations)  # 第一帧：直接创建
        return

    # 设置 Tracker
    self.tracker.set_current_frame(curr_observations)
    self.tracker.set_ref_map(self.local_map)
    self.tracker.matching_map()

    # 匹配完成后，observation 中的 matched_obj_idx 被填充
    curr_observations = self.tracker.get_current_frame()
    self.update_local_map(curr_observations)
```

**Tracker 匹配逻辑** (`utils/tracker.py:matching_map`):

局部匹配使用两种相似度之和：

**空间相似度** (`compute_spatial_sim`): 基于点云重叠率

```python
# 先用 3D IoU 预筛选（跳过不可能重叠的对）
iou = self.compute_3d_iou_batch(map_bbox_torch, curr_bbox_torch)

for idx_a in range(len_map):
    for idx_b in range(len_curr):
        if iou[idx_a, idx_b] < 1e-6:   # bbox 无交集，跳过
            continue
        # FAISS 最近邻搜索：当前观测的每个点在地图物体中找最近点
        D, I = indices_map[idx_a].search(points_curr[idx_b], 1)
        # 距离 < voxel_size² 的点算重叠
        overlap = (D < self.cfg.downsample_voxel_size ** 2).sum()
        overlap_matrix[idx_a, idx_b] = overlap / len(points_curr[idx_b])
```

**视觉相似度** (`compute_visual_sim`): CLIP 特征余弦相似度

```python
map_fts = map_feats_torch.unsqueeze(-1)    # (M, D, 1)
curr_fts = curr_feats_torch.T.unsqueeze(0)  # (1, D, N)
visual_sim = F.cosine_similarity(map_fts, curr_fts, dim=1)  # (M, N)
```

**总相似度与匹配分配**:

```python
sim_mat = spatial_sim_mat + visual_sim_mat   # 两者相加
sim_mat = sim_mat.T                          # (curr, map)

for idx in range(len_curr_obs):
    max_sim_value = sim_mat[idx].max()
    if max_sim_value > self.cfg.sim_threshold:   # 默认 1.2
        map_idx = sim_mat[idx].argmax().item()
        self.curr_frame[idx].matched_obj_idx = map_idx
    else:
        self.curr_frame[idx].matched_obj_idx = -1   # 无匹配 → 新物体
```

**含义**: `sim_threshold=1.2` 意味着空间重叠率 + CLIP 余弦相似度之和必须超过 1.2 才算匹配。例如空间重叠率 0.5 + CLIP 相似度 0.75 = 1.25 > 1.2 → 匹配成功。

### 6.2 局部地图更新

**函数**: `update_local_map`

```python
def update_local_map(self, curr_obs):
    for obs in curr_obs:
        if obs.matched_obj_idx == -1:
            # 新物体：创建 LocalObject
            local_obj = LocalObject()
            local_obj.add_observation(obs)
            local_obj.update_info()
            self.local_map.append(local_obj)
            self.graph.add_node(local_obj.uid)
        else:
            # 已有物体：追加观测
            matched_obj = self.local_map[obs.matched_obj_idx]
            matched_obj.add_observation(obs)
            matched_obj.update_info()

    # 分裂检测
    for obj in self.local_map:
        if obj.should_split:
            self.split_local_object(obj)

    self.update_map_and_graph()   # 执行待删除操作

    # 状态更新与动作
    for obj in self.local_map:
        obj.update_status()       # 更新生命周期状态
        self.status_actions(obj)  # 执行状态对应的动作

    self.update_map_and_graph()   # 再次执行待删除操作
```

### 6.3 LocalObject 信息更新

**函数**: `object.py:LocalObject.update_info`

每次追加新观测后更新物体信息：

```python
def update_info(self):
    latest_obs = self.get_latest_observation()

    if self.observed_num == 1:  # 第一次观测
        self.pcd = latest_obs.pcd
        self.bbox = latest_obs.bbox
        self.clip_ft = latest_obs.clip_ft
        self.class_id = latest_obs.class_id
        self.is_low_mobility = latest_obs.is_low_mobility
        return

    # 分裂检测更新
    self.update_split_info(latest_obs)
    if self.should_split:
        return

    # 合并点云（直接拼接）
    self.pcd += latest_obs.pcd

    # 更新包围盒
    self.bbox = self.pcd.get_axis_aligned_bounding_box()

    # 合并 CLIP 特征（加权平均）
    self.clip_ft = (self.clip_ft * (self.observed_num - 1) + latest_obs.clip_ft) / self.observed_num
    self.clip_ft /= np.linalg.norm(self.clip_ft)

    # 贝叶斯类别概率更新
    self.update_class_probs()

    # 多数投票类别
    class_ids = [obs.class_id for obs in self.observations]
    self.class_id = Counter(class_ids).most_common(1)[0][0]

    # 移动性多数投票
    self.is_low_mobility = sum(obs.is_low_mobility for obs in self.observations) > len(self.observations) / 2

    # 周期性点云下采样
    if self.observed_num % 3 == 0:
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
        if self.is_low_mobility:
            self.major_plane_info = self.find_major_plane_info()  # Z 轴直方图峰值
```

### 6.4 分裂检测

**函数**: `update_split_info`

解决的问题：当两个不同物体（如沙发+靠枕）在检测中被频繁交替标记为同一个 LocalObject 时，需要将其分裂。

```python
# 维护 split_info: {class_id: deque[frame_idx]}
# 例如: {5: deque([10,12,14]), 8: deque([11,13,15])}
# 表示这个物体在帧10/12/14被检测为class 5，在帧11/13/15被检测为class 8

# 如果两个类别的共现帧数 > max_common_th(6)，标记需要分裂
if self.max_common > self._cfg.max_common_th:
    self.should_split = True
```

分裂后根据各观测的 class_id 将原物体拆成两个新 LocalObject。

### 6.5 物体生命周期状态机

**函数**: `LocalObject.update_status`

```
                    ┌──────────────────────────────────────┐
                    │    滑动窗口内（最近10帧有观测）         │
                    │                                      │
                    │         UPDATING                     │
                    │    (pending_count=0, waiting=0)      │
                    └─────────────┬────────────────────────┘
                                  │ 滑出窗口
                                  ▼
                    ┌──────────────────────────┐
                    │ 稳定性检查 (stability_check)│
                    └─────────┬──────┬─────────┘
                    不稳定    │      │ 稳定
                              ▼      ▼
                    ┌──────────┐ ┌────────────┐
                    │ PENDING  │ │ WAITING    │
                    │ (count++)│ │ (count++)  │
                    └────┬─────┘ └──────┬─────┘
                         │ >5帧         │ >5帧
                         ▼              ▼
                    ┌──────────┐  ┌─────┴──────────┐
                    │ELIMINATION│  │ is_low_mobility?│
                    │(不稳定丢弃)│  └──┬──────┬──────┘
                    └──────────┘     │ Yes  │ No
                                     ▼      ▼
                              ┌──────────┐ ┌──────────────┐
                              │LM_ELIM   │ │HM_ELIM       │
                              │→全局地图  │ │→丢弃(或保留) │
                              └──────────┘ └──────────────┘
```

**稳定性检查** (`stability_check`):

```python
# 条件1: 观测次数 ≥ 8
if self.observed_num < 8:
    self.is_stable = False; return

# 条件2: 主类别占比 > 1/3
if most_common_count > self.observed_num / 3:
    self.is_stable = True; return

# 条件3: 贝叶斯类别概率收敛
if self.is_class_converged():
    self.is_stable = True; return
```

`is_class_converged` 检查三个指标之一满足即可：

- 最大类别概率 > 0.50
- 类别概率分布的熵 < 0.2
- 最近 3 次概率变化率 < 0.2

### 6.6 状态动作与全局观测生成

**函数**: `status_actions`

当物体进入 `LM_ELIMINATION` 状态时：

```python
if obj.status == LocalObjStatus.LM_ELIMINATION:
    related_objs = self.get_related_objects(obj.uid)   # 从图中查询关联物体

    if len(related_objs) == 0:
        # 无关联物体 → 直接生成全局观测
        global_obs = self.create_global_observation(obj)
        self.global_observations.append(global_obs)
        self.to_be_eliminated.add(obj.uid)
    else:
        # 有关联物体 → 等待关联物体也就绪后一起生成
        if all_related_ready:
            global_obs = self.create_global_observation(obj, related_objs=related_objs)
            self.global_observations.append(global_obs)
            self.to_be_eliminated.add(obj.uid)
            for r in related_objs:
                self.to_be_eliminated.add(r.uid)
```

### 6.7 On-Relation 检测

**函数**: `on_relation_check`

检测"物体 A 在物体 B 的表面上"的关系（如杯子在桌子上）：

```python
def on_relation_check(self, base_obj, test_obj):
    # 前提: 恰好一个有 major_plane_info（即一个是平面物体如桌子），另一个没有
    if base_obj.major_plane_info is None and test_obj.major_plane_info is None:
        return False
    if base_obj.major_plane_info is not None and test_obj.major_plane_info is not None:
        return False

    # 确保 base_obj 是有平面信息的那个（桌子）
    if base_obj.major_plane_info is None:
        base_obj, test_obj = test_obj, base_obj

    # 检查 XY 投影重叠率
    overlap_ratio = compute_xy_overlap(base_obj.bbox, test_obj.bbox)
    if overlap_ratio < 0.8:
        return False

    # 检查 test_obj 的底面 Z 是否接近 base_obj 的主平面 Z（容差 0.1m）
    if not (test_min_z - 0.1 <= base_obj.major_plane_info <= test_min_z + 0.2):
        return False

    return True
```

**`major_plane_info`** 是什么？ 对低移动性物体，对其点云 Z 坐标做直方图，峰值对应的 Z 就是"主平面"高度。例如桌子的主平面就是桌面高度。

### 6.8 创建全局观测

**函数**: `create_global_observation`

```python
def create_global_observation(self, obj, related_objs=[]):
    curr_obs = GlobalObservation()
    curr_obs.uid = obj.uid
    curr_obs.class_id = obj.class_id
    curr_obs.pcd = obj.pcd                               # 3D 点云
    curr_obs.bbox = obj.pcd.get_axis_aligned_bounding_box()  # 3D AABB
    curr_obs.clip_ft = obj.clip_ft                       # 512D CLIP 特征

    # 生成 2D 投影（俯视图）
    pcd_2d = obj.voxel_downsample_2d(obj.pcd, voxel_size=0.02)
    curr_obs.pcd_2d = pcd_2d          # 只保留 XY，Z 设为 floor_height
    curr_obs.bbox_2d = pcd_2d.get_axis_aligned_bounding_box()

    # 附带关联物体信息（如桌子上的杯子）
    for related_obj in related_objs:
        curr_obs.related_objs.append(related_obj.clip_ft)    # CLIP 特征
        curr_obs.related_bbox.append(related_obj.bbox)       # 包围盒
        curr_obs.related_color.append(related_obj.class_id)  # 类别 ID

    return curr_obs
```

**`voxel_downsample_2d`** (`object.py`): 将 3D 点云投影到 XY 平面，做 2D 体素下采样，Z 坐标统一设为 `floor_height`。生成的 `pcd_2d` 用于俯视图导航规划。

---

## 7. 阶段六：全局地图管理（GlobalMapManager）

**文件**: `utils/global_map_manager.py`

全局地图是持久化的语义地图层。

### 7.1 全局匹配

```python
def process_observations(self, curr_observations):
    if not self.is_initialized:
        self.global_map = self.init_from_observation(curr_observations)
        self.is_initialized = True
        self._update_semantic_outputs()   # 写 JSON
        return

    # 全局匹配
    self.tracker.set_current_frame(curr_observations)
    self.tracker.set_ref_map(self.global_map)
    self.tracker.matching_map()          # 全局模式: 只用几何匹配

    curr_observations = self.tracker.get_current_frame()
    self.update_global_map(curr_observations)

    self._update_semantic_outputs()      # 每次更新后写 JSON
```

**全局匹配与局部匹配的区别**: 全局匹配只使用 2D 包围盒交集比率（不用 CLIP），因为进入全局地图的物体已经经过局部地图的充分验证。

```python
# tracker.py:compute_global_spatial_sim
def compute_global_spatial_sim(self):
    # 提取所有全局物体和当前观测的 2D bbox (min_x, min_y, max_x, max_y)
    ratio = self.compute_match_by_intersection_ratio(map_bbox_torch, curr_bbox_torch)
    return ratio

# 交集比率: intersection_area / min(area_a, area_b)
# 阈值: object_tracking.max_similarity = 0.8
```

### 7.2 全局地图更新

```python
def update_global_map(self, curr_observations):
    for obs in curr_observations:
        if obs.matched_obj_idx == -1:
            # 新全局物体
            global_obj = GlobalObject()
            global_obj.add_observation(obs)
            global_obj.update_info()
            self.global_map.append(global_obj)
        else:
            # 更新已有全局物体
            matched_obj = self.global_map[obs.matched_obj_idx]
            matched_obj.add_observation(obs)
            matched_obj.update_info()
```

**GlobalObject.update_info** (`object.py`):

```python
def update_info(self):
    latest_obs = self.get_latest_observation()

    if self.observed_num == 1:
        # 首次：直接赋值
        self.uid = latest_obs.uid
        self.pcd = latest_obs.pcd
        self.pcd_2d = latest_obs.pcd_2d
        self.bbox_2d = latest_obs.bbox_2d
        self.clip_ft = latest_obs.clip_ft
        self.class_id = latest_obs.class_id
        self.related_objs = latest_obs.related_objs
        return

    # 合并 3D 点云
    self.pcd += latest_obs.pcd
    self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
    self.bbox = self.pcd.get_axis_aligned_bounding_box()

    # 合并 2D 投影
    self.pcd_2d += latest_obs.pcd_2d
    self.pcd_2d = self.voxel_downsample_2d(pcd=self.pcd_2d, voxel_size=0.02)
    self.bbox_2d = self.pcd_2d.get_axis_aligned_bounding_box()

    # 特征更新: 点数多的观测优先
    if original_num < incoming_num:
        self.class_id = latest_obs.class_id
        self.clip_ft = latest_obs.clip_ft

    # 累积关联物体
    self.related_objs += latest_obs.related_objs
    self.related_bbox += latest_obs.related_bbox
    self.related_color += latest_obs.related_color
```

---

## 8. 阶段七：持久化保存

### 8.1 实时保存：语义地图 JSON

**文件**: `utils/semantic_map_manager.py` (DualMap_ROS 新增)

每次 `process_observations` 完成后立即调用 `_update_semantic_outputs()`：

```python
def _update_semantic_outputs(self):
    self.semantic_map_manager.update_from_global_map(self.global_map)
    self.spatial_relation_graph.update_from_global_map(self.global_map)
```

**`update_from_global_map`** 的实现：

```python
def update_from_global_map(self, global_map):
    current_uids = set()
    for obj in global_map:
        uid_str = str(obj.uid)
        current_uids.add(uid_str)
        entry = self._build_entry(obj)     # 构建 JSON 条目
        self._entries[uid_str] = entry     # 新增或替换

    # 删除已不在全局地图中的条目
    stale_uids = set(self._entries.keys()) - current_uids
    for uid in stale_uids:
        del self._entries[uid]

    self._save_to_disk()                   # 原子性写入
```

**`_build_entry`** 提取的信息：

```python
entry = {
    "instance_id": str(obj.uid),        # UUID
    "class_name": "chair",              # 语义类别名
    "class_id": 5,                      # 类别索引
    "bbox_3d": {
        "center": [1.2, 3.4, 0.5],     # 3D 包围盒中心（世界坐标）
        "extent": [0.6, 0.6, 0.9],     # 3D 包围盒尺寸 (长/宽/高 米)
        "min_bound": [...],
        "max_bound": [...]
    },
    "bbox_2d": {
        "center": [1.2, 3.4, 0.0],     # 俯视图包围盒中心
        "extent": [0.6, 0.6, 0.0]
    },
    "num_points_3d": 1234,              # 3D 点云点数
    "num_points_2d": 456,               # 2D 投影点数
    "observed_num": 12,                 # 累计观测次数
    "related_objects": [                # 关联物体（on-relation）
        {"class_name": "laptop", "class_id": 8, "bbox_center": [...]}
    ],
    "last_updated": "2026-03-24T10:30:00"
}
```

**原子性写入**: 先写 `.tmp` 文件，再 `os.replace()` 重命名，确保即使进程崩溃也不会留下损坏的 JSON 文件。

### 8.2 实时保存：空间关系图 JSON

**文件**: `utils/spatial_relation_graph.py` (DualMap_ROS 新增)

```python
def update_from_global_map(self, global_map):
    # 重建节点
    for obj in global_map:
        self._nodes[str(obj.uid)] = self._build_node(obj)

    # 两两计算空间关系
    for i, j in combinations(range(len(obj_list)), 2):
        relations = self._compute_relations(obj_list[i], obj_list[j])
        for rel_type, properties in relations:
            self._edges.append(self._build_edge(...))

    self._save_to_disk()
```

**空间关系计算** (`_compute_relations`) 的五种关系：

| 关系           | 判定条件                                                |
| -------------- | ------------------------------------------------------- |
| `on_top_of`  | A 底面 Z ≈ B 顶面 Z（容差 0.15m） 且 XY 投影重叠 > 30% |
| `same_level` | 两者底面 Z 差 < 0.10m                                   |
| `near`       | 水平距离 (XY) < 1.0m                                    |
| `adjacent`   | 包围盒最小间隙 < 0.15m                                  |
| `contains`   | 交集体积 / 内部体积 > 70%                               |

### 8.3 结束时保存：Pickle 文件

**函数**: `core.py:end_process` → `global_map_manager.py:save_map`

```python
def end_process(self):
    self.stop_threading()

    # 清空局部地图：逐步推进状态机直到所有物体被处理
    for i in range(active_window_size + max_pending_count + 1):
        self.local_map_manager.end_process()     # 推进状态
        global_obs_list = self.local_map_manager.get_global_observations()
        self.global_map_manager.process_observations(global_obs_list)

    # 保存
    if self.cfg.save_global_map:
        self.global_map_manager.save_map()
    if self.cfg.save_layout:
        self.detector.save_layout()
```

**`save_map`** 实现:

```python
def save_map(self):
    save_dir = self.cfg.map_save_path
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)       # 清空旧目录
    os.makedirs(save_dir)

    for i, obj in enumerate(self.global_map):
        obj.save_path = obj._initialize_save_path()   # {save_dir}/{uid}.pkl
        obj.save_to_disk()                              # pickle.dump(self, f)

    self._update_semantic_outputs()    # 最终版 JSON
```

**`save_to_disk`** (`object.py`):

```python
def save_to_disk(self):
    with open(self.save_path, "wb") as f:
        pickle.dump(self, f)
```

实际序列化内容由 `__getstate__` 控制：

```python
def __getstate__(self):    # BaseObject
    return {
        "uid": self.uid,
        "pcd_points": np.asarray(self.pcd.points).tolist(),
        "pcd_colors": np.asarray(self.pcd.colors).tolist(),
        "clip_ft": self.clip_ft.tolist(),
        "class_id": self.class_id,
        "nav_goal": self.nav_goal,
    }

def __getstate__(self):    # GlobalObject 追加
    state = super().__getstate__()
    state["pcd_2d_points"] = np.asarray(self.pcd_2d.points).tolist()
    state["pcd_2d_colors"] = np.asarray(self.pcd_2d.colors).tolist()
    state["related_objs"] = [arr.tolist() for arr in self.related_objs]
    state["related_bbox"] = [{"min_bound": ..., "max_bound": ...} for bbox in self.related_bbox]
    state["related_color"] = self.related_color
    return state
```

### 8.4 布局点云保存

**函数**: `object_detector.py:save_layout`

```python
def save_layout(self):
    layout_pcd = self.get_layout_pointcloud()
    o3d.io.write_point_cloud(save_dir + "/layout.pcd", layout_pcd)
```

`layout.pcd` 的生成过程是在独立后台线程中完成的：

- 每当相机移动 >1m 或旋转 >20°，将当前深度图（16 倍降采样）反投影为世界坐标系点云
- 与已累积的布局点云合并
- 做 `voxel_down_sample(0.05m)` 控制规模

### 8.5 全部输出文件总结

| 文件                            | 保存时机         | 内容                            | 格式   |
| ------------------------------- | ---------------- | ------------------------------- | ------ |
| `semantic/semantic_map.json`  | 每次全局地图更新 | 所有物体的坐标/尺寸/类别/实例ID | JSON   |
| `semantic/spatial_graph.json` | 每次全局地图更新 | 物体间空间关系图(nodes+edges)   | JSON   |
| `map/{uid}.pkl`               | 结束时           | 每个全局物体的完整序列化数据    | Pickle |
| `map/layout.pcd`              | 结束时           | 场景几何布局点云                | PCD    |
| `map/wall.pcd`                | 结束时(如启用)   | 提取的墙壁障碍物点云            | PCD    |
| `path/global_path/{n}.json`   | 路径规划时       | 全局导航路径 [[x,y,z],...]      | JSON   |
| `path/local_path/{n}.json`    | 路径规划时       | 局部避障路径                    | JSON   |
| `path/action_path/{n}.json`   | 路径规划时       | 合并执行路径                    | JSON   |
| `system_time.csv`             | 结束时           | 各阶段平均耗时统计              | CSV    |
| `log/*.log`                   | 全程             | 运行日志                        | LOG    |

---

## 9. 并行线程模型

`use_parallel: true` 时的线程分布：

```
┌───────────────────────────────────────────────────────────────┐
│ 主线程 (ROS 定时器驱动)                                        │
│                                                               │
│   run_once() → check_keyframe() → parallel_process()          │
│     │                                                         │
│     ├── Detector: YOLO+SAM (主线程)  ←并行→  FastSAM (子线程)  │
│     ├── Detector: 3D点云 (子线程)    ←并行→  CLIP (主线程)     │
│     ├── 结果入队 → detection_results_queue                     │
│     ├── 导航路径规划 (使用上一帧的地图结果)                      │
│     └── Rerun 可视化 + ROS 发布                                │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│ 映射线程 (后台守护线程)                                         │
│                                                               │
│   run_mapping_thread():                                       │
│     while not stop:                                           │
│       obs, frame_id = queue.get(timeout=1)                    │
│       local_map_manager.process_observations(obs)             │
│       global_obs = local_map_manager.get_global_observations()│
│       global_map_manager.process_observations(global_obs)     │
│       → 触发 semantic_map.json + spatial_graph.json 写入      │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│ 布局点云线程 (后台线程，每帧创建)                                │
│                                                               │
│   _process_data_input_thread():                               │
│     if 移动>1m or 旋转>20°:                                   │
│       depth → 3D点云 → 累积到 layout_pointcloud               │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│ 配置监控线程 (后台守护线程)                                      │
│                                                               │
│   monitor_config_file():                                      │
│     while not stop:                                           │
│       OmegaConf.load(actions.yaml)                            │
│       if calculate_path → self.calculate_path = True          │
│       if inquiry_sentence → self.inquiry = "..."              │
│       sleep(0.1)                                              │
└───────────────────────────────────────────────────────────────┘
```

**队列**: `detection_results_queue = Queue(maxsize=10)`。如果映射线程处理慢导致队列满，主线程会跳过该帧（`put(timeout=1)` 超时不阻塞）。

---

## 10. 附录：关键数据结构字段速查

### DataInput

| 字段           | 类型           | 含义                 |
| -------------- | -------------- | -------------------- |
| `idx`        | int            | 关键帧序号           |
| `time_stamp` | float          | ROS 时间戳(秒)       |
| `color`      | ndarray(H,W,3) | RGB 图像 uint8       |
| `depth`      | ndarray(H,W,1) | 深度图 float32 米    |
| `intrinsics` | ndarray(3,3)   | 相机内参矩阵         |
| `pose`       | ndarray(4,4)   | 世界坐标系位姿 SE(3) |

### LocalObservation

| 字段                | 类型         | 含义                      |
| ------------------- | ------------ | ------------------------- |
| `idx`             | int          | 帧号                      |
| `class_id`        | int          | 类别索引                  |
| `conf`            | float        | YOLO 置信度               |
| `pcd`             | PointCloud   | 世界坐标系 3D 点云        |
| `bbox`            | AABB         | 3D 包围盒                 |
| `clip_ft`         | ndarray(512) | CLIP 加权特征             |
| `is_low_mobility` | bool         | 低移动性标记              |
| `distance`        | float        | 到相机距离(米)            |
| `matched_obj_idx` | int          | 匹配的地图物体索引(-1=新) |

### LocalObject

| 字段                 | 类型           | 含义               |
| -------------------- | -------------- | ------------------ |
| `uid`              | UUID           | 唯一标识           |
| `pcd`              | PointCloud     | 累积合并的 3D 点云 |
| `bbox`             | AABB           | 当前 3D 包围盒     |
| `clip_ft`          | ndarray(512)   | 加权平均 CLIP 特征 |
| `class_id`         | int            | 多数投票类别       |
| `observed_num`     | int            | 观测次数           |
| `status`           | LocalObjStatus | 生命周期状态       |
| `is_stable`        | bool           | 是否稳定           |
| `is_low_mobility`  | bool           | 低移动性           |
| `class_probs`      | ndarray(C)     | 贝叶斯类别概率分布 |
| `major_plane_info` | float          | 主平面 Z 值        |
| `split_info`       | dict           | 分裂检测信息       |

### GlobalObservation

| 字段              | 类型          | 含义                    |
| ----------------- | ------------- | ----------------------- |
| `uid`           | UUID          | 来源 LocalObject 的 UID |
| `pcd`           | PointCloud    | 3D 点云                 |
| `bbox`          | AABB          | 3D 包围盒               |
| `pcd_2d`        | PointCloud    | 俯视图 2D 投影点云      |
| `bbox_2d`       | AABB          | 俯视图 2D 包围盒        |
| `clip_ft`       | ndarray(512)  | CLIP 特征               |
| `class_id`      | int           | 类别                    |
| `related_objs`  | list[ndarray] | 关联物体 CLIP 特征列表  |
| `related_bbox`  | list[AABB]    | 关联物体包围盒          |
| `related_color` | list[int]     | 关联物体类别 ID         |

### GlobalObject

继承 BaseObject 的所有字段，额外添加：

| 字段              | 类型          | 含义                        |
| ----------------- | ------------- | --------------------------- |
| `pcd_2d`        | PointCloud    | 俯视图投影 (Z=floor_height) |
| `bbox_2d`       | AABB          | 俯视图包围盒                |
| `related_objs`  | list[ndarray] | 累积的关联物体 CLIP 特征    |
| `related_bbox`  | list[AABB]    | 累积的关联物体包围盒        |
| `related_color` | list[int]     | 累积的关联物体类别 ID       |
