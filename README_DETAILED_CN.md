# DualMap - 双层语义地图构建系统 (详细技术文档)

## 项目概述

DualMap 是一个基于 RGB-D 视觉感知的实时语义地图构建系统，支持 ROS1/ROS2 和离线数据集两种运行模式。系统通过双层地图架构（局部地图 + 全局地图）实现物体级别的语义建图、物体跟踪匹配和语义导航规划。

### 核心能力
- **物体检测与分割**：YOLO + SAM + FastSAM 组合检测管线
- **视觉语义特征**：CLIP/MobileCLIP 视觉-语言特征提取
- **双层地图架构**：局部地图（滑动窗口跟踪） → 全局地图（语义锚点聚合）
- **物体生命周期管理**：创建 → 跟踪 → 稳定性判定 → 移动性分类 → 提升/消除
- **语义导航**：基于全局地图的路径规划，支持随机/点击/语义查询三种目标模式
- **实时可视化**：Rerun 3D 可视化 + ROS RViz 可视化

---

## 项目结构

```
DualMap/
├── dualmap/                        # 核心映射引擎
│   ├── __init__.py
│   └── core.py                     # Dualmap 主类，系统编排中心 (797行)
│
├── applications/                   # 应用层入口
│   ├── __init__.py
│   ├── runner_ros.py               # ROS 版本自动检测入口 (47行)
│   ├── runner_dataset.py           # 离线数据集处理入口
│   ├── runner_record_3d.py         # Record3D iPhone 数据模式
│   ├── offline_local_map_query.py  # 离线地图语义查询工具
│   ├── generate_replica_class_color.py  # Replica 数据集类别颜色生成
│   └── utils/
│       ├── runner_ros_base.py      # ROS 运行器基类（共享逻辑, 197行）
│       ├── runner_ros1.py          # ROS1 运行器（rospy, 149行）
│       ├── runner_ros2.py          # ROS2 运行器（rclpy, 158行）
│       └── ros_publisher.py        # ROS 消息发布器（ROS2, 295行）
│
├── utils/                          # 核心工具模块
│   ├── types.py                    # 数据结构定义 (DataInput, Observation, ObjectClasses 等)
│   ├── object.py                   # 物体类 (BaseObject, LocalObject, GlobalObject)
│   ├── object_detector.py          # 检测管线 (YOLO + SAM + CLIP, 600+行)
│   ├── tracker.py                  # 匹配追踪器（空间+视觉相似度, 525行）
│   ├── local_map_manager.py        # 局部地图管理器 (550+行)
│   ├── global_map_manager.py       # 全局地图管理器 (450+行)
│   ├── base_map_manager.py         # 地图管理器基类 (25行)
│   ├── navigation_helper.py        # 导航辅助 (LayoutMap + NavigationGraph)
│   ├── pcd_utils.py                # 点云工具 (深度投影、DBSCAN去噪、聚类)
│   ├── visualizer.py               # Rerun 可视化器 (单例模式)
│   ├── dataset.py                  # 数据集基类（离线模式使用）
│   ├── model_utils.py              # 模型工具函数
│   ├── logging_helper.py           # 日志配置
│   ├── time_utils.py               # 性能计时工具
│   └── eval/                       # 评估工具
│
├── config/                         # 配置文件
│   ├── base_config.yaml            # 基础配置（数据集选择、输出路径）
│   ├── system_config.yaml          # 系统配置（模型路径、检测、匹配参数）
│   ├── runner_ros.yaml             # ROS 运行配置（频率、话题、压缩设置）
│   ├── runner_dataset.yaml         # 数据集运行配置
│   ├── actions.yaml                # 运行时动作触发器（路径规划、查询等）
│   ├── class_list/                 # 物体类别列表文件
│   │   ├── gpt_indoor_general.txt  # 通用室内类别
│   │   ├── gpt_indoor_office.txt   # 办公室场景类别
│   │   ├── gpt_outdoor_general.txt # 通用室外类别
│   │   └── scannet200_classes.txt  # ScanNet200 类别集
│   ├── data_config/
│   │   ├── ros/                    # ROS 话题配置
│   │   │   └── self_collected.yaml # 话题名、深度因子、内参
│   │   └── dataset/                # 数据集配置
│   └── support_config/             # 辅助配置
│       ├── demo_config.yaml        # 演示配置
│       ├── mobility_config.yaml    # 移动性分类配置
│       └── logging_config.yaml     # 日志格式配置
│
├── evaluation/                     # 评估脚本
├── model/                          # 预训练模型目录
├── 3rdparty/                       # 第三方库 (MobileCLIP)
├── scripts/                        # 工具脚本
└── resources/                      # 文档与资源
```

---

## 系统架构

### 1. 数据流概览

```
传感器数据 (ROS Topic)
    │
    ├── RGB 图像 ──────┐
    ├── 深度图像 ───────┤─→ 时间同步 (ApproximateTimeSynchronizer)
    └── 里程计位姿 ─────┘         │
                                  ↓
                          DataInput 数据包
                          (图像 + 深度 + 位姿 + 内参)
                                  │
                          ┌───────┴───────┐
                          ↓ (关键帧选择)    ↓ (非关键帧丢弃)
                    Dualmap.process()
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
          物体检测    布局点云累积   可视化
         (Detector)   (layout PCD)  (Rerun)
              │
    ┌─────────┴─────────┐
    ↓                   ↓
YOLO 检测          FastSAM 开放词汇
(已知类别)         (未知类别补充)
    │                   │
    └─────────┬─────────┘
              ↓
        SAM 精细分割
              │
              ↓
    掩膜 + 深度 → 3D 点云
              │
              ↓
      CLIP 特征提取 (512D)
              │
              ↓
    Observation 列表
              │
    ┌─────────┴─────────────┐
    ↓                       ↓
局部地图匹配             (队列传递 - 并行模式)
(LocalMapManager)              │
    │                          ↓
    ├── 物体跟踪           全局地图匹配
    ├── 稳定性检查         (GlobalMapManager)
    ├── 移动性分类              │
    └── 低移动性物体 ──→ GlobalObservation
              │
              ↓
    全局语义地图 + 导航规划
```

### 2. 双层地图架构

#### 局部地图 (Concrete Map / Local Map)
- **时间窗口**：`active_window_size`（默认 10 帧）滑动窗口
- **物体跟踪**：在窗口内通过空间重叠 + CLIP 视觉相似度进行匹配
- **稳定性判定**：需 `stable_num`（默认 8）次观测 + 贝叶斯类别概率收敛
- **生命周期状态机**：UPDATING → PENDING → STABLE → {LM_ELIMINATION, HM_ELIMINATION}
- **输出**：低移动性稳定物体提升为 GlobalObservation 送入全局地图

#### 全局地图 (Abstract Map / Global Map)
- **聚合方式**：从局部地图接收稳定物体，全局空间匹配后合并或新增
- **匹配策略**：2D 包围盒交集比率（纯几何匹配，阈值 0.8）
- **2D 投影**：每个全局物体维护 `pcd_2d`（俯视图投影）用于导航规划
- **关联物体**：维护 related_objs（on-relation 检测的关联物体 CLIP 特征）
- **持久化**：Pickle (.pkl) 格式序列化保存/加载

### 3. 物体数据结构层次

```
BaseObject
├── uid               # UUID 唯一标识
├── pcd               # Open3D 3D 点云 (PointCloud)
├── bbox              # 轴对齐包围盒 (AxisAlignedBoundingBox)
├── clip_ft           # CLIP 特征向量 (np.ndarray, 512D)
├── class_id          # 类别 ID (int)
├── observed_num      # 累计观测次数
├── observations[]    # 观测历史列表
├── nav_goal          # 是否为导航目标
├── save_path         # 磁盘保存路径 ({map_save_path}/{uid}.pkl)
├── save_to_disk()    # Pickle 序列化保存
└── load_from_disk()  # Pickle 反序列化加载

LocalObject (继承 BaseObject)
├── is_low_mobility   # 低/高移动性标记 (bool)
├── status            # 生命周期状态 (LocalObjStatus 枚举)
├── class_probs       # 贝叶斯类别概率分布 (np.ndarray)
├── class_probs_history # 概率历史（滑动窗口）
├── split_info        # 分裂检测信息 (dict: class_id → deque[frame_idx])
├── should_split      # 是否应分裂标记
├── major_plane_info  # 主平面 Z 值（用于 on-relation 检测）
├── is_stable         # 稳定性标记
├── pending_count     # 等待计数
└── waiting_count     # 稳定后等待计数

GlobalObject (继承 BaseObject)
├── pcd_2d            # 2D 投影点云（俯视图, PointCloud）
├── bbox_2d           # 2D 包围盒 (AxisAlignedBoundingBox)
├── related_objs[]    # 关联物体 CLIP 特征列表 (List[np.ndarray])
├── related_bbox[]    # 关联物体包围盒列表
└── related_color[]   # 关联物体颜色类别 ID 列表
```

### 4. 坐标变换链

```
世界坐标系位姿 = world_transform × (odom_pose × extrinsics)

其中：
- world_transform: 从 config 的 roll/pitch/yaw 构建的 SE(3) 旋转矩阵
- odom_pose:       来自 ROS Odometry 话题（四元数 + 平移 → 4×4 变换矩阵）
- extrinsics:      相机安装外参（相机坐标系 → 机器人坐标系, 4×4 矩阵）
```

---

## 核心模块详解

### Dualmap (dualmap/core.py)
系统编排中心，负责：
- 初始化所有子模块（Detector, LocalMapManager, GlobalMapManager, Visualizer）
- **关键帧选择**：基于时间间隔(0.5s)、平移距离(0.1m)、旋转角度(3°)三重阈值
- **顺序处理** (`sequential_process`): 检测→局部地图→全局地图，单线程
- **并行处理** (`parallel_process`): 检测在主线程，映射在后台线程（队列解耦）
- **导航路径计算**：全局路径 + 局部路径 → 合并为动作路径
- **配置文件监控**：后台线程每 0.1s 监控 actions.yaml，支持运行时触发路径规划
- **结束处理**：清空局部地图、保存全局地图、输出计时统计

### Detector (utils/object_detector.py)
检测管线 (600+ 行)，职责：
- **YOLO 检测**：输入 RGB → 输出边界框 + 类别 + 置信度
- **SAM 分割**：输入边界框 → 输出精细掩膜
- **FastSAM 开放词汇**：对未被 YOLO 覆盖的区域进行补充检测
- **CLIP 特征**：输入裁剪图像 → 512D 语义向量 (image_weight=0.7 加权)
- **3D 点云生成**：掩膜 × 深度 → 相机坐标系 3D 点 → 世界坐标系点云
- **移动性分类**：CLIP 文本原型相似度判定（低/高移动性）
- **布局点云**：后台线程累积关键帧的场景几何点云
- **位姿低通滤波**：PoseLowPassFilter 平滑位姿抖动

### LocalMapManager (utils/local_map_manager.py)
局部地图管理 (550+ 行)，职责：
- **观测匹配**：Tracker 计算空间+视觉相似度 → 匈牙利分配
- **物体更新**：合并点云、贝叶斯更新类别概率、周期性下采样
- **分裂检测**：同一物体被多个类别标记 → 自动分裂为两个物体
- **稳定性检查**：观测次数 ≥ stable_num 且满足以下之一：
  - 主类别占比 > 1/3
  - 最大类别概率 > 0.50
  - 类别概率熵 < 0.2
  - 概率变化率 < 0.2
- **On-Relation 检测**：物体 A 在物体 B 表面上（Z 轴几何分析）
- **全局观测提取**：稳定的低移动性物体 → GlobalObservation（含 2D 投影）
- **局部路径规划**：基于局部物体的避障规划
- **NetworkX 图**：管理物体间关系的无向图

### GlobalMapManager (utils/global_map_manager.py)
全局地图管理 (450+ 行)，职责：
- **全局匹配**：2D 包围盒交集比率 > 0.8 → 匹配更新，否则新增
- **物体更新**：合并 3D 点云、2D 投影、关联物体
- **导航图构建**：布局点云 → 占用栅格 → Voronoi 自由空间图
- **路径规划**：Dijkstra 最短路径，支持三种目标模式
- **语义查询**：CLIP 特征余弦相似度搜索全局物体
- **地图保存/加载**：.pkl 文件目录读写
- **墙壁处理**：wall.pcd 加载与生成

### Tracker (utils/tracker.py)
匹配追踪器 (525 行)：
- **局部空间相似度**：FAISS 加速的点云重叠率计算（先 3D IoU 预筛选）
- **局部视觉相似度**：CLIP 特征余弦相似度矩阵
- **总相似度**：空间 + 视觉，阈值 sim_threshold=1.2
- **全局空间相似度**：2D 包围盒交集比率
- **匹配分配**：贪心最大值分配（非严格匈牙利）
- **合并检测**：连通分量分析用于后处理合并

### NavigationHelper (utils/navigation_helper.py)
导航辅助模块：
- **LayoutMap**：
  - 点云 → 2D 直方图占用栅格
  - 百分位阈值二值化 + 形态学操作
  - 墙壁提取 → 3D 墙壁点云
  - 交互式编辑支持
- **NavigationGraph**：
  - 占用图 → NetworkX 图（8 连通）
  - 自由空间采样 + 目标点吸附
  - Dijkstra 最短路径搜索
  - 路径平滑（锐角消除）

---

## ROS 集成

### 自动版本检测 (runner_ros.py)
系统通过检测 `rclpy`/`rospy` 模块自动判断 ROS 版本：
```python
if importlib.util.find_spec("rclpy"):   → ROS2
elif importlib.util.find_spec("rospy"): → ROS1
```

### 话题订阅
| 话题 | 消息类型 | 用途 |
|------|----------|------|
| `/camera/rgb/image_raw` | `Image` / `CompressedImage` | RGB 图像 |
| `/camera/depth/image_raw` | `Image` / `CompressedImage` | 深度图像 |
| `/camera/pose` | `Odometry` | 里程计位姿 |
| `/camera_info` | `CameraInfo` | 相机内参（备用） |

### 话题发布 (ROS2)
| 话题 | 消息类型 | 内容 |
|------|----------|------|
| `/global_path` | `Path` | 全局导航路径 |
| `/local_path` | `Path` | 局部避障路径 |
| `/action_path` | `Path` | 合并执行路径 |
| `/annotated_image` | `Image` | 检测标注图像 |
| `/odom` | `Odometry` | 变换后位姿 |
| `/local_map/semantic` | `PointCloud2` | 局部语义点云 |
| `/global_map/semantic` | `PointCloud2` | 全局语义点云 |

### 消息同步
使用 `ApproximateTimeSynchronizer` 对 RGB + Depth + Odometry 进行时间同步：
- `queue_size=10`
- `slop=0.1s`（允许的最大时间偏差）

### 深度图处理
支持多种深度格式：
- `uint16`：通常为毫米单位（depth_factor=1000.0）
- `float32/float64`：通常为米单位（depth_factor=1.0）

---

## 配置系统

使用 Hydra + OmegaConf 配置框架，层次化覆盖：

```
runner_ros.yaml (入口)
├── defaults: base_config.yaml      # 数据集、输出路径
├── defaults: system_config.yaml    # 模型、检测、匹配参数
├── defaults: mobility_config.yaml  # 移动性分类
└── ROS 特有覆盖配置               # 频率、话题、并行模式
```

### 关键配置参数

| 分类 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| ROS | `ros_rate` | 15 | 主循环频率 (Hz) |
| ROS | `sync_threshold` | 0.1 | 话题同步时间容差 (s) |
| ROS | `use_compressed_topic` | false | 是否使用压缩话题 |
| 关键帧 | `time_threshold` | 0.5 | 时间间隔阈值 (s) |
| 关键帧 | `pose_threshold` | 0.1 | 平移距离阈值 (m) |
| 关键帧 | `rotation_threshold` | 3 | 旋转角度阈值 (°) |
| 检测 | `fastsam_confidence` | 0.80 | FastSAM 置信度 |
| 检测 | `image_weight` | 0.7 | CLIP 图像特征权重 |
| 检测 | `small_mask_th` | 204 | 小掩膜过滤阈值 |
| 点云 | `downsample_voxel_size` | 0.02 | 下采样体素 (m) |
| 点云 | `dbscan_eps` | 0.1 | DBSCAN 邻域半径 |
| 匹配 | `sim_threshold` | 1.2 | 局部匹配总相似度阈值 |
| 全局匹配 | `object_tracking.max_similarity` | 0.8 | 全局匹配阈值 |
| 局部物体 | `active_window_size` | 10 | 滑动窗口大小 (帧) |
| 局部物体 | `stable_num` | 8 | 稳定判定最少观测 |
| 局部物体 | `max_pending_count` | 5 | 最大等待帧数 |
| 系统 | `use_parallel` | true | 并行处理模式 |
| 系统 | `use_rerun` | true | Rerun 可视化 |

---

## 数据保存格式

| 数据类型 | 格式 | 保存路径 | 触发条件 |
|----------|------|----------|----------|
| 全局地图物体 | Pickle (.pkl) | `{map_save_path}/{uid}.pkl` | `save_global_map: true` |
| 局部地图物体 | Pickle (.pkl) | `{map_save_path}/{uid}.pkl` | `save_local_map: true` |
| 全局导航路径 | JSON | `{output_path}/path/global_path/{n}.json` | `save_all_path: true` |
| 局部导航路径 | JSON | `{output_path}/path/local_path/{n}.json` | `save_all_path: true` |
| 动作路径 | JSON | `{output_path}/path/action_path/{n}.json` | `save_all_path: true` |
| 布局点云 | PCD | `{map_save_path}/layout.pcd` | `save_layout: true` |
| 墙壁点云 | PCD | `{map_save_path}/wall.pcd` | `save_wall: true` |
| 系统计时 | CSV | `{map_save_path}/../system_time.csv` | 结束处理时自动保存 |
| 日志 | LOG | `{output_path}/log/` | 始终保存 |

### 物体序列化内容 (Pickle)
```python
{
    "uid": UUID,                    # 唯一标识
    "pcd_points": [[x,y,z], ...],  # 3D 点云坐标
    "pcd_colors": [[r,g,b], ...],  # 点云颜色 (0~1)
    "clip_ft": [float, ...],       # CLIP 特征 (512D)
    "class_id": int,               # 类别 ID
    "nav_goal": bool,              # 导航目标标记
    # GlobalObject 额外字段:
    "pcd_2d_points": [[x,y,z],...],# 2D 投影点
    "pcd_2d_colors": [[r,g,b],...],# 2D 投影颜色
    "related_objs": [[float,...]],  # 关联物体 CLIP 特征
    "related_bbox": [{"min_bound":[],"max_bound":[]}], # 关联包围盒
    "related_color": [int, ...]    # 关联物体类别 ID
}
```

---

## 线程模型

### 并行模式 (`use_parallel: true`)
```
主线程:
├── 检测处理 (帧 t)
│   ├── YOLO + SAM + CLIP
│   └── 结果入队 → 映射线程
├── 导航路径规划 (使用帧 t-1 的映射结果)
├── Rerun 可视化
└── ROS 消息发布

映射线程 (后台守护线程):
├── 从 detection_results_queue 获取检测结果
├── 局部地图处理 (匹配、跟踪、稳定性)
└── 全局地图处理 (匹配、更新)

配置监控线程 (后台):
└── 每 0.1s 读取 actions.yaml
    ├── calculate_path → 触发全局路径规划
    ├── get_goal_mode → 设置目标模式 (random/click/inquiry)
    ├── inquiry_sentence → 设置语义查询关键词
    └── trigger_find_next → 重新规划

布局点云线程 (后台):
└── 累积关键帧的场景几何点云 (独立于映射流程)
```

---

## 运行方式

### ROS 模式
```bash
# 激活环境
conda activate dualmap
source /opt/ros/humble/setup.bash  # ROS2

# 启动 (自动检测 ROS 版本)
python applications/runner_ros.py
```

### 数据集模式
```bash
python applications/runner_dataset.py
```

### 离线查询
```bash
python applications/offline_local_map_query.py
```

### 运行时控制
编辑 `config/actions.yaml` 即可实时触发功能：
```yaml
calculate_path: true           # 触发路径规划
get_goal_mode: inquiry         # 切换到语义查询模式
inquiry_sentence: "red chair"  # 查找红色椅子
```
