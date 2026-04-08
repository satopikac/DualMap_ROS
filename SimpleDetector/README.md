# SimpleDetector

精简版 RGBD 语义检测工具 — 从 DualMap_ROS 项目简化而来。

## 功能

- 实时订阅 ROS1 的 RGB、Depth、Odom 话题
- 用户按 Enter 手动触发当前帧的语义识别
- **YOLO 检测** → **SAM 分割** → **深度图点云** → **DBSCAN 去噪** → **3D AABB 包围盒**
- 检测结果累积保存到 `output/semantic_map.json`

## 与原 DualMap_ROS 的区别

| 特性 | DualMap_ROS | SimpleDetector |
|------|------------|----------------|
| 文件数 | 15+ 个模块 | 1 个主文件 + 1 个配置 |
| 关键帧选择 | 自动（时间/位移/旋转） | 手动触发 |
| 物体去重/跟踪 | LocalMap + GlobalMap 双层 | 无（每次独立检测） |
| 稳定性/贝叶斯分类 | 有 | 无 |
| CLIP 特征 | 有 | 无 |
| 导航规划 | 有 | 无 |
| 可视化 | ReRun | 终端日志 |

## 目录结构

```
SimpleDetector/
├── simple_detector.py   # 主程序（单文件）
├── config.yaml          # 配置文件
├── output/              # 检测输出
│   └── semantic_map.json
└── README.md
```

## 依赖

```
rospy, cv_bridge, message_filters      # ROS1
sensor_msgs, nav_msgs                  # ROS 消息类型
ultralytics                            # YOLO + SAM
open3d                                 # 点云处理
torch                                  # GPU 加速
numpy, scipy, opencv-python, PyYAML    # 基础库
```

## 使用方法

### 1. 修改配置

编辑 `config.yaml`，设置 ROS 话题名、相机内参、模型路径等。

### 2. 运行

```bash
# 确保 ROS 环境已加载
source /opt/ros/noetic/setup.bash
# 或 source your_ws/devel/setup.bash

# 启动（使用默认配置）
python simple_detector.py

# 或指定配置文件
python simple_detector.py --config my_config.yaml
```

### 3. 操作

- 程序启动后会自动订阅 ROS 话题，实时接收数据
- **按 Enter**: 触发当前帧的语义检测
- **输入 q + Enter**: 退出程序
- 检测结果自动追加到 `output/semantic_map.json`

## 输出格式

`semantic_map.json` 示例：

```json
{
  "metadata": {
    "total_objects": 5,
    "last_updated": "2026-04-08T14:30:00.123456"
  },
  "objects": [
    {
      "instance_id": "a1b2c3d4-...",
      "class_name": "chair",
      "class_id": 5,
      "bbox_3d": {
        "center": [1.23, -0.45, 0.67],
        "extent": [0.50, 0.50, 0.80],
        "min_bound": [0.98, -0.70, 0.27],
        "max_bound": [1.48, -0.20, 1.07]
      },
      "num_points_3d": 342,
      "confidence": 0.87,
      "last_updated": "2026-04-08T14:30:00.123456"
    }
  ]
}
```

## 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ros_topics.rgb` | RGB 图像话题 | `/camera/color/image_raw` |
| `ros_topics.depth` | 深度图话题 | `/camera/aligned_depth_to_color/image_raw` |
| `ros_topics.odom` | 里程计话题 | `/odom` |
| `depth_factor` | 深度缩放因子 | `1000.0` |
| `pcd_sample_ratio` | 点云采样比例 | `0.04` |
| `dbscan_eps` | DBSCAN 邻域半径 | `0.1` |
| `device` | 计算设备 | `cuda` |
