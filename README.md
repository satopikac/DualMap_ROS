# DualMap_ROS - 语义地图构建系统 (ROS 专用版)

本项目从 DualMap 中提取了 ROS 数据处理相关代码，移除了离线数据集处理和 iPhone Record3D 模式，专注于 ROS1/ROS2 实时语义建图。

## 相比原版 DualMap 的改动

### 1. 仅保留 ROS 处理流程
- 移除了 `runner_dataset.py`、`runner_record_3d.py`、`offline_local_map_query.py` 等离线模式
- 移除了 `utils/dataset.py` 等离线数据集基类
- 保留了完整的 ROS1/ROS2 运行器和消息发布系统

### 2. 新增: 语义地图 JSON 实时保存 (`utils/semantic_map_manager.py`)
在全局地图更新时，自动将所有已识别物体的信息实时保存为 JSON 格式：

**输出文件**: `{output_path}/{scene}/semantic/semantic_map.json`

**JSON 结构示例**:
```json
{
  "metadata": {
    "total_objects": 15,
    "last_updated": "2026-03-24T10:30:00"
  },
  "objects": [
    {
      "instance_id": "550e8400-e29b-41d4-a716-446655440000",
      "class_name": "chair",
      "class_id": 5,
      "bbox_3d": {
        "center": [1.2, 3.4, 0.5],
        "extent": [0.6, 0.6, 0.9],
        "min_bound": [0.9, 3.1, 0.05],
        "max_bound": [1.5, 3.7, 0.95]
      },
      "bbox_2d": {
        "center": [1.2, 3.4, 0.0],
        "extent": [0.6, 0.6, 0.0]
      },
      "num_points_3d": 1234,
      "num_points_2d": 456,
      "observed_num": 12,
      "is_nav_goal": false,
      "related_objects": [
        {
          "class_name": "laptop",
          "class_id": 8,
          "bbox_center": [1.3, 3.5, 1.0],
          "bbox_extent": [0.3, 0.2, 0.05]
        }
      ],
      "last_updated": "2026-03-24T10:30:00"
    }
  ]
}
```

**实时更新机制**:
- 每次全局地图更新后自动同步
- 物体被匹配更新时，替换对应 JSON 条目
- 物体不再存在于全局地图中时，自动删除条目
- 原子性写入（先写临时文件再重命名）

### 3. 新增: 空间关系图 JSON 实时保存 (`utils/spatial_relation_graph.py`)
使用图结构描述已识别物体间的空间联系：

**输出文件**: `{output_path}/{scene}/semantic/spatial_graph.json`

**支持的空间关系类型**:

| 关系类型 | 描述 | 判定条件 |
|----------|------|----------|
| `on_top_of` | A 在 B 上方 | A 底面 Z ≈ B 顶面 Z，且 XY 投影重叠 > 30% |
| `near` | A 与 B 相近 | 水平距离 < 1.0m |
| `adjacent` | A 与 B 紧邻 | 包围盒间隙 < 0.15m |
| `same_level` | 同一水平面 | 底面 Z 差 < 0.10m |
| `contains` | A 包含 B | 包含体积比 > 70% |

**JSON 结构示例**:
```json
{
  "metadata": {
    "total_nodes": 15,
    "total_edges": 28,
    "relation_counts": {
      "near": 12,
      "on_top_of": 5,
      "same_level": 8,
      "adjacent": 3
    },
    "last_updated": "2026-03-24T10:30:00"
  },
  "nodes": [
    {
      "instance_id": "uuid-1",
      "class_name": "table",
      "class_id": 3,
      "position": [1.0, 2.0, 0.4],
      "extent": [1.2, 0.8, 0.8],
      "min_bound": [0.4, 1.6, 0.0],
      "max_bound": [1.6, 2.4, 0.8]
    }
  ],
  "edges": [
    {
      "source": "uuid-1",
      "source_class": "table",
      "target": "uuid-2",
      "target_class": "laptop",
      "relation": "on_top_of",
      "properties": {
        "z_gap": 0.02,
        "xy_overlap": 0.85,
        "description": "laptop is on top of table"
      }
    }
  ]
}
```

---

## 项目结构

```
DualMap_ROS/
├── dualmap/
│   ├── __init__.py
│   └── core.py                     # 主编排类 (已修改: 集成语义管理器)
├── applications/
│   ├── __init__.py
│   ├── runner_ros.py               # ROS 入口 (自动检测 ROS1/ROS2)
│   └── utils/
│       ├── runner_ros_base.py      # ROS 运行器基类
│       ├── runner_ros1.py          # ROS1 运行器
│       ├── runner_ros2.py          # ROS2 运行器
│       └── ros_publisher.py        # ROS 消息发布
├── utils/
│   ├── types.py                    # 数据结构
│   ├── object.py                   # 物体类
│   ├── object_detector.py          # 检测管线
│   ├── tracker.py                  # 匹配追踪器
│   ├── local_map_manager.py        # 局部地图管理
│   ├── global_map_manager.py       # 全局地图管理 (已修改: 集成语义输出)
│   ├── base_map_manager.py         # 地图管理基类
│   ├── semantic_map_manager.py     # [NEW] 语义地图 JSON 管理
│   ├── spatial_relation_graph.py   # [NEW] 空间关系图 JSON 管理
│   ├── navigation_helper.py        # 导航辅助
│   ├── pcd_utils.py                # 点云工具
│   ├── visualizer.py               # Rerun 可视化
│   ├── logging_helper.py           # 日志配置
│   └── time_utils.py               # 计时工具
├── config/                         # 配置文件 (同原版)
└── README.md                       # 本文件
```

## 运行方式

```bash
conda activate dualmap
source /opt/ros/humble/setup.bash  # ROS2

python applications/runner_ros.py
```

## 输出文件

| 文件 | 路径 | 说明 |
|------|------|------|
| `semantic_map.json` | `{output}/semantic/` | 语义地图 (实时更新) |
| `spatial_graph.json` | `{output}/semantic/` | 空间关系图 (实时更新) |
| `*.pkl` | `{output}/map/` | 物体 Pickle 文件 |
| `layout.pcd` | `{output}/map/` | 布局点云 |
| `wall.pcd` | `{output}/map/` | 墙壁点云 |
