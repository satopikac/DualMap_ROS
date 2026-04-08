# DualMap BBox 计算完整细节

本文档追踪项目中每一处 BBox（包围盒）的创建、更新、使用，覆盖从像素到世界坐标的全过程。

---

## 目录

1. [核心结论](#1-核心结论)
2. [3D BBox 的创建](#2-3d-bbox-的创建)
3. [3D BBox 的更新](#3-3d-bbox-的更新)
4. [3D BBox 在匹配中的使用](#4-3d-bbox-在匹配中的使用)
5. [2D BBox 的创建](#5-2d-bbox-的创建)
6. [2D BBox 在全局匹配中的使用](#6-2d-bbox-在全局匹配中的使用)
7. [BBox 在 On-Relation 中的使用](#7-bbox-在-on-relation-中的使用)
8. [BBox 在语义地图 JSON 中的导出](#8-bbox-在语义地图-json-中的导出)
9. [major_plane_info：平面物体的 Z 峰值](#9-major_plane_info平面物体的-z-峰值)

---

## 1. 核心结论

| 属性 | 3D BBox (`obj.bbox`) | 2D BBox (`obj.bbox_2d`) |
|------|---------------------|------------------------|
| **类型** | `o3d.geometry.AxisAlignedBoundingBox` | `o3d.geometry.AxisAlignedBoundingBox` |
| **坐标系** | 世界坐标系 (X, Y, Z) | 世界坐标系 XY 平面 (Z = floor_height) |
| **对齐方式** | 与世界坐标轴 XYZ 严格平行 | 与世界坐标轴 XY 严格平行 |
| **有旋转吗** | **没有**。永远是轴对齐的，不随物体朝向旋转 | **没有** |
| **计算方法** | 点云所有点在 X/Y/Z 上的 min/max | 3D 点云投影到 XY 后取 min/max |
| **用途** | 局部匹配 3D IoU 预筛选、On-Relation | 全局匹配交集比率、导航规划 |

---

## 2. 3D BBox 的创建

### 2.1 首次创建：检测阶段

**位置**: `object_detector.py:1011-1017`

```python
# 从掩膜+深度生成的点云此时在相机坐标系
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(self.masked_points[i])  # 相机坐标系
pcd.colors = o3d.utility.Vector3dVector(self.masked_colors[i])

# 变换到世界坐标系
pcd.transform(self.curr_data.pose)   # pose 是 4×4 SE(3) 世界位姿

# 计算轴对齐包围盒
bbox = safe_create_bbox(pcd)
```

### 2.2 safe_create_bbox 实现

**位置**: `pcd_utils.py:208-223`

```python
def safe_create_bbox(pcd):
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        # 空点云返回零 bbox
        return o3d.geometry.AxisAlignedBoundingBox(
            np.array([0, 0, 0]), np.array([0, 0, 0])
        )
    else:
        return pcd.get_axis_aligned_bounding_box()
```

### 2.3 Open3D `get_axis_aligned_bounding_box` 的内部逻辑

Open3D 的 AABB 计算等价于：

```python
points = np.asarray(pcd.points)          # (N, 3) 世界坐标系点
min_bound = points.min(axis=0)           # [x_min, y_min, z_min]
max_bound = points.max(axis=0)           # [x_max, y_max, z_max]
bbox = AxisAlignedBoundingBox(min_bound, max_bound)
```

由此得到的 AABB 具有以下属性：

```python
bbox.get_center()     → (min_bound + max_bound) / 2     # 中心点 [cx, cy, cz]
bbox.get_extent()     → max_bound - min_bound            # 尺寸 [dx, dy, dz]
bbox.get_min_bound()  → min_bound                        # 最小角 [x_min, y_min, z_min]
bbox.get_max_bound()  → max_bound                        # 最大角 [x_max, y_max, z_max]
bbox.get_box_points() → 8个顶点 (8, 3)                   # 长方体 8 个角点
```

### 2.4 关于方向

AABB 的六个面分别垂直于世界坐标系的 X、Y、Z 轴。**它不具有旋转自由度**。

具体含义：
- 一把正放的椅子（正对 X 轴），其 bbox 的 extent 大致是 `[宽, 深, 高]`
- 同一把椅子旋转 45° 放置，其点云所有点的 X/Y 范围都会变大，bbox 的 XY extent 会膨胀（对角线方向），但 bbox 本身仍然与坐标轴平行
- 极端情况：一根斜放的扫帚（从地面到墙角），AABB 的体积会远大于扫帚实际体积

这是有意的设计取舍 — AABB 的优势是：
1. **O(N) 计算**：遍历一次点云取 min/max
2. **O(1) IoU**：两个 AABB 的交集/并集体积是简单的逐维度比较
3. **无需估计朝向**：OBB 需要 PCA 或凸包算法，且两个 OBB 的 IoU 计算是 NP 问题的近似

---

## 3. 3D BBox 的更新

### 3.1 LocalObject 更新

**位置**: `object.py:LocalObject.update_info`（每次追加新观测时调用）

```python
# 合并多帧点云（直接拼接）
self.pcd += latest_obs.pcd

# 重新计算 AABB — 现在覆盖了所有历史观测的点
self.bbox = self.pcd.get_axis_aligned_bounding_box()
```

随着观测次数增加，同一物体从不同视角被看到，点云覆盖范围更完整，bbox 也逐渐趋向稳定。

每 3 帧做一次点云下采样控制规模（不影响 bbox 范围，因为下采样保留了空间分布）：

```python
if self.observed_num % 3 == 0:    # downsample_interval = 3
    self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
```

### 3.2 GlobalObject 更新

**位置**: `object.py:GlobalObject.update_info`

```python
# 合并入新的点云
self.pcd += latest_obs.pcd
# 立即下采样
self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
# 重新计算 3D AABB
self.bbox = self.pcd.get_axis_aligned_bounding_box()
```

---

## 4. 3D BBox 在匹配中的使用

### 4.1 局部匹配：3D IoU 预筛选

**位置**: `tracker.py:compute_spatial_sim`（行 168-232）

局部匹配计算空间相似度时，先用 3D bbox IoU 快速排除不可能重叠的物体对，避免昂贵的逐点最近邻搜索。

**步骤 1：提取 bbox 8 顶点**

```python
# 地图物体
for obj in self.ref_map:
    obj_bbox = np.asarray(obj.bbox.get_box_points())   # (8, 3)
    map_bbox_values.append(torch.from_numpy(obj_bbox))
map_bbox_torch = torch.stack(map_bbox_values, dim=0)    # (M, 8, 3)

# 当前观测
for obs in self.curr_frame:
    obs_bbox = np.asarray(obs.bbox.get_box_points())    # (8, 3)
    curr_bbox_values.append(torch.from_numpy(obs_bbox))
curr_bbox_torch = torch.stack(curr_bbox_values, dim=0)  # (N, 8, 3)
```

`get_box_points()` 返回 AABB 的 8 个角点，顺序是 Open3D 的约定：
```
[min_x,min_y,min_z], [max_x,min_y,min_z], [min_x,max_y,min_z], ...
即 (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)
相对位置排列。
```

**步骤 2：批量 3D IoU 计算**

`tracker.py:compute_3d_iou_batch`（行 493-524）：

```python
def compute_3d_iou_batch(self, bbox1, bbox2):
    # bbox1: (M, 8, 3), bbox2: (N, 8, 3)，在 GPU 上运算

    # 计算每个 bbox 的体积
    volume1 = self.compute_box_volume_torch(bbox1)    # (M,)
    volume2 = self.compute_box_volume_torch(bbox2)    # (N,)

    # 计算两两交集体积
    intersection_volume = self.compute_intersection_volume_torch(bbox1, bbox2)  # (M, N)

    # IoU = 交集 / (A + B - 交集)
    iou = intersection_volume / (volume1[:, None] + volume2 - intersection_volume)
    return iou    # (M, N)
```

**体积计算** (`compute_box_volume_torch`，行 467-472)：

```python
def compute_box_volume_torch(self, box):
    # box: (M, 8, 3)
    # Open3D 的 get_box_points() 中，顶点 0/1/3/4 形成三条相邻边
    edge1 = torch.norm(box[:, 1] - box[:, 0], dim=-1)   # X 方向边长
    edge2 = torch.norm(box[:, 3] - box[:, 0], dim=-1)   # Y 方向边长
    edge3 = torch.norm(box[:, 4] - box[:, 0], dim=-1)   # Z 方向边长
    return edge1 * edge2 * edge3
```

因为是 AABB，三条边分别平行于 X/Y/Z 轴，所以 `norm(box[1]-box[0])` 就是 X 方向的 extent。

**交集体积计算** (`compute_intersection_volume_torch`，行 474-491)：

```python
def compute_intersection_volume_torch(self, bbox1, bbox2):
    # 从 8 顶点恢复 min/max 角
    min_corner1 = torch.min(bbox1, dim=1).values    # (M, 3)
    max_corner1 = torch.max(bbox1, dim=1).values    # (M, 3)
    min_corner2 = torch.min(bbox2, dim=1).values    # (N, 3)
    max_corner2 = torch.max(bbox2, dim=1).values    # (N, 3)

    # 逐维度计算交集范围
    min_intersection = torch.maximum(min_corner1[:, None], min_corner2)   # (M, N, 3)
    max_intersection = torch.minimum(max_corner1[:, None], max_corner2)   # (M, N, 3)

    # 交集维度（负值 clamp 为 0 表示该维度无交集）
    intersection_dims = torch.clamp(max_intersection - min_intersection, min=0)  # (M, N, 3)

    # 交集体积 = dx × dy × dz
    return torch.prod(intersection_dims, dim=-1)    # (M, N)
```

数学公式：
```
dx = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))
dy = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
dz = max(0, min(max_z1, max_z2) - max(min_z1, min_z2))
交集体积 = dx × dy × dz
```

**步骤 3：IoU 筛选后做精细点云重叠率计算**

```python
for idx_a in range(len_map):
    for idx_b in range(len_curr):
        if iou[idx_a, idx_b] < 1e-6:
            counter += 1
            continue          # bbox 无交集 → 跳过（不做最近邻搜索）

        # FAISS 最近邻：当前观测的每个点在地图物体中找最近点
        D, I = indices_map[idx_a].search(points_curr[idx_b], 1)

        # 距离 < voxel_size² (0.02² = 0.0004) 的点算"重叠"
        overlap = (D < self.cfg.downsample_voxel_size ** 2).sum()

        # 重叠率 = 重叠点数 / 当前观测总点数
        overlap_matrix[idx_a, idx_b] = overlap / len(points_curr[idx_b])
```

这里 FAISS 的 `IndexFlatL2` 返回的 `D` 是 **L2 距离的平方**（不是距离本身），所以阈值用 `voxel_size²`。

### 4.2 合并匹配的 overlap_spatial_sim

**位置**: `tracker.py:compute_overlap_spatial_sim`（行 126-166）

这个函数用于局部地图后处理合并（`merge_local_map`）时判断两个 LocalObject 是否应合并。逻辑与上面类似，但使用双向 FAISS 搜索：

```python
def find_overlapping_ratio_faiss(self, pcd1, pcd2, radius=0.02):
    # 双向搜索
    D1, I1 = index2.search(pcd1, k=1)   # pcd1 的每个点在 pcd2 中找最近
    D2, I2 = index1.search(pcd2, k=1)   # pcd2 的每个点在 pcd1 中找最近

    overlap1 = np.sum(D1 < radius**2) / len(pcd1)
    overlap2 = np.sum(D2 < radius**2) / len(pcd2)

    return max(overlap1, overlap2)       # 取较大者
```

---

## 5. 2D BBox 的创建

### 5.1 投影过程

**位置**: `object.py:voxel_downsample_2d`（行 157-196）

从 3D 点云生成 2D 投影点云和 2D bbox：

```python
def voxel_downsample_2d(self, pcd, voxel_size):
    points_arr = np.asarray(pcd.points)     # (N, 3) 世界坐标系 3D 点
    colors_arr = np.asarray(pcd.colors)

    # ====== 步骤1: 丢弃 Z 轴，只保留 XY ======
    points_2d = points_arr[:, :2]           # (N, 2)

    # ====== 步骤2: 2D 体素网格下采样 ======
    # 将连续坐标离散化到网格
    grid_indices = np.floor(points_2d / voxel_size).astype(np.int32)
    # 找到唯一的网格单元
    unique_indices, inverse_indices = np.unique(
        grid_indices, axis=0, return_inverse=True
    )

    # ====== 步骤3: 每个网格单元内取均值 ======
    downsampled_points_2d = np.zeros_like(unique_indices, dtype=np.float64)
    downsampled_colors = np.zeros((len(unique_indices), 3), dtype=np.float64)

    for i in range(len(unique_indices)):
        mask = inverse_indices == i
        downsampled_points_2d[i] = points_2d[mask].mean(axis=0)    # XY 均值
        downsampled_colors[i] = colors_arr[mask].mean(axis=0)      # 颜色均值

    # ====== 步骤4: Z 坐标设为 floor_height ======
    downsampled_points = np.zeros((len(downsampled_points_2d), 3))
    downsampled_points[:, :2] = downsampled_points_2d
    downsampled_points[:, 2] = self._cfg.floor_height    # 如 0.0

    # 创建 2D 投影点云
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)
    return downsampled_pcd
```

### 5.2 2D BBox 的计算

创建了 2D 投影点云后，bbox_2d 的计算：

```python
pcd_2d = obj.voxel_downsample_2d(obj.pcd, voxel_size=0.02)
bbox_2d = pcd_2d.get_axis_aligned_bounding_box()
```

因为所有点的 Z 都等于 `floor_height`（常量），所以这个 AABB 的结果：

```
min_bound = [x_min, y_min, floor_height]
max_bound = [x_max, y_max, floor_height]
center    = [(x_min+x_max)/2, (y_min+y_max)/2, floor_height]
extent    = [x_max-x_min, y_max-y_min, 0]     ← Z 维度为 0
```

**本质上是世界坐标系 XY 平面上的一个矩形**，与 X/Y 轴平行。

### 5.3 2D BBox 的调用位置

**创建 GlobalObservation 时** (`local_map_manager.py:create_global_observation`，行 465-481)：
```python
pcd_2d = obj.voxel_downsample_2d(obj.pcd, self.cfg.downsample_voxel_size)
curr_obs.pcd_2d = pcd_2d
curr_obs.bbox_2d = pcd_2d.get_axis_aligned_bounding_box()
```

**GlobalObject 更新时** (`object.py:GlobalObject.update_info`，行 868-871)：
```python
self.pcd_2d += latest_obs.pcd_2d
self.pcd_2d = self.voxel_downsample_2d(pcd=self.pcd_2d, voxel_size=0.02)
self.bbox_2d = self.pcd_2d.get_axis_aligned_bounding_box()
```

---

## 6. 2D BBox 在全局匹配中的使用

### 6.1 提取 2D BBox 为 4 值向量

**位置**: `tracker.py:compute_global_spatial_sim`（行 234-271）

全局匹配不用 3D 信息，只用 2D bbox：

```python
def compute_global_spatial_sim(self):
    # 地图物体的 2D bbox
    for obj in self.ref_map:
        min_bound = obj.bbox_2d.get_min_bound()
        max_bound = obj.bbox_2d.get_max_bound()
        map_bbox_values.append(
            torch.tensor([min_bound[0], min_bound[1], max_bound[0], max_bound[1]])
        )
    map_bbox_torch = torch.stack(map_bbox_values, dim=0)     # (M, 4)

    # 当前观测的 2D bbox
    for obs in self.curr_frame:
        min_bound = obs.bbox_2d.get_min_bound()
        max_bound = obs.bbox_2d.get_max_bound()
        curr_bbox_values.append(
            torch.tensor([min_bound[0], min_bound[1], max_bound[0], max_bound[1]])
        )
    curr_bbox_torch = torch.stack(curr_bbox_values, dim=0)   # (N, 4)

    # 计算交集比率
    ratio = self.compute_match_by_intersection_ratio(map_bbox_torch, curr_bbox_torch)
    return ratio
```

注意：4 值格式为 `[min_x, min_y, max_x, max_y]`，Z 被完全忽略（因为所有 Z 都是 floor_height）。

### 6.2 交集比率计算

**位置**: `tracker.py:compute_match_by_intersection_ratio`（行 273-332）

```python
def compute_match_by_intersection_ratio(self, bboxes1, bboxes2, threshold=0.8):
    # bboxes1: (M, 4) [min_x, min_y, max_x, max_y]
    # bboxes2: (N, 4)

    # 提取坐标
    b1_min_x, b1_min_y, b1_max_x, b1_max_y = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
    b2_min_x, b2_min_y, b2_max_x, b2_max_y = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    # 计算交集矩形
    inter_min_x = torch.max(b1_min_x[:, None], b2_min_x)     # (M, N)
    inter_min_y = torch.max(b1_min_y[:, None], b2_min_y)
    inter_max_x = torch.min(b1_max_x[:, None], b2_max_x)
    inter_max_y = torch.min(b1_max_y[:, None], b2_max_y)

    # 交集面积（负值 clamp 为 0）
    inter_width  = (inter_max_x - inter_min_x).clamp(min=0)
    inter_height = (inter_max_y - inter_min_y).clamp(min=0)
    inter_area = inter_width * inter_height                    # (M, N)

    # 各自面积
    area1 = (b1_max_x - b1_min_x) * (b1_max_y - b1_min_y)   # (M,)
    area2 = (b2_max_x - b2_min_x) * (b2_max_y - b2_min_y)   # (N,)

    # 交集占各自面积的比率
    ratio1 = inter_area / area1[:, None]    # 交集 / bbox1 面积
    ratio2 = inter_area / area2             # 交集 / bbox2 面积

    # 取较大者
    match_matrix = torch.max(ratio1, ratio2)    # (M, N)
    return match_matrix
```

**数学公式**：
```
ratio = max(inter_area / area_A, inter_area / area_B)
```

取 max 的原因：如果一个小物体完全在大物体内部，`inter_area/area_小 = 1.0` 但 `inter_area/area_大` 可能很小。取 max 确保小物体被大物体包含时也能匹配到。

在 `tracker.py:update_obs_with_sim_mat` 中，匹配阈值为 `object_tracking.max_similarity = 0.8`，即交集面积至少占较小 bbox 的 80% 才算匹配。

---

## 7. BBox 在 On-Relation 中的使用

**位置**: `local_map_manager.py:on_relation_check`（行 390-463）

On-Relation 检测使用 3D bbox 的 min_bound/max_bound 来判断"物体 A 在物体 B 表面上"：

```python
def on_relation_check(self, base_obj, test_obj):
    # 确保 base_obj 是有平面信息的（如桌子），test_obj 是没有的（如杯子）
    if base_obj.major_plane_info is None:
        base_obj, test_obj = test_obj, base_obj

    base_aabb = base_obj.bbox
    test_aabb = test_obj.bbox

    # ====== XY 平面重叠检查 ======
    base_min = base_aabb.get_min_bound()    # [x_min, y_min, z_min]
    base_max = base_aabb.get_max_bound()    # [x_max, y_max, z_max]
    test_min = test_aabb.get_min_bound()
    test_max = test_aabb.get_max_bound()

    overlap_x = max(0, min(base_max[0], test_max[0]) - max(base_min[0], test_min[0]))
    overlap_y = max(0, min(base_max[1], test_max[1]) - max(base_min[1], test_min[1]))
    test_area = (test_max[0] - test_min[0]) * (test_max[1] - test_min[1])
    overlap_ratio = (overlap_x * overlap_y) / test_area

    if overlap_ratio < 0.8:     # object_matching.overlap_ratio
        return False

    # ====== Z 轴平面检查 ======
    # test_obj 的底面 Z（test_min[2]）应该接近 base_obj 的主平面 Z
    plane_distance = 0.1    # on_relation.plane_distance
    major_z = base_obj.major_plane_info     # 桌面高度

    if not (test_min[2] - plane_distance <= major_z <= test_min[2] + plane_distance * 2):
        return False

    return True
```

图示：
```
                  ┌─────────┐  ← test_obj (杯子)
                  │         │     test_min[2] ≈ major_z
     ─────────────┼─────────┼───── major_z (桌面高度)
     │            └─────────┘    │
     │       base_obj (桌子)     │
     │                           │
     └───────────────────────────┘
```

条件含义：
- XY 投影重叠 ≥ 80%：杯子在桌子正上方，不是悬在旁边
- Z 轴：杯子底面 ≈ 桌面高度 ± 容差

---

## 8. BBox 在语义地图 JSON 中的导出

**位置**: `semantic_map_manager.py:_build_entry`

```python
def _build_entry(self, obj):
    bbox_3d = obj.bbox
    bbox_center_3d = np.asarray(bbox_3d.get_center()).tolist()     # [cx, cy, cz]
    bbox_extent_3d = np.asarray(bbox_3d.get_extent()).tolist()     # [dx, dy, dz]
    bbox_min_3d = np.asarray(bbox_3d.get_min_bound()).tolist()     # [x_min, y_min, z_min]
    bbox_max_3d = np.asarray(bbox_3d.get_max_bound()).tolist()     # [x_max, y_max, z_max]

    bbox_2d = obj.bbox_2d
    bbox_center_2d = np.asarray(bbox_2d.get_center()).tolist()     # [cx, cy, fh]
    bbox_extent_2d = np.asarray(bbox_2d.get_extent()).tolist()     # [dx, dy, 0]

    entry = {
        "bbox_3d": {
            "center": bbox_center_3d,
            "extent": bbox_extent_3d,
            "min_bound": bbox_min_3d,
            "max_bound": bbox_max_3d,
        },
        "bbox_2d": {
            "center": bbox_center_2d,
            "extent": bbox_extent_2d,
        },
        ...
    }
```

**空间关系图** (`spatial_relation_graph.py`) 中也使用 3D bbox 的 min_bound/max_bound 来计算 on_top_of、near、adjacent 等关系。

---

## 9. major_plane_info：平面物体的 Z 峰值

**位置**: `object.py:find_major_plane_info`（行 690-714）

这不是 bbox，而是与 bbox 配合使用的辅助信息，用于 On-Relation 检测中判断"桌面在哪个高度"。

```python
def find_major_plane_info(self, bin_size=0.02):
    # 取物体所有点的 Z 坐标
    z_axis = np.asarray(self.pcd.points)[:, 2]

    # Z 坐标直方图（bin 宽 2cm）
    bin_edges = np.arange(z_axis.min(), z_axis.max() + bin_size, bin_size)
    hist, bin_edges = np.histogram(z_axis, bins=bin_edges)

    # 取直方图峰值对应的 Z（即点最密集的高度）
    peak_index = np.argmax(hist)
    major_plane_z = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2.0

    return major_plane_z
```

**原理**: 桌子的点云中，桌面上的点最密集（因为桌面面积大且可见性好），所以 Z 直方图的峰值就是桌面高度。

调用时机：每次 LocalObject 下采样时（每 3 帧），对低移动性物体计算：
```python
if self.observed_num % 3 == 0:
    self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
    if self.is_low_mobility:
        self.major_plane_info = self.find_major_plane_info()
```

只有低移动性物体（家具）才有 `major_plane_info`。On-Relation 检测要求恰好一个物体有此信息（平面载体）、另一个没有（放置其上的物体）。
