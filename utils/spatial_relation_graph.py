"""
SpatialRelationGraph - 物体空间关系图模块

使用图结构描述已识别物体之间的空间联系，并以 JSON 格式实时保存。

空间关系类型：
- "on_top_of": 物体 A 在物体 B 的上方（A 底面 Z ≈ B 顶面 Z，且 XY 投影重叠）
- "below":     物体 A 在物体 B 的下方（on_top_of 的反向）
- "near":      物体 A 与物体 B 在水平距离上相近（< near_threshold）
- "adjacent":  物体 A 与物体 B 的包围盒在任一轴向上紧邻（间距 < adjacent_threshold）
- "contains":  物体 A 的包围盒包含物体 B 的大部分体积
- "same_level": 物体 A 和 B 的底面 Z 接近（在同一水平面上）

图以 JSON 格式保存，包含节点列表（物体）和边列表（关系）。
"""

import json
import logging
import os
import threading
from datetime import datetime
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


class SpatialRelationGraph:
    """管理物体间空间关系的图结构，实时保存为 JSON。"""

    def __init__(
        self,
        save_dir: str,
        obj_classes=None,
        near_threshold: float = 1.0,
        adjacent_threshold: float = 0.15,
        on_top_z_tolerance: float = 0.15,
        same_level_z_tolerance: float = 0.10,
        overlap_ratio_threshold: float = 0.3,
    ):
        """
        Args:
            save_dir: JSON 文件保存目录
            obj_classes: ObjectClasses 实例，用于获取类别名称
            near_threshold: "near" 关系的最大水平距离 (m)
            adjacent_threshold: "adjacent" 关系的最大间隙距离 (m)
            on_top_z_tolerance: "on_top_of" 关系的 Z 轴容差 (m)
            same_level_z_tolerance: "same_level" 关系的 Z 轴容差 (m)
            overlap_ratio_threshold: XY 投影重叠比率阈值
        """
        self.save_dir = save_dir
        self.obj_classes = obj_classes
        self.graph_path = os.path.join(save_dir, "spatial_graph.json")
        self._lock = threading.Lock()

        self.near_threshold = near_threshold
        self.adjacent_threshold = adjacent_threshold
        self.on_top_z_tolerance = on_top_z_tolerance
        self.same_level_z_tolerance = same_level_z_tolerance
        self.overlap_ratio_threshold = overlap_ratio_threshold

        # 图数据
        self._nodes = {}  # uid_str -> node_dict
        self._edges = []  # list of edge_dict

        os.makedirs(save_dir, exist_ok=True)

    def set_obj_classes(self, obj_classes):
        """设置物体类别管理器。"""
        self.obj_classes = obj_classes

    def update_from_global_map(self, global_map: list):
        """
        从全局地图重建空间关系图。

        Args:
            global_map: GlobalObject 列表
        """
        with self._lock:
            self._nodes = {}
            self._edges = []

            # 构建节点
            for obj in global_map:
                node = self._build_node(obj)
                self._nodes[node["instance_id"]] = node

            # 构建边（两两检测空间关系）
            obj_list = list(global_map)
            for i, j in combinations(range(len(obj_list)), 2):
                relations = self._compute_relations(obj_list[i], obj_list[j])
                for rel_type, properties in relations:
                    edge = self._build_edge(obj_list[i], obj_list[j], rel_type, properties)
                    self._edges.append(edge)

            # 保存到磁盘
            self._save_to_disk()

    def _build_node(self, obj) -> dict:
        """构建节点信息。"""
        bbox = obj.bbox
        center = np.asarray(bbox.get_center()).tolist()
        extent = np.asarray(bbox.get_extent()).tolist()

        class_name = self._get_class_name(obj.class_id)

        return {
            "instance_id": str(obj.uid),
            "class_name": class_name,
            "class_id": int(obj.class_id) if obj.class_id is not None else -1,
            "position": center,
            "extent": extent,
            "min_bound": np.asarray(bbox.get_min_bound()).tolist(),
            "max_bound": np.asarray(bbox.get_max_bound()).tolist(),
        }

    def _build_edge(self, obj_a, obj_b, relation_type: str, properties: dict) -> dict:
        """构建边信息。"""
        return {
            "source": str(obj_a.uid),
            "source_class": self._get_class_name(obj_a.class_id),
            "target": str(obj_b.uid),
            "target_class": self._get_class_name(obj_b.class_id),
            "relation": relation_type,
            "properties": properties,
        }

    def _compute_relations(self, obj_a, obj_b) -> list:
        """
        计算两个物体间的所有空间关系。

        Returns:
            list of (relation_type, properties_dict)
        """
        relations = []

        bbox_a = obj_a.bbox
        bbox_b = obj_b.bbox

        min_a = np.asarray(bbox_a.get_min_bound())
        max_a = np.asarray(bbox_a.get_max_bound())
        min_b = np.asarray(bbox_b.get_min_bound())
        max_b = np.asarray(bbox_b.get_max_bound())

        center_a = (min_a + max_a) / 2
        center_b = (min_b + max_b) / 2

        extent_a = max_a - min_a
        extent_b = max_b - min_b

        # --- 1. On-top-of 关系 ---
        # A 在 B 上方: A 的底面 Z ≈ B 的顶面 Z, 且 XY 投影重叠
        xy_overlap = self._compute_xy_overlap_ratio(min_a, max_a, min_b, max_b)

        if xy_overlap > self.overlap_ratio_threshold:
            # A on top of B
            z_gap_a_on_b = abs(min_a[2] - max_b[2])
            if z_gap_a_on_b < self.on_top_z_tolerance and center_a[2] > center_b[2]:
                relations.append(("on_top_of", {
                    "z_gap": round(float(z_gap_a_on_b), 4),
                    "xy_overlap": round(float(xy_overlap), 4),
                    "description": f"{self._get_class_name(obj_a.class_id)} is on top of {self._get_class_name(obj_b.class_id)}",
                }))

            # B on top of A
            z_gap_b_on_a = abs(min_b[2] - max_a[2])
            if z_gap_b_on_a < self.on_top_z_tolerance and center_b[2] > center_a[2]:
                relations.append(("on_top_of", {
                    "z_gap": round(float(z_gap_b_on_a), 4),
                    "xy_overlap": round(float(xy_overlap), 4),
                    "description": f"{self._get_class_name(obj_b.class_id)} is on top of {self._get_class_name(obj_a.class_id)}",
                }))
                # 添加反向 edge, swap source/target
                # 这里只记录 A→B 方向，在 _build_edge 中 source=A, target=B
                # on_top_of 的反向在下面的 below 处理

        # --- 2. Same-level 关系 ---
        z_bottom_diff = abs(min_a[2] - min_b[2])
        if z_bottom_diff < self.same_level_z_tolerance:
            relations.append(("same_level", {
                "z_difference": round(float(z_bottom_diff), 4),
                "description": f"{self._get_class_name(obj_a.class_id)} and {self._get_class_name(obj_b.class_id)} are on the same level",
            }))

        # --- 3. Near 关系 ---
        # 水平距离（XY 平面）
        horizontal_dist = np.linalg.norm(center_a[:2] - center_b[:2])
        if horizontal_dist < self.near_threshold:
            relations.append(("near", {
                "horizontal_distance": round(float(horizontal_dist), 4),
                "euclidean_distance": round(float(np.linalg.norm(center_a - center_b)), 4),
                "description": f"{self._get_class_name(obj_a.class_id)} is near {self._get_class_name(obj_b.class_id)}",
            }))

        # --- 4. Adjacent 关系 ---
        # 检查包围盒在各轴向上是否紧邻（间隙小于阈值）
        gap = self._compute_bbox_gap(min_a, max_a, min_b, max_b)
        if gap is not None and gap < self.adjacent_threshold:
            relations.append(("adjacent", {
                "gap_distance": round(float(gap), 4),
                "description": f"{self._get_class_name(obj_a.class_id)} is adjacent to {self._get_class_name(obj_b.class_id)}",
            }))

        # --- 5. Contains 关系 ---
        containment_ab = self._compute_containment_ratio(min_a, max_a, min_b, max_b)
        containment_ba = self._compute_containment_ratio(min_b, max_b, min_a, max_a)

        if containment_ab > 0.7:
            relations.append(("contains", {
                "containment_ratio": round(float(containment_ab), 4),
                "description": f"{self._get_class_name(obj_a.class_id)} contains {self._get_class_name(obj_b.class_id)}",
            }))
        elif containment_ba > 0.7:
            relations.append(("contains", {
                "containment_ratio": round(float(containment_ba), 4),
                "description": f"{self._get_class_name(obj_b.class_id)} contains {self._get_class_name(obj_a.class_id)}",
            }))

        return relations

    def _compute_xy_overlap_ratio(self, min_a, max_a, min_b, max_b) -> float:
        """计算两个包围盒在 XY 平面上的重叠比率。"""
        # XY 平面交集
        inter_min_x = max(min_a[0], min_b[0])
        inter_max_x = min(max_a[0], max_b[0])
        inter_min_y = max(min_a[1], min_b[1])
        inter_max_y = min(max_a[1], max_b[1])

        if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
            return 0.0

        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        area_a = (max_a[0] - min_a[0]) * (max_a[1] - min_a[1])
        area_b = (max_b[0] - min_b[0]) * (max_b[1] - min_b[1])

        min_area = min(area_a, area_b)
        if min_area < 1e-8:
            return 0.0

        return inter_area / min_area

    def _compute_bbox_gap(self, min_a, max_a, min_b, max_b) -> float:
        """
        计算两个包围盒之间的最小间隙距离。
        如果有重叠则返回 None（不算 adjacent）。
        """
        gaps = []
        for dim in range(3):
            if max_a[dim] < min_b[dim]:
                gaps.append(min_b[dim] - max_a[dim])
            elif max_b[dim] < min_a[dim]:
                gaps.append(min_a[dim] - max_b[dim])
            # else: 在该维度上有重叠

        if not gaps:
            # 所有维度都有重叠 → 包围盒相交，不算 adjacent
            return None

        return min(gaps)

    def _compute_containment_ratio(self, min_outer, max_outer, min_inner, max_inner) -> float:
        """计算 inner 被 outer 包含的体积比率。"""
        # 交集
        inter_min = np.maximum(min_outer, min_inner)
        inter_max = np.minimum(max_outer, max_inner)
        inter_dims = np.maximum(inter_max - inter_min, 0)
        inter_vol = np.prod(inter_dims)

        # inner 体积
        inner_dims = max_inner - min_inner
        inner_vol = np.prod(np.maximum(inner_dims, 1e-8))

        return inter_vol / inner_vol

    def _get_class_name(self, class_id) -> str:
        """获取类别名称。"""
        if self.obj_classes is not None and class_id is not None:
            try:
                classes_arr = self.obj_classes.get_classes_arr()
                if 0 <= class_id < len(classes_arr):
                    return classes_arr[class_id]
            except (IndexError, ValueError):
                pass
        return f"class_{class_id}" if class_id is not None else "unknown"

    def _save_to_disk(self):
        """保存空间关系图为 JSON 文件。"""
        # 统计各类型关系数量
        relation_counts = {}
        for edge in self._edges:
            rel = edge["relation"]
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

        output = {
            "metadata": {
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
                "relation_counts": relation_counts,
                "last_updated": datetime.now().isoformat(),
            },
            "nodes": list(self._nodes.values()),
            "edges": self._edges,
        }

        try:
            tmp_path = self.graph_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.graph_path)
        except Exception as e:
            logger.error(f"[SpatialGraph] Failed to save: {e}")

    def get_relations_for_object(self, uid_str: str) -> list:
        """获取指定物体的所有关系。"""
        with self._lock:
            return [
                e for e in self._edges
                if e["source"] == uid_str or e["target"] == uid_str
            ]

    def get_neighbors(self, uid_str: str) -> list:
        """获取指定物体的所有邻居节点 ID。"""
        with self._lock:
            neighbors = set()
            for e in self._edges:
                if e["source"] == uid_str:
                    neighbors.add(e["target"])
                elif e["target"] == uid_str:
                    neighbors.add(e["source"])
            return list(neighbors)

    def get_graph_summary(self) -> dict:
        """获取图的摘要统计。"""
        with self._lock:
            relation_counts = {}
            for edge in self._edges:
                rel = edge["relation"]
                relation_counts[rel] = relation_counts.get(rel, 0) + 1

            return {
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
                "relation_counts": relation_counts,
            }
