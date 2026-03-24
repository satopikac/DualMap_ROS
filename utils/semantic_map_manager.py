"""
SemanticMapManager - 语义地图 JSON 实时管理模块

负责将全局地图中检测到的物体信息以 JSON 格式实时保存，包括：
- 全局坐标（包围盒中心点）
- 几何尺寸（包围盒 extent）
- 语义信息（类别名称）
- 实例 ID（UUID）
- 观测次数
- 移动性分类
- 关联物体信息

支持实时增删改查，保持 JSON 文件与内存中全局地图同步。
"""

import json
import logging
import os
import threading
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class SemanticMapManager:
    """管理语义地图的 JSON 文件实时保存与更新。"""

    def __init__(self, save_dir: str, obj_classes=None):
        """
        Args:
            save_dir: JSON 文件保存目录
            obj_classes: ObjectClasses 实例，用于获取类别名称
        """
        self.save_dir = save_dir
        self.obj_classes = obj_classes
        self.semantic_map_path = os.path.join(save_dir, "semantic_map.json")
        self._lock = threading.Lock()

        # 内存中的语义地图条目 {uid_str: entry_dict}
        self._entries = {}

        os.makedirs(save_dir, exist_ok=True)

        # 如果已存在旧文件则加载
        if os.path.exists(self.semantic_map_path):
            try:
                with open(self.semantic_map_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "objects" in data:
                    for entry in data["objects"]:
                        uid = entry.get("instance_id", "")
                        if uid:
                            self._entries[uid] = entry
                logger.info(
                    f"[SemanticMap] Loaded {len(self._entries)} existing entries"
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning("[SemanticMap] Failed to load existing file, starting fresh")
                self._entries = {}

    def set_obj_classes(self, obj_classes):
        """设置物体类别管理器。"""
        self.obj_classes = obj_classes

    def update_from_global_map(self, global_map: list):
        """
        从全局地图同步更新语义地图 JSON。

        对全局地图中的每个物体，提取其信息并更新/新增到 JSON 中。
        对已不在全局地图中的物体条目进行删除。

        Args:
            global_map: GlobalObject 列表
        """
        with self._lock:
            current_uids = set()

            for obj in global_map:
                uid_str = str(obj.uid)
                current_uids.add(uid_str)

                entry = self._build_entry(obj)
                self._entries[uid_str] = entry

            # 删除已不在全局地图中的条目
            stale_uids = set(self._entries.keys()) - current_uids
            for uid in stale_uids:
                del self._entries[uid]
                logger.info(f"[SemanticMap] Removed stale object: {uid}")

            # 写入文件
            self._save_to_disk()

    def add_or_update_object(self, obj):
        """添加或更新单个物体条目。"""
        with self._lock:
            uid_str = str(obj.uid)
            entry = self._build_entry(obj)
            self._entries[uid_str] = entry
            self._save_to_disk()

    def remove_object(self, uid):
        """删除单个物体条目。"""
        with self._lock:
            uid_str = str(uid)
            if uid_str in self._entries:
                del self._entries[uid_str]
                self._save_to_disk()
                logger.info(f"[SemanticMap] Removed object: {uid_str}")

    def _build_entry(self, obj) -> dict:
        """从 GlobalObject 构建 JSON 条目。"""
        # 3D 包围盒信息
        bbox_3d = obj.bbox
        bbox_center_3d = np.asarray(bbox_3d.get_center()).tolist()
        bbox_extent_3d = np.asarray(bbox_3d.get_extent()).tolist()
        bbox_min_3d = np.asarray(bbox_3d.get_min_bound()).tolist()
        bbox_max_3d = np.asarray(bbox_3d.get_max_bound()).tolist()

        # 2D 包围盒信息 (俯视图)
        bbox_2d = obj.bbox_2d
        bbox_center_2d = np.asarray(bbox_2d.get_center()).tolist()
        bbox_extent_2d = np.asarray(bbox_2d.get_extent()).tolist()

        # 类别名称
        class_name = "unknown"
        if self.obj_classes is not None and obj.class_id is not None:
            try:
                classes_arr = self.obj_classes.get_classes_arr()
                if 0 <= obj.class_id < len(classes_arr):
                    class_name = classes_arr[obj.class_id]
            except (IndexError, ValueError):
                class_name = f"class_{obj.class_id}"

        # 点云统计
        pcd_points = np.asarray(obj.pcd.points)
        num_points_3d = len(pcd_points)

        pcd_2d_points = np.asarray(obj.pcd_2d.points)
        num_points_2d = len(pcd_2d_points)

        # 关联物体
        related_objects = []
        if hasattr(obj, "related_color") and obj.related_color:
            for i, color_id in enumerate(obj.related_color):
                related_name = "unknown"
                if self.obj_classes is not None:
                    try:
                        classes_arr = self.obj_classes.get_classes_arr()
                        if 0 <= color_id < len(classes_arr):
                            related_name = classes_arr[color_id]
                    except (IndexError, ValueError):
                        related_name = f"class_{color_id}"

                related_entry = {
                    "class_name": related_name,
                    "class_id": int(color_id),
                }

                # 如果有关联包围盒则添加
                if i < len(obj.related_bbox):
                    r_bbox = obj.related_bbox[i]
                    related_entry["bbox_center"] = np.asarray(
                        r_bbox.get_center()
                    ).tolist()
                    related_entry["bbox_extent"] = np.asarray(
                        r_bbox.get_extent()
                    ).tolist()

                related_objects.append(related_entry)

        entry = {
            "instance_id": str(obj.uid),
            "class_name": class_name,
            "class_id": int(obj.class_id) if obj.class_id is not None else -1,
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
            "num_points_3d": num_points_3d,
            "num_points_2d": num_points_2d,
            "observed_num": obj.observed_num,
            "is_nav_goal": obj.nav_goal,
            "related_objects": related_objects,
            "last_updated": datetime.now().isoformat(),
        }

        return entry

    def _save_to_disk(self):
        """将当前语义地图写入 JSON 文件。"""
        output = {
            "metadata": {
                "total_objects": len(self._entries),
                "last_updated": datetime.now().isoformat(),
            },
            "objects": list(self._entries.values()),
        }

        try:
            # 写入临时文件再重命名，确保原子性
            tmp_path = self.semantic_map_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.semantic_map_path)
        except Exception as e:
            logger.error(f"[SemanticMap] Failed to save: {e}")

    def get_entry(self, uid_str: str) -> dict:
        """获取单个物体条目。"""
        with self._lock:
            return self._entries.get(uid_str, None)

    def get_all_entries(self) -> list:
        """获取所有物体条目列表。"""
        with self._lock:
            return list(self._entries.values())

    def get_object_count(self) -> int:
        """获取当前物体总数。"""
        return len(self._entries)
