import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R

from utils.global_map_manager import GlobalMapManager
from utils.local_map_manager import LocalMapManager
from utils.navigation_helper import (
    remaining_path,
    remove_sharp_turns_3d,
)
from utils.object_detector import Detector
from utils.time_utils import (
    get_map_memory_usage,
    print_timing_results,
    save_timing_results,
    timing_context,
)
from utils.types import DataInput, GoalMode
from utils.visualizer import ReRunVisualizer

# Set up the module-level logger
logger = logging.getLogger(__name__)


class Dualmap:
    def __init__(self, cfg: DictConfig):
        """
        Initialize Dualmap with configuration and essential components.
        """
        self.cfg = cfg

        # print config into console
        self.print_cfg()

        # Initialization
        self.visualizer = ReRunVisualizer(cfg)
        self.detector = Detector(cfg)
        self.local_map_manager = LocalMapManager(cfg)
        self.global_map_manager = GlobalMapManager(cfg)

        # === NEW: Pass obj_classes to semantic managers ===
        obj_classes = self.visualizer.obj_classes
        self.global_map_manager.semantic_map_manager.set_obj_classes(obj_classes)
        self.global_map_manager.spatial_relation_graph.set_obj_classes(obj_classes)

        # Additional initialization for visualization
        self.visualizer.set_use_rerun(cfg.use_rerun)
        self.visualizer.init("refactor_mapping")
        self.visualizer.spawn()

        # Keyframe Selection
        self.keyframe_counter = 0
        self.last_keyframe_time = None
        self.last_keyframe_pose = None
        self.time_threshold = cfg.time_threshold
        self.pose_threshold = cfg.pose_threshold
        self.rotation_threshold = cfg.rotation_threshold

        # pose memory
        self.curr_pose = None
        self.prev_pose = None
        self.wait_count = 0

        # check if need to preload the global map
        if self.cfg.preload_global_map:
            logger.warning("[Core][Init] Preloading global map...")
            self.global_map_manager.load_map()

        if self.cfg.preload_layout:
            logger.warning("[Core][Init] Preloading layout...")
            self.detector.load_layout()

            # load wall.pcd directly from disk
            self.global_map_manager.load_wall()

            # only generate wall.pcd if it doesn't exist
            if self.global_map_manager.layout_map.wall_pcd is None:
                logger.warning("[Core][Init] wall.pcd not found, generating from layout.pcd...")
                layout_pcd = self.detector.get_layout_pointcloud()
                self.global_map_manager.set_layout_info(layout_pcd)

        # Start the file monitoring thread
        self.stop_thread = False  # Signal to stop the thread
        self.monitor_thread = threading.Thread(
            target=self.monitor_config_file, args=(cfg.config_file_path,)
        )
        self.monitor_thread.start()

        # flags for monitoring
        self.calculate_path = False
        self.reset_cal_path_flag = False
        self.trigger_find_next = False
        self.reset_trigger_find_next = False

        # Mode for Getting the Goal
        self.get_goal_mode = GoalMode.RANDOM
        self.inquiry = ""
        self.inquiry_feat = None

        # Start local planning
        self.begin_local_planning = False

        # Final path for agent to follow
        self.action_path = None
        self.curr_global_path = None
        self.curr_local_path = None
        self.start_action_path = False

        # debug param: path counter
        self.path_counter = 0

        # Parallel for mapping thread
        if self.cfg.use_parallel:
            self.detection_results_queue = queue.Queue(maxsize=10)
            self.mapping_thread = threading.Thread(
                target=self.run_mapping_thread, daemon=True
            )
            self.mapping_thread.start()

    def print_cfg(self):
        log_file_path = ""

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_path = handler.baseFilename

        cfg_items = [
            ("Log Path", log_file_path),
            ("Output Dir", self.cfg.output_path),
            ("Map Save Dir", self.cfg.map_save_path),
            ("Class List Path", self.cfg.yolo.given_classes_path),
            ("Use FastSAM for OV?", self.cfg.use_fastsam),
            ("Running Concrete Map Only?", self.cfg.run_local_mapping_only),
            ("Save Concrete Map?", self.cfg.save_local_map),
            ("Save Global Map?", self.cfg.save_global_map),
            ("Use Preload Global Map?", self.cfg.preload_global_map),
            ("Use Rerun for Visualization?", self.cfg.use_rerun),
        ]

        if "ros_stream_config_path" in self.cfg:
            cfg_items.append(
                ("ROS Stream Config Path", self.cfg.ros_stream_config_path)
            )

        line_length = 60
        print("=" * line_length)
        for key, value in cfg_items:
            print(f"{key:<30} : {value}")
        print("=" * line_length)

    def get_keyframe_idx(self):
        return self.keyframe_counter

    def check_keyframe(self, time_stamp, curr_pose):
        is_keyframe = False
        if self.last_keyframe_pose is not None:
            translation_diff = np.linalg.norm(
                curr_pose[:3, 3] - self.last_keyframe_pose[:3, 3]
            )
            if translation_diff >= self.pose_threshold:
                self.last_keyframe_time = time_stamp
                self.last_keyframe_pose = curr_pose
                logger.info(
                    "[Core][CheckKeyframe] New keyframe detected by translation"
                )
                is_keyframe = True

            curr_rotation = R.from_matrix(curr_pose[:3, :3])
            last_rotation = R.from_matrix(self.last_keyframe_pose[:3, :3])
            rotation_diff = curr_rotation.inv() * last_rotation
            angle_diff = rotation_diff.magnitude() * (180 / np.pi)

            if angle_diff >= self.rotation_threshold:
                self.last_keyframe_time = time_stamp
                self.last_keyframe_pose = curr_pose
                logger.info("[Core][CheckKeyframe] New keyframe detected by rotation")
                is_keyframe = True

        if (
            self.last_keyframe_time is None
            or abs(time_stamp - self.last_keyframe_time) >= self.time_threshold
        ):
            self.last_keyframe_time = time_stamp
            self.last_keyframe_pose = curr_pose
            logger.info("[Core][CheckKeyframe] New keyframe detected by time")
            is_keyframe = True

        if is_keyframe:
            self.keyframe_counter += 1
            logger.info(
                f"[Core][CheckKeyframe] Current frame is keyframe: {self.keyframe_counter}"
            )
            return True
        else:
            return False

    def get_total_memory_by_keyword(self, keyword="applications"):
        total_rss = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                cmdline = proc.info.get("cmdline")
                if isinstance(cmdline, list) and keyword in " ".join(cmdline):
                    total_rss += proc.info["memory_info"].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        return total_rss / 1024 / 1024  # MB

    def sequential_process(self, data_input: DataInput):
        self.curr_frame_id = data_input.idx
        self.curr_pose = data_input.pose

        self.visualizer.set_time_sequence("frame", self.curr_frame_id)
        self.visualizer.set_camera_info(data_input.intrinsics, data_input.pose)
        self.visualizer.set_image(data_input.color)

        with timing_context("Detection Process", self):
            self.detector.set_data_input(data_input)

            if self.cfg.run_detection:
                self.detector.process_detections()
                with timing_context("Save Detection", self):
                    if self.cfg.save_detection:
                        self.detector.save_detection_results()
            else:
                self.detector.load_detection_results()

            with timing_context("Vis Detection", self):
                self.detector.calculate_observations()
                if self.cfg.use_rerun:
                    self.detector.visualize_detection()

        with timing_context("Local Mapping", self):
            curr_obs_list = self.detector.get_curr_observations()
            self.detector.update_state()
            self.detector.update_data()
            self.local_map_manager.set_curr_idx(self.curr_frame_id)
            self.local_map_manager.process_observations(curr_obs_list)

        with timing_context("Global Mapping", self):
            global_obs_list = self.local_map_manager.get_global_observations()
            self.local_map_manager.clear_global_observations()
            self.global_map_manager.process_observations(global_obs_list)

    def parallel_process(self, data_input: DataInput):
        self.curr_frame_id = data_input.idx
        self.curr_pose = data_input.pose

        start_time = time.time()
        with timing_context("Observation Generation", self):
            self.detector.set_data_input(data_input)

            with timing_context("Process Detection", self):
                if self.cfg.run_detection:
                    self.detector.process_detections()
                    with timing_context("Save Detection", self):
                        if self.cfg.save_detection:
                            self.detector.save_detection_results()
                else:
                    self.detector.load_detection_results()

            with timing_context("Observation Formatting", self):
                self.detector.calculate_observations()

            curr_obs_list = self.detector.get_curr_observations()
            self.detector.update_state()

            try:
                self.detection_results_queue.put(
                    (curr_obs_list, self.curr_frame_id), timeout=1
                )
            except queue.Full:
                logger.warning(
                    f"[Core] Mapping queue is full, skipping frame {self.curr_frame_id}."
                )

        end_time = time.time()

        self.visualizer.set_time_sequence("frame", self.curr_frame_id)
        self.visualizer.set_camera_info(data_input.intrinsics, data_input.pose)
        self.visualizer.set_image(data_input.color)

        if self.cfg.use_rerun:
            elapsed_time = end_time - start_time
            self.detector.visualize_time(elapsed_time)

        if self.calculate_path and self.global_map_manager.has_global_map():
            logger.info("[Core] Global Navigation enabled! Triggering functionality...")

            self.global_map_manager.has_action_path = False
            self.inquiry_feat = self.convert_inquiry_to_feat(self.inquiry)
            self.global_map_manager.inquiry = self.inquiry_feat

            layout_pcd = self.detector.get_layout_pointcloud()
            self.global_map_manager.set_layout_info(layout_pcd)

            self.curr_global_path = self.global_map_manager.calculate_global_path(
                self.curr_pose, goal_mode=self.get_goal_mode
            )

            if self.cfg.save_all_path:
                self.path_counter += 1

                save_dir = os.path.join(self.cfg.output_path, "path", "global_path")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                file_name = f"{self.path_counter}.json"
                save_path = os.path.join(save_dir, file_name)

                with open(save_path, "w") as f:
                    json.dump(self.curr_global_path, f, indent=4)

                logger.warning(f"[Core] Global path saved to {save_path}")

            self.reset_cal_path_flag = True
            self.curr_local_path = None

            if self.get_goal_mode == GoalMode.RANDOM or (
                self.get_goal_mode == GoalMode.CLICK
                and self.global_map_manager.nav_graph.snapped_goal is None
            ):
                self.begin_local_planning = False
            else:
                self.begin_local_planning = True

        if self.begin_local_planning and self.local_map_manager.has_local_map():
            logger.info("[Core] Local Navigation enabled! Triggering functionality...")
            self.local_map_manager.inquiry = self.inquiry_feat

            if self.trigger_find_next:
                self.begin_local_planning = False
                self.reset_trigger_find_next = True
                self.global_map_manager.lost_and_found = True

            global_path = self.curr_global_path

            start = global_path[-1]
            x, y, z = start
            start_pose = np.eye(4)
            start_pose[0, 3] = x
            start_pose[1, 3] = y
            start_pose[2, 3] = z

            if self.global_map_manager.nav_graph.snapped_goal is not None:
                click_goal = self.global_map_manager.nav_graph.snapped_goal
                self.local_map_manager.set_click_goal(click_goal)

            if self.global_map_manager.global_candidate_bbox is not None:
                goal_bbox = self.global_map_manager.global_candidate_bbox
                goal_score = self.global_map_manager.global_candidate_score
                self.local_map_manager.set_global_bbox(goal_bbox)
                self.local_map_manager.set_global_score(goal_score)

                global_map = self.global_map_manager.global_map
                self.local_map_manager.set_global_map(global_map)

            self.curr_local_path = self.local_map_manager.calculate_local_path(
                start_pose, goal_mode=self.get_goal_mode
            )

            if self.curr_local_path is not None:
                logger.info("[Core] Local Navigation has finished!")

                self.global_map_manager.global_candidate_score = 0.0
                self.global_map_manager.best_candidate_name = None
                self.global_map_manager.ignore_global_obj_list = []
                self.wait_count = 0

                self.begin_local_planning = False
                self.start_action_path = True

                self.global_map_manager.lost_and_found = False

                if self.cfg.save_all_path:
                    save_dir = os.path.join(self.cfg.output_path, "path", "local_path")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    file_name = f"{self.path_counter}.json"
                    save_path = os.path.join(save_dir, file_name)

                    with open(save_path, "w") as f:
                        json.dump(self.curr_local_path, f, indent=4)

                    logger.warning(f"[Core] Local path saved to {save_path}")

            self.prev_pose = self.curr_pose

        self.get_action_path()

    def run_mapping_thread(self):
        while not self.stop_thread:
            try:
                curr_obs_list, curr_frame_id = self.detection_results_queue.get(
                    timeout=1
                )
                logger.info(
                    f"[Core][MappingThread] Received data for frame {curr_frame_id}, Queue size {self.detection_results_queue.qsize()}"
                )

                self.visualizer.set_time_sequence("frame", self.curr_frame_id)

                if self.cfg.use_rerun:
                    self.detector.visualize_detection()
                self.detector.update_data()

                with timing_context("Local Mapping", self):
                    self.local_map_manager.set_curr_idx(curr_frame_id)
                    self.local_map_manager.process_observations(curr_obs_list)

                with timing_context("Global Mapping", self):
                    global_obs_list = self.local_map_manager.get_global_observations()
                    self.local_map_manager.clear_global_observations()
                    self.global_map_manager.process_observations(global_obs_list)

            except queue.Empty:
                continue

    def get_action_path(self):
        if self.curr_global_path is None:
            logger.info("[Core][ActionPath] No Global Path! Action Path not available!")
            self.action_path = None
            self.global_map_manager.action_path = self.action_path
            return

        if self.curr_local_path is None:
            logger.info(
                "[Core][ActionPath] No Local Path! Action Path Now Using Global Path!"
            )
            self.action_path = self.curr_global_path
            self.global_map_manager.action_path = self.action_path
            return

        if self.start_action_path:
            logger.info("[Core][ActionPath] Start Action Path Calculation!")

            self.action_path = self.curr_global_path + self.curr_local_path[1:]
            self.action_path = remaining_path(self.action_path, self.curr_pose)

            if self.cfg.use_remove_sharp_turns:
                self.action_path = remove_sharp_turns_3d(self.action_path)

            if self.cfg.save_all_path:
                save_dir = os.path.join(self.cfg.output_path, "path", "action_path")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                file_name = f"{self.path_counter}.json"
                save_path = os.path.join(save_dir, file_name)

                with open(save_path, "w") as f:
                    json.dump(self.action_path, f, indent=4)

                logger.warning(f"[Core] Action path saved to {save_path}")

            self.global_map_manager.action_path = self.action_path
            self.global_map_manager.has_action_path = True
            self.start_action_path = False

    def end_process(self):
        end_frame_id = self.curr_frame_id

        self.stop_threading()

        end_range = self.cfg.active_window_size + self.cfg.max_pending_count + 1

        for i in range(end_range):
            logger.info("[Core][EndProcess] End Counter: %d", end_frame_id + i + 1)
            self.visualizer.set_time_sequence("frame", end_frame_id + i + 1)

            self.local_map_manager.set_curr_idx(end_frame_id + i + 1)
            self.local_map_manager.end_process()
            local_map_obj_num = len(self.local_map_manager.local_map)
            logger.info("[Core][EndProcess] Local Objects num: %d", local_map_obj_num)

            global_obs_list = self.local_map_manager.get_global_observations()
            self.local_map_manager.clear_global_observations()
            self.global_map_manager.process_observations(global_obs_list)
            global_map_obj_num = len(self.global_map_manager.global_map)
            logger.info("[Core][EndProcess] Global Objects num: %d", global_map_obj_num)

            if local_map_obj_num == 0:
                logger.warning(
                    "[EndProcess] End Processing End. to: %d", end_frame_id + i + 1
                )
                break

        with timing_context("Merging", self):
            if self.cfg.merge_local_map:
                self.local_map_manager.merge_local_map()
                self.visualizer.set_time_sequence("frame", end_range + 1)
                logger.info("[Core][EndProcess] Local Map Merged")

        if self.cfg.save_local_map:
            self.local_map_manager.save_map()

        if self.cfg.save_global_map:
            self.global_map_manager.save_map()

        if self.cfg.save_layout:
            self.detector.save_layout()

        if hasattr(self, "timing_results"):
            print_timing_results("Dualmap", self.timing_results)
            system_time_path = self.cfg.map_save_path + "/../system_time.csv"
            save_timing_results(self.timing_results, system_time_path)

        if hasattr(self.detector, "timing_results"):
            print_timing_results("Detector", self.detector.timing_results)
            detector_time_path = self.cfg.map_save_path + "/../detector_time.csv"
            save_timing_results(self.detector.timing_results, detector_time_path)

    def monitor_config_file(self, config_file_path: str):
        config_file = Path(config_file_path)
        logger.info(f"[Core][Monitor] Monitoring file: {config_file}")

        while not self.stop_thread:
            try:
                if config_file.exists():
                    config_data = OmegaConf.load(config_file)

                    if "calculate_path" in config_data:
                        if self.reset_cal_path_flag:
                            self.reset_cal_path_flag = False
                            config_data.calculate_path = False
                            with open(config_file, "w") as file:
                                OmegaConf.save(config_data, file)

                        calculate_path = config_data.calculate_path
                        if calculate_path:
                            self.calculate_path = True
                        else:
                            self.calculate_path = False

                    if "trigger_find_next" in config_data:
                        if self.reset_trigger_find_next:
                            self.reset_trigger_find_next = False
                            config_data.trigger_find_next = False
                            config_data.calculate_path = True
                            with open(config_file, "w") as file:
                                OmegaConf.save(config_data, file)

                        trigger_find_next = config_data.trigger_find_next
                        if trigger_find_next:
                            self.trigger_find_next = True
                        else:
                            self.trigger_find_next = False

                    if "get_goal_mode" in config_data:
                        mode_value = config_data.get_goal_mode
                        try:
                            self.get_goal_mode = GoalMode(mode_value)
                        except ValueError:
                            logger.warning(
                                f"[Core][Monitor] Invalid mode '{mode_value}' in configuration. Defaulting to RANDOM."
                            )
                            self.get_goal_mode = GoalMode.RANDOM
                            config_data.get_goal_mode = GoalMode.RANDOM.value
                            with open(config_file, "w") as file:
                                OmegaConf.save(config_data, file)

                    if "inquiry_sentence" in config_data:
                        self.inquiry = config_data.inquiry_sentence

                else:
                    logger.error(
                        f"[Core][Monitor] Config file not found: {config_file}"
                    )
            except Exception as e:
                logger.error(f"[Core][Monitor] Error monitoring config file: {e}")

            time.sleep(self.cfg.monitor_interval)

    def set_calculate_path(self, config_file_path: str):
        config_file = Path(config_file_path)
        config_data = OmegaConf.load(config_file)
        if "calculate_path" in config_data:
            config_data.calculate_path = True
            with open(config_file, "w") as file:
                OmegaConf.save(config_data, file)

    def stop_threading(self):
        self.stop_thread = True
        self.monitor_thread.join()

        if self.cfg.use_parallel:
            self.mapping_thread.join()

        logger.info("[Core] Stopped monitoring config file and mapping thread.")

    def convert_inquiry_to_feat(self, inquiry_sentence: str):
        text_query_tokenized = self.detector.clip_tokenizer(inquiry_sentence).to("cuda")
        text_query_ft = self.detector.clip_model.encode_text(text_query_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        return text_query_ft
