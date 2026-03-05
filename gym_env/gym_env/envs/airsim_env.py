import gym
from gym import spaces
import airsim
from configparser import NoOptionError
import keyboard

import torch as th
import numpy as np
import math
import cv2

from .dynamics.multirotor_simple import MultirotorDynamicsSimple
from .dynamics.multirotor_airsim import MultirotorDynamicsAirsim
from .dynamics.fixedwing_simple import FixedwingDynamicsSimple
# from .lgmd.LGMD import LGMD

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal


class AirsimGymEnv(gym.Env, QtCore.QThread):
    # pyqt signal for visualization
    # action_signal = pyqtSignal(int, np.ndarray)
    # state_signal = pyqtSignal(int, np.ndarray)
    # attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)
    # reward_signal = pyqtSignal(int, float, float)
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    # lgmd_signal = pyqtSignal(float, float, np.ndarray)
    episode_signal = pyqtSignal(
        float, int, str, float
    )  # total_reward, episode_steps, done_reason, min_dist_to_obs

    def __init__(self) -> None:
        super().__init__()
        np.set_printoptions(formatter={"float": "{: 4.2f}".format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("init airsim-gym-env.")
        self.model = None
        self.data_path = None
        self.lgmd = None
        self.goal_points = []
        # Cached terminal-condition results for the current timestep.
        # Set by is_done(); reused by info and compute_reward to avoid
        # re-querying AirSim after the drone state may have changed.
        self._term_is_success = False
        self._term_is_crashed = False
        self._term_is_not_in_workspace = False
        self._term_is_timeout = False

    def set_config(self, cfg):
        """get config from .ini file"""
        self.cfg = cfg
        self.env_name = cfg.get("options", "env_name")
        self.dynamic_name = cfg.get("options", "dynamic_name")
        self.keyboard_debug = cfg.getboolean("options", "keyboard_debug")
        self.generate_q_map = cfg.getboolean("options", "generate_q_map")
        self.perception_type = cfg.get("options", "perception")

        # create LGMD agent
        if self.perception_type == "lgmd":
            self.lgmd = LGMD(
                type="origin",
                p_threshold=50,
                s_threshold=0,
                Ki=2,
                i_layer_size=3,
                activate_coeff=1,
                use_on_off=True,
            )
            self.split_out_last = np.array([0, 0, 0, 0, 0])

        print(
            "Environment: ",
            self.env_name,
            "Dynamics: ",
            self.dynamic_name,
            "Perception: ",
            self.perception_type,
        )

        # set dynamics
        if self.dynamic_name == "SimpleFixedwing":
            self.dynamic_model = FixedwingDynamicsSimple(cfg)
        elif self.dynamic_name == "SimpleMultirotor":
            self.dynamic_model = MultirotorDynamicsSimple(cfg)
        elif self.dynamic_name == "Multirotor":
            self.dynamic_model = MultirotorDynamicsAirsim(cfg)
        else:
            raise Exception("Invalid dynamic_name!", self.dynamic_name)

        # set start and goal position according to different environment
        if self.env_name == "NH_center":
            start_position = [0, 0, 5]
            goal_rect = [-128, -128, 128, 128]  # rectangular goal pose
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goal(random_angle=math.pi * 2, rect=goal_rect)
            self.work_space_x = [-140, 140]
            self.work_space_y = [-140, 140]
            self.work_space_z = [0.5, 20]
        elif self.env_name == "NH_tree":
            start_position = [110, 180, 5]
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model.set_goal(distance=90, random_angle=0)
            self.work_space_x = [
                start_position[0],
                start_position[0] + goal_distance + 10,
            ]
            self.work_space_y = [start_position[1] - 30, start_position[1] + 30]
            self.work_space_z = [0.5, 10]
        elif self.env_name == "City":
            start_position = [40, -30, 40]
            goal_position = [280, -200, 40]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 350]
            self.work_space_y = [-300, 100]
            self.work_space_z = [0, 100]
        elif self.env_name == "City_400":
            # note: the start and end points will be covered by update_start_and_goal_pose_random function
            start_position = [0, 0, 50]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-220, 220]
            self.work_space_y = [-220, 220]
            self.work_space_z = [0, 100]
        elif self.env_name == "Tree_200":
            # note: the start and end points will be covered by
            # update_start_and_goal_pose_random function
            start_position = [0, 0, 8]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
        elif self.env_name == "SimpleAvoid":
            start_position = [0, 0, 5]
            goal_distance = 50
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi * 2
            )
            self.work_space_x = [
                start_position[0] - goal_distance - 10,
                start_position[0] + goal_distance + 10,
            ]
            self.work_space_y = [
                start_position[1] - goal_distance - 10,
                start_position[1] + goal_distance + 10,
            ]
            self.work_space_z = [0.5, 50]
        elif self.env_name == "Forest":
            start_position = [0, 0, 10]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
        elif self.env_name == "Trees":
            start_position = [0, 0, 5]
            goal_distance = 70
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi * 2
            )
            self.work_space_x = [
                start_position[0] - goal_distance - 10,
                start_position[0] + goal_distance + 10,
            ]
            self.work_space_y = [
                start_position[1] - goal_distance - 10,
                start_position[1] + goal_distance + 10,
            ]
            self.work_space_z = [0.5, 50]
        elif self.env_name == "Mountains":
            start_position = [0, 0, 10]
            self.goal_points = [
                [-0.55265, -31.9786, 19.0225],
                [48.59735, -33.3286, 30.07256],
                # [193.5974, -55.0786, 46.32256],
                # [369.2474, 35.32137, 62.5725],
                # [541.3474, 143.6714, 32.07256]
            ]
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goals(self.goal_points)
            self.work_space_x = None
            self.work_space_y = None
            self.work_space_z = None
        elif self.env_name == "Mountains_Easy":
            start_position = [0, 0, 10]
            self.goal_points = [[-0.55265, -31.9786, 19.0225]]
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goals(self.goal_points)
            self.work_space_x = [-40, 40]
            self.work_space_y = [-60, 40]
            self.work_space_z = [3, 30]
        elif self.env_name == "Custom":
            # Select start/goal pair through options.fig (1, 2, or 3).
            try:
                fig = cfg.getint("options", "fig")
            except (NoOptionError, ValueError):
                fig = 1

            custom_routes = {
                1: {
                    "start": [-23, -13, 5],
                    "goal": [23, 13, 5],
                    "yaw_offset_deg": 0,
                    "yaw_random_deg": 90,
                },
                2: {
                    "start": [-23, 0, 5],
                    "goal": [23, 0, 5],
                    "yaw_offset_deg": -60,
                    "yaw_random_deg": 120,
                },
                3: {
                    "start": [-23, 13, 5],
                    "goal": [23, -13, 5],
                    "yaw_offset_deg": -90,
                    "yaw_random_deg": 90,
                },
            }
            if fig not in custom_routes:
                raise Exception("Invalid fig for Custom env!", fig)

            route_cfg = custom_routes[fig]
            start_position = route_cfg["start"]
            self.start_position = start_position  # Save for reward calculation
            goal_position = route_cfg["goal"]
            self.dynamic_model.set_start(
                start_position,
                random_angle=math.radians(route_cfg["yaw_random_deg"]),
                yaw_offset=math.radians(route_cfg["yaw_offset_deg"]),
            )
            self.dynamic_model.set_goal_position(goal_position)
            self.work_space_x = [-25, 25]
            self.work_space_y = [-15, 15]
            self.work_space_z = [0.5, 15]
        else:
            raise Exception("Invalid env_name!", self.env_name)

        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length
        self.cnn_feature_length = self.cfg.getint("options", "cnn_feature_num")

        # training state
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.episode_steps = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_to_goal = 0.0
        self.current_distance_to_goal = 0.0

        # other settings
        self.max_episode_steps = cfg.getint("environment", "max_steps")
        self.crash_distance = cfg.getfloat("environment", "crash_distance")
        self.accept_radius = cfg.getint("environment", "accept_radius")

        self.max_depth_meters = cfg.getint("environment", "max_depth_meters")
        self.screen_height = cfg.getint("environment", "screen_height")
        self.screen_width = cfg.getint("environment", "screen_width")

        self.trajectory_list = []

        # observation space vector or image
        if self.perception_type == "vector" or self.perception_type == "lgmd":
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(1, self.cnn_feature_length + self.state_feature_length),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.screen_height, self.screen_width, 2),
                dtype=np.uint8,
            )

        self.action_space = self.dynamic_model.action_space

        self.reward_type = None
        try:
            self.reward_type = cfg.get("options", "reward_type")
            print("Reward type: ", self.reward_type)
        except NoOptionError:
            self.reward_type = None

        # learning starts settings
        self.learning_starts = cfg.getint("DRL", "learning_starts")

        # store mission start parameters
        self.mission_start_position = list(self.dynamic_model.start_position)
        self.mission_start_random_angle = self.dynamic_model.start_random_angle
        self.mission_start_yaw_offset = self.dynamic_model.start_yaw_offset

        # Compute fixed mission distance (start → goal) once.
        # Used as normalization denominator in both state and reward so the
        # scale stays consistent regardless of random start positions.
        ms = self.mission_start_position
        gp = self.dynamic_model.goal_position
        self.dynamic_model.goal_distance = math.sqrt(
            (ms[0] - gp[0]) ** 2 + (ms[1] - gp[1]) ** 2 + (ms[2] - gp[2]) ** 2
        )

    def reset(self):
        # implement random start within learning starts
        if self.total_step < self.learning_starts:
            # Random Start within mission-to-goal corridor (10m offset)
            goal_pos = self.dynamic_model.goal_position
            start_pos = self.mission_start_position

            # Calculate vector and distance in XY plane
            vec_x = goal_pos[0] - start_pos[0]
            vec_y = goal_pos[1] - start_pos[1]
            dist_xy = math.sqrt(vec_x**2 + vec_y**2)

            # Perpendicular unit vector for width offset
            unit_perp_x = -vec_y / dist_xy
            unit_perp_y = vec_x / dist_xy

            while True:
                # Alpha: progress along the line (0.0 to 0.9 to keep at least
                # 10% of mission distance from goal, preventing trivial episodes)
                # Beta: perpendicular offset (-7.5m to +7.5m)
                alpha = np.random.uniform(0.0, 0.9)
                beta = np.random.uniform(-7.5, 7.5)

                rand_x = start_pos[0] + alpha * vec_x + beta * unit_perp_x
                rand_y = start_pos[1] + alpha * vec_y + beta * unit_perp_y
                rand_z = 5.0  # Fixed height at 5 meters

                # Check if the sampled point is within workspace boundaries
                if not (
                        self.work_space_x[0] < rand_x < self.work_space_x[1]
                        and self.work_space_y[0] < rand_y < self.work_space_y[1]
                    ):
                        continue  # Out of workspace, try again

                self.dynamic_model.set_start(
                    [rand_x, rand_y, rand_z],
                    random_angle=math.pi * 2,
                    yaw_offset=0,
                )
                self.dynamic_model.reset()

                # Force synchronization: advance physics for one frame (0.01s)
                # to ensure collision info and depth image are updated
                self.client.simPause(False)
                self.client.simContinueForTime(0.01)
                self.client.simPause(True)

                # Check safety: min distance to obs >= 2.0m and no collision
                if (
                    self.get_depth_image().min() >= 2.0
                    and not self.client.simGetCollisionInfo().has_collided
                ):
                    break
                
        else:
            # Outside learning starts, always use mission start
            self.dynamic_model.set_start(
                self.mission_start_position,
                random_angle=self.mission_start_random_angle,
                yaw_offset=self.mission_start_yaw_offset,
            )
            self.dynamic_model.reset()

        self.episode_num += 1
        self.step_num = 0
        self.episode_steps = 0
        self.cumulated_episode_reward = 0
        self.total_distance = 0.0

        # Cache actual start position once; used by reward deviation penalty and
        # distance tracking.
        _start_pos = np.array(self.dynamic_model.get_position())
        self.start_position = list(_start_pos)
        self.previous_position_metric = _start_pos

        self.update_dynamic_parameters()

        actual_dist = self.get_distance_to_goal()
        self.previous_distance_to_goal = actual_dist
        self.current_distance_to_goal = actual_dist

        self.trajectory_list = []
        self._term_is_success = False
        self._term_is_crashed = False
        self._term_is_not_in_workspace = False
        self._term_is_timeout = False

        obs = self.get_obs()

        return obs

    def step(self, action):
        # set action
        if self.dynamic_name == "SimpleFixedwing":
            # add step to calculate pitch flap deg Fixed wing only
            self.dynamic_model.set_action(action, self.step_num)
        else:
            self.dynamic_model.set_action(action)

        self.goal_reached_flag = False

        if self.env_name in {"Mountains", "Mountains_Easy"}:
            prev_index = self.dynamic_model.current_goal_index
            self.dynamic_model.check_and_switch_goal(self.accept_radius)

            # 如果索引变大了，说明刚刚到达了一个中间点
            if self.dynamic_model.current_goal_index > prev_index:
                self.goal_reached_flag = True
                print(f"Goal {prev_index} reached!")

                self.update_dynamic_parameters()
                self.step_num = 0

                if self.dynamic_model.current_goal_index < len(
                    self.dynamic_model.goal_points
                ):  # 如果不是最后一个点导致结束
                    # 更新为到新目标的距离，并同步更新分母 goal_distance
                    self.current_distance_to_goal = self.get_distance_to_goal()
                    self.previous_distance_to_goal = self.current_distance_to_goal
                    self.dynamic_model.goal_distance = self.current_distance_to_goal

        position_ue4 = np.array(self.dynamic_model.get_position())
        self.trajectory_list.append(position_ue4)

        # Calculate distance traveled
        self.total_distance += np.linalg.norm(position_ue4 - self.previous_position_metric)
        self.previous_position_metric = position_ue4

        # get new obs
        obs = self.get_obs()

        # Read distance after obs so both reflect the same drone state (s_{t+1})
        self.current_distance_to_goal = self.get_distance_to_goal()

        # Increment counters before is_done() so timeout fires at the correct step
        self.step_num += 1
        self.total_step += 1
        self.episode_steps += 1

        done = self.is_done()

        # ----------------compute reward---------------------------
        if self.dynamic_name == "SimpleFixedwing":
            # reward = self.compute_reward_fixedwing(done, action)
            reward = self.compute_reward_final_fixedwing(done, action)
        elif self.reward_type == "reward_with_action":
            reward = self.compute_reward_with_action(done, action)
        elif self.reward_type == "reward_new":
            reward = self.compute_reward_multirotor_new(done, action)
        elif self.reward_type == "reward_lqr":
            reward = self.compute_reward_lqr(done, action)
        elif self.reward_type == "reward_final":
            reward = self.compute_reward_final(done, action)
        elif self.reward_type == "reward_single_goal":
            reward = self.compute_reward_single_goal(done, action)
        elif self.reward_type == "reward_custom":
            reward = self.compute_reward_custom(done, action)
        else:
            reward = self.compute_reward(done, action)

        # Advance distance reference for next step's reward calculation
        self.previous_distance_to_goal = self.current_distance_to_goal

        self.cumulated_episode_reward += reward

        info = {
            "is_success": self._term_is_success,
            "is_crash": self._term_is_crashed,
            "is_not_in_workspace": self._term_is_not_in_workspace,
            "is_timeout": self._term_is_timeout,
            "episode_steps": self.episode_steps,
            "episode_reward": self.cumulated_episode_reward,
            "episode_distance": self.total_distance,
        }
        if done:
            if info["is_success"]:
                done_reason = "reach"
            elif info["is_crash"]:
                done_reason = "crash"
            elif info["is_not_in_workspace"]:
                done_reason = "outside"
            else:
                done_reason = "timeout"
            self.episode_signal.emit(
                self.cumulated_episode_reward,
                self.episode_steps,
                done_reason,
                float(self.min_distance_to_obstacles),
            )
            print(
                f"done_reason: {done_reason}, "
                f"episode_steps: {info['episode_steps']}, "
                f"episode_reward: {info['episode_reward']:.2f}"
            )

        # ----------------print info---------------------------
        self.print_train_info_airsim(action, obs, reward, info)

        if self.cfg.get("options", "dynamic_name") == "SimpleFixedwing":
            self.set_pyqt_signal_fixedwing(action, reward, done)
        else:
            self.set_pyqt_signal_multirotor(action, reward)

        if self.keyboard_debug:
            action_copy = np.copy(action)
            action_copy[-1] = math.degrees(action_copy[-1])
            state_copy = np.copy(self.dynamic_model.state_raw)

            np.set_printoptions(formatter={"float": "{: 0.3f}".format})
            print(
                "============================================================================="
            )
            print(
                "episode",
                self.episode_num,
                "step",
                self.step_num,
                "total step",
                self.total_step,
            )
            print("action", action_copy)
            print("state", state_copy)
            print("state_norm", self.dynamic_model.state_norm)
            print("reward {:.3f} {:.3f}".format(reward, self.cumulated_episode_reward))
            print("done", done)
            keyboard.wait("a")

        if self.generate_q_map and (
            self.cfg.get("options", "algo") == "TD3"
            or self.cfg.get("options", "algo") == "SAC"
        ):
            if self.model is not None:
                with th.no_grad():
                    # get q-value for td3
                    obs_copy = obs.copy()
                    if self.perception_type != "vector":
                        obs_copy = obs_copy.swapaxes(0, 1)
                        obs_copy = obs_copy.swapaxes(0, 2)
                    q_value_current = self.model.critic(
                        th.from_numpy(obs_copy[tuple([None])]).float().cuda(),
                        th.from_numpy(action[None]).float().cuda(),
                    )
                    q_1 = q_value_current[0].cpu().numpy()[0]
                    q_2 = q_value_current[1].cpu().numpy()[0]

                    q_value = min(q_1, q_2)[0]

                    self.visual_log_q_value(q_value, action, reward)

        return obs, reward, done, info

    # ! -------------------------get obs------------------------------------------
    def get_obs(self):
        if self.perception_type == "vector":
            obs = self.get_obs_vector()
        elif self.perception_type == "lgmd":
            obs = self.get_obs_lgmd()
        else:
            obs = self.get_obs_image()

        return obs

    def get_obs_image(self):
        # Normal mode: get depth image then transfer to matrix with state
        # 1. get current depth image and transfer to 0-255  0-20m 255-0m
        image = self.get_depth_image()  # 0-6550400.0 float 32
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        self.min_distance_to_obstacles = image.min()
        # switch 0 and 255
        image_scaled = (
            np.clip(image_resize, 0, self.max_depth_meters)
            / self.max_depth_meters
            * 255
        )
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        # 2. get current state (relative_pose, velocity)
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_array[0, 0 : self.state_feature_length] = state_feature

        # 3. generate image with state
        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state

    def get_depth_gray_image(self):
        # get depth and rgb image
        # scene vision image in png format
        responses = self.client.simGetImages(
            [
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ]
        )

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                ]
            )

        # get depth image
        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width, responses[0].height
        )
        depth_meter = depth_img * 100

        # get gary image
        img_1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 3
        img_rgb = img_1d.reshape(responses[1].height, responses[1].width, 3)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('test', img_rgb)
        # cv2.waitKey(1)

        return depth_meter, img_gray

    def get_depth_image(self):
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)]
        )

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)]
            )

        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width, responses[0].height
        )

        depth_meter = depth_img * 100

        return depth_meter

    def get_obs_vector(self):
        image = self.get_depth_image()  # 0-6550400.0 float 32
        self.min_distance_to_obstacles = image.min()

        image_scaled = (
            np.clip(image, 0, self.max_depth_meters) / self.max_depth_meters * 255
        )
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        image_obs = image_uint8
        split_row = 1
        split_col = 5

        v_split_list = np.vsplit(image_obs, split_row)

        split_final = []
        for i in range(split_row):
            h_split_list = np.hsplit(v_split_list[i], split_col)
            for j in range(split_col):
                split_final.append(h_split_list[j].max())

        img_feature = np.array(split_final) / 255.0

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all

    def get_obs_lgmd(self):
        # get depth and gray image
        depth_meter, img_gray = self.get_depth_gray_image()
        self.min_distance_to_obstacles = depth_meter.min()

        self.lgmd.update(img_gray)

        split_col_num = 5
        s_layer = self.lgmd.s_layer  # (192, 320)
        s_layer_split = np.hsplit(s_layer, split_col_num)  # (192, 109)

        lgmd_out_list = []
        activate_coeff = 0.5
        for i in range(split_col_num):
            s_layer_activated_sum = abs(np.sum(s_layer_split[i]))
            Kf = -(s_layer_activated_sum * activate_coeff) / (192 * 64)  # 0 - 1
            a = np.exp(Kf)
            lgmd_out_norm = (1 / (1 + a) - 0.5) * 2
            lgmd_out_list.append(lgmd_out_norm)

        # show iamges
        heatmapshow = None
        heatmapshow = cv2.normalize(
            s_layer,
            heatmapshow,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imshow("gray image", img_gray)
        cv2.imshow("depth image", np.clip(depth_meter, 0, 255) / 255)
        cv2.imshow("s-layer", heatmapshow)
        cv2.waitKey(1)

        # update LGMD
        split_final = np.array(lgmd_out_list)

        filter_coeff = 0.8
        split_final_filter = (
            filter_coeff * split_final + (1 - filter_coeff) * self.split_out_last
        )
        self.split_out_last = split_final_filter

        img_feature = np.array(split_final_filter)

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all

    # ! ---------------------calculate rewards-------------------------------------

    def compute_reward(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if not done:
            reward_distance = (
                (self.previous_distance_to_goal - self.current_distance_to_goal)
                / self.dynamic_model.goal_distance
                * 500
            )  # normalized to 100 according to goal_distance

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.1 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * ((abs(action[1]) / self.dynamic_model.v_z_max) ** 2)
                z_err_cost = 0.05 * (
                    (
                        abs(self.dynamic_model.state_raw[1])
                        / self.dynamic_model.max_vertical_difference
                    )
                    ** 2
                )
                action_cost += v_z_cost + z_err_cost

            action_cost += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self._term_is_success:
                reward = reward_reach
            if self._term_is_crashed:
                reward = reward_crash
            if self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    def compute_reward_final(self, done, action):
        reward = 0
        reward_reach = 15
        reward_crash = -20
        reward_outside = -10

        if self.env_name == "NH_center":
            distance_reward_coef = 500
        elif self.env_name == "Mountains" or self.env_name == "Mountains_Easy":
            distance_reward_coef = 75
        else:
            distance_reward_coef = 50

        if not done:
            # 1 - goal reward
            reward_distance = (
                distance_reward_coef
                * (self.previous_distance_to_goal - self.current_distance_to_goal)
                / self.dynamic_model.goal_distance
            )  # normalized to 100 according to goal_distance

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            z = current_pose[2]
            x_g = goal_pose[0]
            y_g = goal_pose[1]
            z_g = goal_pose[2]

            prev_goal = self.dynamic_model.start_position
            if self.dynamic_model.current_goal_index > 0:
                prev_goal = self.dynamic_model.goal_points[
                    self.dynamic_model.current_goal_index - 1
                ]
            punishment_xy = np.clip(
                self.getDis(x, y, prev_goal[0], prev_goal[1], x_g, y_g) / 10, 0, 1
            )
            punishment_z = 0.5 * np.clip((z - z_g) / 5, 0, 1)

            punishment_pose = punishment_xy + punishment_z

            if self.min_distance_to_obstacles < 10:
                punishment_obs = 1 - np.clip(
                    (self.min_distance_to_obstacles - self.crash_distance) / 5, 0, 1
                )
            else:
                punishment_obs = 0

            punishment_action = 0

            # add yaw_rate cost
            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = (abs(action[1]) / self.dynamic_model.v_z_max) ** 2
                z_err_cost = (
                    abs(self.dynamic_model.state_raw[1])
                    / self.dynamic_model.max_vertical_difference
                ) ** 2
                punishment_action += v_z_cost + z_err_cost

            punishment_action += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            reward = (
                reward_distance
                - 0.3 * punishment_pose
                - 0.2 * punishment_obs
                - 0.1 * punishment_action
                - 0.5 * yaw_error_cost
            )

            if hasattr(self, "goal_reached_flag") and self.goal_reached_flag:
                reward += reward_reach

        else:
            if self._term_is_success:
                reward = reward_reach
            if self._term_is_crashed:
                reward = reward_crash
            if self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    def compute_reward_single_goal(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if self.env_name == "NH_center":
            distance_reward_coef = 500
        else:
            distance_reward_coef = 50

        if not done:
            # 1 - goal reward
            reward_distance = (
                distance_reward_coef
                * (self.previous_distance_to_goal - self.current_distance_to_goal)
                / self.dynamic_model.goal_distance
            )  # normalized to 100 according to goal_distance

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            z = current_pose[2]
            x_g = goal_pose[0]
            y_g = goal_pose[1]
            z_g = goal_pose[2]
            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip((z - z_g) / 5, 0, 1)

            punishment_pose = punishment_xy + punishment_z

            if self.min_distance_to_obstacles < 10:
                punishment_obs = 1 - np.clip(
                    (self.min_distance_to_obstacles - self.crash_distance) / 5, 0, 1
                )
            else:
                punishment_obs = 0

            punishment_action = 0

            # add yaw_rate cost
            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = (abs(action[1]) / self.dynamic_model.v_z_max) ** 2
                z_err_cost = (
                    abs(self.dynamic_model.state_raw[1])
                    / self.dynamic_model.max_vertical_difference
                ) ** 2
                punishment_action += v_z_cost + z_err_cost

            punishment_action += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            reward = (
                reward_distance
                - 0.1 * punishment_pose
                - 0.2 * punishment_obs
                - 0.1 * punishment_action
                - 0.5 * yaw_error_cost
            )
        else:
            if self._term_is_success:
                reward = reward_reach
            if self._term_is_crashed:
                reward = reward_crash
            if self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    def compute_reward_custom(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if self.env_name == "NH_center":
            distance_reward_coef = 500
        else:
            distance_reward_coef = 50

        if not done:
            # 1 - goal reward
            reward_distance = (
                distance_reward_coef
                * (self.previous_distance_to_goal - self.current_distance_to_goal)
                / self.dynamic_model.goal_distance
            )  # normalized to 100 according to goal_distance

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            z = current_pose[2]
            x_g = goal_pose[0]
            y_g = goal_pose[1]
            z_g = goal_pose[2]

            # Use actual start position for calculating deviation from the straight line
            x_s = self.start_position[0]
            y_s = self.start_position[1]

            # Penalty saturates at 10m deviation from start-to-goal line
            punishment_xy = np.clip(self.getDis(x, y, x_s, y_s, x_g, y_g) / 10, 0, 1)
            # Maintain Z alignment punishment
            punishment_z = 0.5 * np.clip((z - z_g) / 5, 0, 1)

            punishment_trajectory_deviation = punishment_xy + punishment_z

            if self.min_distance_to_obstacles < 2.0:
                punishment_obs = 1 - np.clip(
                    (self.min_distance_to_obstacles - self.crash_distance)
                    / (2.0 - self.crash_distance),
                    0,
                    1,
                )
            else:
                punishment_obs = 0

            punishment_action = 0

            # add yaw_rate cost
            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = (abs(action[1]) / self.dynamic_model.v_z_max) ** 2
                z_err_cost = (
                    abs(self.dynamic_model.state_raw[1])
                    / self.dynamic_model.max_vertical_difference
                ) ** 2
                punishment_action += v_z_cost + z_err_cost

            punishment_action += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            reward = (
                reward_distance
                - 0.1 * punishment_trajectory_deviation
                - 0.2 * punishment_obs
                - 0.1 * punishment_action
                - 0.5 * yaw_error_cost
            )
        else:
            if self._term_is_success:
                reward = reward_reach
            elif self._term_is_crashed:
                reward = reward_crash
            elif self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    def compute_reward_final_fixedwing(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if not done:
            # 1 - goal reward
            reward_distance = (
                300
                * (self.previous_distance_to_goal - self.current_distance_to_goal)
                / self.dynamic_model.goal_distance
            )  # normalized to 100 according to goal_distance

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            x_g = goal_pose[0]
            y_g = goal_pose[1]

            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 50, 0, 1)
            # punishment_z = 0.5 * np.clip((z - z_g)/5, 0, 1)

            punishment_pose = punishment_xy

            if self.min_distance_to_obstacles < 20:
                punishment_obs = 1 - np.clip(
                    (self.min_distance_to_obstacles - self.crash_distance) / 15, 0, 1
                )
            else:
                punishment_obs = 0

            # action cost
            punishment_action = abs(action[0]) / self.dynamic_model.roll_rate_max

            yaw_error = self.dynamic_model.state_raw[1]
            yaw_error_cost = abs(yaw_error / 90)

            reward = (
                reward_distance
                - 0.1 * punishment_pose
                - 0.5 * punishment_obs
                - 0.1 * punishment_action
                - 0.1 * yaw_error_cost
            )
            # reward = reward

            # print("r_dist: {:.2f} p_pose: {:.2f} p_obs: {:.2f} p_action: {:.2f}, p_yaw_e: {:.2f}".format(reward_distance, punishment_pose, punishment_obs, punishment_action, yaw_error_cost))
        else:
            if self._term_is_success:
                reward = reward_reach
            if self._term_is_crashed:
                reward = reward_crash
            if self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    def compute_reward_test(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -100
        reward_outside = -10

        if not done:
            reward_distance = (
                (self.previous_distance_to_goal - self.current_distance_to_goal)
                / self.dynamic_model.goal_distance
                * 100
            )  # normalized to 100 according to goal_distance

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.1 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * abs(action[1]) / self.dynamic_model.v_z_max
                z_err_cost = (
                    0.05
                    * abs(self.dynamic_model.state_raw[1])
                    / self.dynamic_model.max_vertical_difference
                )
                action_cost += v_z_cost + z_err_cost

            action_cost += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self._term_is_success:
                reward = reward_reach
            if self._term_is_crashed:
                reward = reward_crash
            if self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    def compute_reward_fixedwing(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -50
        reward_outside = -10

        if not done:
            reward_distance = (
                (self.previous_distance_to_goal - self.current_distance_to_goal)
                / self.dynamic_model.goal_distance
                * 300
            )  # normalized to 100 according to goal_distance

            # 只有action cost和obs cost
            # 由于没有速度控制，所以前面那个也取消了
            # action_cost = 0
            # obs_cost = 0

            # relative_yaw_cost = abs(
            #     (self.dynamic_model.state_norm[0]/255-0.5) * 2)
            # action_cost = abs(action[0]) / self.dynamic_model.roll_rate_max

            # obs_punish_distance = 15
            # if self.min_distance_to_obstacles < obs_punish_distance:
            #     obs_cost = 1 - (self.min_distance_to_obstacles -
            #                     self.crash_distance) / (obs_punish_distance -
            #                                             self.crash_distance)
            #     obs_cost = 0.5 * obs_cost ** 2
            # reward = reward_distance - (2 * relative_yaw_cost + 0.5 * action_cost + obs_cost)

            action_cost = abs(action[0]) / self.dynamic_model.roll_rate_max

            yaw_error_deg = self.dynamic_model.state_raw[1]
            yaw_error_cost = 0.1 * abs(yaw_error_deg / 180)

            reward = reward_distance - action_cost - yaw_error_cost
        else:
            if self._term_is_success:
                yaw_error_deg = self.dynamic_model.state_raw[1]
                reward = reward_reach * (1 - abs(yaw_error_deg / 180))
                # reward = reward_reach
            if self._term_is_crashed:
                reward = reward_crash
            if self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    def compute_reward_multirotor_new(self, done, action):
        reward = 0
        reward_reach = 100
        reward_crash = -100
        reward_outside = 0

        if not done:
            reward_distance = (
                (self.previous_distance_to_goal - self.current_distance_to_goal)
                / self.dynamic_model.goal_distance
                * 5
            )

            state_cost = 0
            action_cost = 0
            obs_cost = 0

            yaw_error_deg = self.dynamic_model.state_raw[1]

            relative_yaw_cost = abs(yaw_error_deg / 180)
            action_cost = abs(action[1]) / self.dynamic_model.yaw_rate_max_rad

            obs_punish_dist = 5
            if self.min_distance_to_obstacles < obs_punish_dist:
                obs_cost = 1 - (
                    self.min_distance_to_obstacles - self.crash_distance
                ) / (obs_punish_dist - self.crash_distance)
                obs_cost = 0.5 * obs_cost**2
            reward = -(2 * relative_yaw_cost + 0.5 * action_cost)
        else:
            if self._term_is_success:
                # 到达之后根据yaw偏差对reward进行scale
                reward = reward_reach * (1 - abs(self.dynamic_model.state_norm[1]))
                # reward = reward_reach
            if self._term_is_crashed:
                reward = reward_crash
            if self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    def compute_reward_with_action(self, done, action):
        reward = 0
        reward_reach = 50
        reward_crash = -50
        reward_outside = -10

        step_cost = 0.01  # 10 for max 1000 steps

        if not done:
            reward_distance = (
                (self.previous_distance_to_goal - self.current_distance_to_goal)
                / self.dynamic_model.goal_distance
                * 10
            )  # normalized to 100 according to goal_distance

            reward_obs = 0
            action_cost = 0

            # add action cost
            # speed 0-8  cruise speed is 4, punish for too fast and too slow
            v_xy_cost = 0.02 * abs(action[0] - 5) / 4
            yaw_rate_cost = 0.02 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.02 * abs(action[1]) / self.dynamic_model.v_z_max
                action_cost += v_z_cost
            action_cost += v_xy_cost + yaw_rate_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.05 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self._term_is_success:
                reward = reward_reach
            if self._term_is_crashed:
                reward = reward_crash
            if self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    def compute_reward_lqr(self, done, action):
        # 模仿matlab提供的mix reward的思想设计
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = 0

        if not done:
            action_cost = 0
            # add yaw_rate cost
            yaw_speed_cost = 0.2 * (
                (action[-1] / self.dynamic_model.yaw_rate_max_rad) ** 2
            )

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * ((action[1] / self.dynamic_model.v_z_max) ** 2)
                z_err_cost = 0.1 * (
                    (
                        self.dynamic_model.state_raw[1]
                        / self.dynamic_model.max_vertical_difference
                    )
                    ** 2
                )
                action_cost += v_z_cost + z_err_cost

            action_cost += yaw_speed_cost

            yaw_error_clip = min(max(-60, self.dynamic_model.state_raw[2]), 60) / 60
            yaw_error_cost = 1.0 * (yaw_error_clip**2)

            reward = -(action_cost + yaw_error_cost)

            # print('r: {:.2f} y_r: {:.2f} y_e: {:.2f} z_r: {:.2f} z_e: {:.2f}'.format(reward, yaw_speed_cost, yaw_error_cost, v_z_cost, z_err_cost))
        else:
            if self._term_is_success:
                yaw_error_clip = min(max(-30, self.dynamic_model.state_raw[2]), 30) / 30
                reward = reward_reach * (1 - yaw_error_clip**2)
            if self._term_is_crashed:
                reward = reward_crash
            if self._term_is_not_in_workspace:
                reward = reward_outside

        return reward

    # ! ------------------ is done-----------------------------------------------

    def is_done(self):
        episode_done = False

        is_not_inside_workspace_now = self.is_not_inside_workspace()
        too_close_to_obstable = self.is_crashed()

        has_reached_des_pos = False
        if self.env_name in {"Mountains", "Mountains_Easy"}:
            # 如果索引已经超过了列表长度，说明所有目标都跑完了
            if self.dynamic_model.current_goal_index >= len(
                self.dynamic_model.goal_points
            ):
                has_reached_des_pos = True
        else:
            has_reached_des_pos = self.is_in_desired_pose()

        is_timeout = self.step_num >= self.max_episode_steps

        # We see if we are outside the Learning Space
        episode_done = (
            is_not_inside_workspace_now
            or has_reached_des_pos
            or too_close_to_obstable
            or is_timeout
        )

        # Cache all sub-conditions so info and compute_reward can reuse them
        # without re-querying AirSim (drone state may change after this call).
        self._term_is_success = has_reached_des_pos
        self._term_is_crashed = too_close_to_obstable
        self._term_is_not_in_workspace = is_not_inside_workspace_now
        self._term_is_timeout = is_timeout

        return episode_done

    def is_not_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        current_position = self.dynamic_model.get_position()

        if (
            current_position[0] < self.work_space_x[0]
            or current_position[0] > self.work_space_x[1]
            or current_position[1] < self.work_space_y[0]
            or current_position[1] > self.work_space_y[1]
            or current_position[2] < self.work_space_z[0]
            or current_position[2] > self.work_space_z[1]
        ):
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):
        return self.current_distance_to_goal < self.accept_radius

    def is_crashed(self):
        collision_info = self.client.simGetCollisionInfo()
        return (
            collision_info.has_collided
            or self.min_distance_to_obstacles < self.crash_distance
        )

    # ! ----------- useful functions-------------------------------------------
    def get_distance_to_goal(self):
        if self.dynamic_model.navigation_3d:
            if hasattr(self.dynamic_model, "get_distance_to_goal_3d"):
                return self.dynamic_model.get_distance_to_goal_3d()
        else:
            if hasattr(self.dynamic_model, "get_distance_to_goal_2d"):
                return self.dynamic_model.get_distance_to_goal_2d()

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        """
        Get distance between Point and Line
        Used to calculate position punishment
        """
        a = lineY2 - lineY1
        b = lineX1 - lineX2
        c = lineX2 * lineY1 - lineX1 * lineY2
        dis = (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))

        return dis

    # ! -----------used for plot or show states------------------------------------------------------------------

    def print_train_info_airsim(self, action, obs, reward, info):
        # if self.perception_type == 'split' or self.perception_type == 'lgmd':
        #     feature_all = self.feature_all
        # elif self.perception_type == 'vector':
        #     feature_all = self.feature_all
        # else:
        #     if self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC':
        #         feature_all = self.model.actor.features_extractor.feature_all
        #     elif self.cfg.get('options', 'algo') == 'PPO':
        #         feature_all = self.model.policy.features_extractor.feature_all

        # self.client.simPrintLogMessage('feature_all: ', str(feature_all))

        msg_train_info = "EP: {} Step: {} Total_step: {}".format(
            self.episode_num, self.step_num, self.total_step
        )

        self.client.simPrintLogMessage("Train: ", msg_train_info)
        self.client.simPrintLogMessage("Action: ", str(action))
        self.client.simPrintLogMessage(
            "reward: ",
            "{:4.4f} total: {:4.4f}".format(reward, self.cumulated_episode_reward),
        )
        self.client.simPrintLogMessage("Info: ", str(info))
        self.client.simPrintLogMessage(
            "Feature_norm: ", str(self.dynamic_model.state_norm)
        )
        self.client.simPrintLogMessage(
            "Feature_raw: ", str(self.dynamic_model.state_raw)
        )
        self.client.simPrintLogMessage(
            "Min_depth: ", str(self.min_distance_to_obstacles)
        )

    def set_pyqt_signal_fixedwing(self, action, reward, done):
        """
        emit signals for pyqt plot
        """
        step = int(self.total_step)
        # action: v_xy, v_z, roll

        action_plot = np.array([10, 0, math.degrees(action[0])])

        state = self.dynamic_model.state_raw  # distance, relative yaw, roll

        # state out 6: d_xy, d_z, yaw_error, v_xy, v_z, roll
        # state in  3: d_xy, yaw_error, roll
        state_output = np.array([state[0], 0, state[1], 10, 0, state[2]])

        # self.action_signal.emit(step, action_plot)
        # self.state_signal.emit(step, state_output)

        # other values
        # self.attitude_signal.emit(
        #     step,
        #     np.asarray(self.dynamic_model.get_attitude()),
        #     np.asarray(self.dynamic_model.get_attitude_cmd()),
        # )
        # self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(
            np.asarray(self.dynamic_model.goal_position),
            np.asarray(self.dynamic_model.start_position),
            np.asarray(self.dynamic_model.get_position()),
            np.asarray(self.trajectory_list),
            np.asarray(self.dynamic_model.get_velocity_vector()),
        )

        # lgmd_signal = pyqtSignal(float, float, np.ndarray)  min_dist, lgmd_out, lgmd_split
        # self.lgmd_signal.emit(self.min_distance_to_obstacles, 0, self.feature_all[:-1])

    def set_pyqt_signal_multirotor(self, action, reward):
        step = int(self.total_step)

        # transfer 2D state and action to 3D
        state = self.dynamic_model.state_raw
        if self.dynamic_model.navigation_3d:
            action_output = action
            state_output = state
        else:
            action_output = np.array([action[0], 0, action[1]])
            state_output = np.array([state[0], 0, state[2], state[3], 0, state[5]])

        # self.action_signal.emit(step, action_output)
        # self.state_signal.emit(step, state_output)

        # other values
        # self.attitude_signal.emit(
        #     step,
        #     np.asarray(self.dynamic_model.get_attitude()),
        #     np.asarray(self.dynamic_model.get_attitude_cmd()),
        # )
        # self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(
            np.asarray(self.dynamic_model.goal_position),
            np.asarray(self.dynamic_model.start_position),
            np.asarray(self.dynamic_model.get_position()),
            np.asarray(self.trajectory_list),
            np.asarray(self.dynamic_model.get_velocity_vector()),
        )

    def visual_log_q_value(self, q_value, action, reward):
        """
        Create grid map (map_size = work_space)
        Log Q value and the best action in grid map
        At any grid position, record:
        1. Q value
        2. action 0
        3. action 1
        4. steps
        5. reward
        Save image every 10k steps
        Used only for 2D explanation
        """

        # create init array if not exist
        map_size_x = self.work_space_x[1] - self.work_space_x[0]
        map_size_y = self.work_space_y[1] - self.work_space_y[0]
        if not hasattr(self, "q_value_map"):
            self.q_value_map = np.full((9, map_size_x + 1, map_size_y + 1), np.nan)

        # record info
        position = self.dynamic_model.get_position()
        pose_x = position[0]
        pose_y = position[1]

        index_x = int(np.round(pose_x) + self.work_space_x[1])
        index_y = int(np.round(pose_y) + self.work_space_y[1])

        # check if index valid
        if index_x in range(0, map_size_x) and index_y in range(0, map_size_y):
            self.q_value_map[0, index_x, index_y] = q_value
            self.q_value_map[1, index_x, index_y] = action[0]
            self.q_value_map[2, index_x, index_y] = action[-1]
            self.q_value_map[3, index_x, index_y] = self.total_step
            self.q_value_map[4, index_x, index_y] = reward
            self.q_value_map[5, index_x, index_y] = q_value
            self.q_value_map[6, index_x, index_y] = action[0]
            self.q_value_map[7, index_x, index_y] = action[-1]
            self.q_value_map[8, index_x, index_y] = reward
        else:
            print(
                "Error: X:{} and Y:{} is outside of range 0~mapsize (visual_log_q_value)"
            )

        # save array every record_step steps
        record_step = self.cfg.getint("options", "q_map_save_steps")
        if (self.total_step + 1) % record_step == 0:
            if self.data_path is not None:
                np.save(
                    self.data_path + "/q_value_map_{}".format(self.total_step + 1),
                    self.q_value_map,
                )
                # refresh 5 6 7 8 to record period data
                self.q_value_map[5, :, :] = np.nan
                self.q_value_map[6, :, :] = np.nan
                self.q_value_map[7, :, :] = np.nan
                self.q_value_map[8, :, :] = np.nan

    def update_dynamic_parameters(self):
        if self.env_name != "Mountains":
            return

        current_idx = self.dynamic_model.current_goal_index
        if current_idx >= len(self.dynamic_model.goal_points):
            return

        # 获取当前目标和上一位置
        current_goal = self.dynamic_model.goal_points[current_idx]
        if current_idx == 0:
            prev_point = self.dynamic_model.start_position
        else:
            prev_point = self.dynamic_model.goal_points[current_idx - 1]

        # 1. XY 轴范围 (保持之前的逻辑)
        margin_xy = 30.0
        x_min = min(prev_point[0], current_goal[0]) - margin_xy
        x_max = max(prev_point[0], current_goal[0]) + margin_xy
        y_min = min(prev_point[1], current_goal[1]) - margin_xy
        y_max = max(prev_point[1], current_goal[1]) + margin_xy
        self.work_space_x = [x_min, x_max]
        self.work_space_y = [y_min, y_max]

        # 2. Z 轴范围 (针对你的提议进行的修正)
        z_min = 5.0  # 你的提议：最低 5米 (很棒的设置)
        margin_z = 10.0  # 额外的安全边距
        z_max = max(prev_point[2], current_goal[2]) + margin_z

        self.work_space_z = [z_min, z_max]

        dist = math.dist(prev_point, current_goal)
        self.max_episode_steps = (dist / self.dynamic_model.dt // 100 + 1) * 100
