import json
import time
import threading
import config
import numpy as np
from threading import Thread
from pathlib import Path

import mujoco
from mujoco import viewer

from ONNXPolicy import ONNXPolicy
from G1Controller import G1Controller
from utilities import set_armature, _solve_damped_pseudoinverse_dq, _joint_state_refs


SCRIPT_DIR = Path(__file__).resolve().parent

def on_key(keycode: int) -> None:
  ctrl.key_callback(keycode)

# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #

locker = threading.Lock()
selected_scene = config.get_robot_scene()
print(f"[CONFIG] Using scene: {selected_scene}")
mj_model = mujoco.MjModel.from_xml_path(selected_scene)

# Load config
config_path = SCRIPT_DIR / "policy_resources/model_config.json"
with open(config_path) as f:
    config_robot = json.load(f)
joint_names = config_robot["joint_names"]

mj_model.opt.timestep = config.SIMULATE_DT
set_armature(mj_model, joint_names)

mj_data = mujoco.MjData(mj_model)

# Init robot pose — spawn behind the table, facing it
mj_data.qpos[0] = -0.6  # x: back from table
mj_data.qpos[2] = 0.76
mj_data.qpos[3:7] = [1, 0, 0, 0]
for name, value in config_robot["default_joint_pos"].items():
    if name in joint_names:
        mj_data.qpos[7 + joint_names.index(name)] = value

# Startup right-arm posture preload: elbox and flexed, forearm and hand lifted
right_arm_spawn_offsets = {
  "right_shoulder_pitch_joint": -0.3,
  "right_shoulder_roll_joint": -0.2,
  "right_elbow_joint": -0.2,
  "right_wrist_pitch_joint": -0.1,
}
for joint_name, delta in right_arm_spawn_offsets.items():
  if joint_name in joint_names:
    qpos_idx = 7 + joint_names.index(joint_name)
    mj_data.qpos[qpos_idx] += delta

mujoco.mj_forward(mj_model, mj_data)

# Load walker policy
print("Loading ONNX policies...")
walker = ONNXPolicy(str(SCRIPT_DIR / "policy_resources/walker.onnx"))

# Create right arm controller #
# Recover initial object to be grasped

# Desired palm orientation in world frame for top-down grasping (Blue cube).
# A local +X roll (negative for inward) is applied like wrist_roll.
wrist_roll_deg = -90.0
roll = np.deg2rad(wrist_roll_deg)
cr1, sr1 = np.cos(roll), np.sin(roll)

roll_local_x = np.array([
  [1.0, 0.0, 0.0],
  [0.0, cr1, -sr1],
  [0.0, sr1, cr1],
], dtype=np.float32)

# Desired palm orientation in world frame for tilted grasping (White mug).
# A local +Z yaw is applied like wrist_yaw.
wrist_yaw_deg = 20.0
yaw = np.deg2rad(wrist_yaw_deg)
cr2, sr2 = np.cos(yaw), np.sin(yaw)

yaw_local_z = np.array([
  [cr2, -sr2, 0.0],
  [sr2, cr2,  0.0],
  [0.0, 0.0,  1.0],
], dtype=np.float32)

# Palm orientation after transportation to ease object relocation.
wrist_yaw_deg = -30.0
yaw = np.deg2rad(wrist_yaw_deg)
cr2, sr2 = np.cos(yaw), np.sin(yaw)

yaw_local_z_after_transport = np.array([
  [cr2, -sr2, 0.0],
  [sr2, cr2,  0.0],
  [0.0, 0.0,  1.0],
], dtype=np.float32)

# Default palm orientation (Red cylinder).
no_rotation = np.array([
  [1.0, 0.0, 0.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.0, 1.0],
], dtype=np.float32)

target_body_name = None
for candidate in ("red_cylinder", "mug_object"):
  bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, candidate)
  if bid >= 0:
    target_body_id = bid
    target_body_name = candidate
    if target_body_name == "red_cylinder":
      ik_target_x_offset_world = 0.02 # offset along x direction cylinder center
      ik_target_y_offset_world = -0.038 # no offset along y direction cylinder center
      ik_target_z_offset_world = 0.0  # no offset above cylinder center
      grasp_rot_world = (no_rotation).astype(np.float32)
    elif target_body_name == "mug_object":
      ik_target_x_offset_world = 0.01 # offset along x direction cylinder center
      ik_target_y_offset_world = -0.14 # no offset along y direction cylinder center
      ik_target_z_offset_world = 0.0  # no offset above cylinder center
      grasp_rot_world = (yaw_local_z).astype(np.float32)
    break
else:
  raise RuntimeError("Body not found in model: red_cylinder/mug_object")
print(f"[IK] Tracking target body: {target_body_name}")

ctrl = G1Controller(mj_model, mj_data, walker, config_robot, target_body_name)
is_scene1_task = (target_body_name == "red_cylinder")
is_scene2_task = (target_body_name == "mug_object")
ctrl.manual_grip_enabled = False

# Warm up ONNX models (first call triggers JIT compilation)
_dummy99 = np.zeros((1, 99), dtype=np.float32)
walker(_dummy99)

viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=on_key)


def SimulationThread():
  global mj_data, mj_model

  right_arm_joint_refs = _joint_state_refs(mj_model, ctrl.right_arm_joint_names)
  right_arm_q_des = np.array([
    float(mj_data.qpos[right_arm_joint_refs[name][0]])
    for name in ctrl.right_arm_joint_names
  ], dtype=np.float32)

  right_arm_q_min = []
  right_arm_q_max = []
  for joint_name in ctrl.right_arm_joint_names:
    joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id >= 0 and mj_model.jnt_limited[joint_id]:
      right_arm_q_min.append(float(mj_model.jnt_range[joint_id][0]))
      right_arm_q_max.append(float(mj_model.jnt_range[joint_id][1]))
    else:
      right_arm_q_min.append(-np.inf)
      right_arm_q_max.append(np.inf)
  right_arm_q_min = np.asarray(right_arm_q_min, dtype=np.float32)
  right_arm_q_max = np.asarray(right_arm_q_max, dtype=np.float32)

  # Apply startup posture on the commanded arm target as well (not only qpos spawn), 
  # this way joint positions are kept until initial movmeent.
  startup_arm_cmd_offsets = {
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": -0.2,
    "right_wrist_pitch_joint": -0.1,
  }
  for i, joint_name in enumerate(ctrl.right_arm_joint_names):
    right_arm_q_des[i] += float(startup_arm_cmd_offsets.get(joint_name, 0.0))
  right_arm_q_des = np.clip(right_arm_q_des, right_arm_q_min, right_arm_q_max)
  # Keep this startup posture as the nominal right-arm command before IK takeover.
  ctrl.frozen_arm_pos = right_arm_q_des.copy()

  ik_enabled = True
  ik_damping = 0.08
  ik_pos_gain = 2.0
  ik_rot_gain = 2.0
  ik_max_joint_speed = 1.8  # rad/s
  ik_startup_delay_s = 0.25  # keep startup posture briefly, then start IK pursuit
  ik_hold_distance = 0.015  # latch/hold when palm is this close to target
  ik_hold_rot_deg = 4.0  # require tighter orientation convergence before hold
  ik_hold_rot_rad = np.deg2rad(ik_hold_rot_deg)
  ik_hold_min_cycles = 15  # consecutive control cycles before hold latch
  ik_hold_active = False
  ik_hold_ready_count = 0
  ik_hold_target_world = None
  post_grasp_lift_delay_s = 4.0
  post_grasp_lift_z_world = 0.07
  post_grasp_lift_duration_s = 1.2
  current_grasp_rot_world = grasp_rot_world.copy()

  # Scene1 task state machine (red cylinder from cube -> blue basket)
  task1_step = 0
  task1_pos_thresh = 0.003
  task1_pos_thresh2 = 0.01
  task1_cube_grasp_pos_thresh = 0.015
  task1_rot_thresh = 0.08
  task1_transfer_delay_s = 2.0
  task1_retarget_cube_delay_s = 1.0
  task1_cube_grip_settle_delay_s = 2.0
  task1_cube_lift_z_tolerance = 0.008
  task1_cube_lift_max_wait_s = 2.0
  task1_close_time = None
  task1_drop_open_time = None
  task1_transfer_target_world = None
  task1_second_transfer_target_world = None
  task1_blue_basket_body_id = -1
  task1_red_basket_body_id = -1
  task1_blue_cube_body_id = -1
  task1_cube_x_offset = 0.019
  task1_cube_y_offset = 0.0
  task1_cube_z_offset = 0.08
  if is_scene1_task:
    task1_blue_basket_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "blue_basket")
    if task1_blue_basket_body_id < 0:
      raise RuntimeError("Body not found in model: blue_basket")
    task1_red_basket_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "red_basket")
    if task1_red_basket_body_id < 0:
      raise RuntimeError("Body not found in model: red_basket")
    task1_blue_cube_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube")
    if task1_blue_cube_body_id < 0:
      raise RuntimeError("Body not found in model: blue_cube")

  # Scene2 task state machine (mug handle grasp -> stabilize -> vertical lift)
  task2_step = 0
  task2_pos_thresh = 0.015
  task2_rot_thresh = np.deg2rad(0.2)
  task2_grip_settle_delay_s = 2.0
  task2_post_lift_wait_s = 2.0
  task2_post_descent_wait_s = 3.0
  task2_release_pos_thresh = 0.025
  task2_descend_pos_thresh = 0.01
  task2_distancing_pos_thresh = 0.02
  task2_close_time = None
  task2_lift_done_time = None
  task2_hold_target_world = None
  task2_pre_grasp_palm_z = None
  task2_transfer_target_world = None
  task2_release_target_world = None
  task2_opening_target_world = None
  task2_coaster_body_id = -1
  if is_scene2_task:
    task2_coaster_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "coaster")
    if task2_coaster_body_id < 0:
      raise RuntimeError("Body not found in model: coaster")

  # -------------------------------------------------------------------- #

  print("nu =", mj_model.nu)
  print("nq =", mj_model.nq)
  print("nv =", mj_model.nv)

  # ------------------------------------------------------------------- #
  # Simulation loop using launch_passive (MuJoCo's built-in viewer)
  # ------------------------------------------------------------------- #

  decimation = 4
  ik_dt = float(mj_model.opt.timestep * decimation)
  control_step = 0
  target_pos = ctrl.default_joint_pos.copy()
  sim_time = 0.0

  print("Launching MuJoCo viewer...")

  # Reset clock AFTER viewer opens — prevents catchup lag burst on startup
  t0 = time.time()
  ik_enable_time = t0 + ik_startup_delay_s
  while viewer.is_running():
    # Step physics in real time (cap catchup to avoid jitter snowball)
    wall = time.time() - t0
    max_catchup = 0.05  # Never try to catch up more than 50ms per frame
    if wall - sim_time > max_catchup:
      sim_time = wall - max_catchup
    while sim_time < wall:

      locker.acquire()

      if control_step % decimation == 0:
        target_pos = ctrl.step()

        ik_can_run = ik_enabled and (time.time() >= ik_enable_time)
        if ik_can_run:
          ee_pos = ctrl._get_palm_pos_in_pelvis().astype(np.float32)

          # Manual-only scenes: after closing fingers, wait a bit and then lift.
          if (not is_scene1_task) and (not is_scene2_task) and (
            ctrl.grip_closed
            and (ctrl.grip_close_time is not None)
            and (not ctrl.post_grasp_lift_active)
            and ((time.time() - ctrl.grip_close_time) >= post_grasp_lift_delay_s)
          ):
            lift_start_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
            lift_final_world = lift_start_world.copy()
            lift_final_world[2] += post_grasp_lift_z_world
            ctrl.post_grasp_lift_start_world = lift_start_world
            ctrl.post_grasp_lift_final_world = lift_final_world
            ctrl.post_grasp_lift_start_time = time.time()
            ctrl.post_grasp_lift_target_world = lift_start_world.copy()
            ctrl.post_grasp_lift_active = True
            ik_hold_active = False
            ik_hold_ready_count = 0
            ik_hold_target_world = None
            print(
              f"[IK] Post-grasp lift triggered (+{post_grasp_lift_z_world:.3f} m in world Z, {post_grasp_lift_duration_s:.2f}s)"
            )

          if is_scene1_task and (task1_step == 2) and (task1_transfer_target_world is not None):
            target_world = task1_transfer_target_world.copy()
          elif is_scene1_task and (task1_step in (4, 5)):
            cube_world = mj_data.xpos[task1_blue_cube_body_id].copy()
            cube_world[0] -= task1_cube_x_offset
            cube_world[1] += task1_cube_y_offset
            cube_world[2] += task1_cube_z_offset
            target_world = cube_world
          elif is_scene1_task and (task1_step in (7, 8)) and (task1_second_transfer_target_world is not None):
            target_world = task1_second_transfer_target_world.copy()
          elif is_scene2_task and (task2_step in (1, 3)) and (task2_hold_target_world is not None):
            target_world = task2_hold_target_world.copy()
          elif is_scene2_task and (task2_step == 4) and (task2_transfer_target_world is not None):
            target_world = task2_transfer_target_world.copy()
          elif is_scene2_task and (task2_step == 5) and (task2_release_target_world is not None):
            target_world = task2_release_target_world.copy()
          elif is_scene2_task and (task2_step == 6) and (task2_opening_target_world is not None):
            target_world = task2_opening_target_world.copy()
          elif is_scene2_task and (task2_step == 7) and (task2_distancing_target_world is not None):
            target_world = task2_distancing_target_world.copy()
          elif ctrl.post_grasp_lift_active and (ctrl.post_grasp_lift_target_world is not None):
            # Handling post-grasp lift by blending from lift start to lift final target to provide 
            # a smooth transition and avoid IK jumps at lift start/end (initially post_grasp_lift_target_world=post_grasp_lift_start_world).
            if (
              (ctrl.post_grasp_lift_start_time is not None)
              and (ctrl.post_grasp_lift_start_world is not None)
              and (ctrl.post_grasp_lift_final_world is not None)
            ):
              lift_elapsed = time.time() - ctrl.post_grasp_lift_start_time
              tau = float(np.clip(lift_elapsed / post_grasp_lift_duration_s, 0.0, 1.0))
              # Smoothstep to avoid velocity jumps at lift start/end.
              blend = tau * tau * (3.0 - 2.0 * tau)
              ctrl.post_grasp_lift_target_world = (
                (1.0 - blend) * ctrl.post_grasp_lift_start_world
                + blend * ctrl.post_grasp_lift_final_world
              )
            target_world = ctrl.post_grasp_lift_target_world.copy()
          elif ik_hold_active and (ik_hold_target_world is not None):
            # Keep tracking the latched pose instead of chasing object motion.
            target_world = ik_hold_target_world.copy()
          else:
            target_world = mj_data.xpos[target_body_id].copy()
            target_world[0] -= ik_target_x_offset_world
            target_world[1] += ik_target_y_offset_world
            target_world[2] += ik_target_z_offset_world

          base_pos_world, base_quat_world = ctrl._get_base_pose()
          ik_goal_pos_pelvis = ctrl._quat_apply_inverse(
            base_quat_world,
            target_world - base_pos_world,
          ).astype(np.float32)
          world_to_pelvis = ctrl._quat_to_rotmat(base_quat_world).T
          ik_goal_rot_pelvis = (world_to_pelvis @ current_grasp_rot_world).astype(np.float32)

          pos_err = ik_goal_pos_pelvis - ee_pos
          pos_err_norm = float(np.linalg.norm(pos_err))

          # Orientation convergence check for hold latch.
          bRe = ctrl._get_palm_rot_in_pelvis_mat().astype(np.float64)
          R_err_hold = bRe.T @ ik_goal_rot_pelvis.astype(np.float64)
          _, theta_hold = ctrl._rot_to_angle_axis(R_err_hold)

          # Scene1 task progression
          if is_scene1_task:
            if task1_step == 0:
              if (pos_err_norm <= task1_pos_thresh) and (theta_hold <= task1_rot_thresh):
                ctrl.set_grip_state(True)
                task1_close_time = time.time()
                task1_step = 1

                lift_start_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                lift_final_world = lift_start_world.copy()
                lift_final_world[2] += post_grasp_lift_z_world
                ctrl.post_grasp_lift_start_world = lift_start_world
                ctrl.post_grasp_lift_final_world = lift_final_world
                ctrl.post_grasp_lift_start_time = time.time()
                ctrl.post_grasp_lift_target_world = lift_start_world.copy()
                ctrl.post_grasp_lift_active = True

                ik_hold_active = False
                ik_hold_ready_count = 0
                ik_hold_target_world = None
                print("[TASK1] Grasp condition met -> close grip and lift")
            elif (task1_step == 1) and (task1_close_time is not None):
              if (time.time() - task1_close_time) >= task1_transfer_delay_s:
                basket_world = mj_data.xpos[task1_blue_basket_body_id].copy()
                basket_world[1] -= 0.05
                z_keep = float(mj_data.site_xpos[ctrl.right_palm_site_id][2]) + 0.06
                task1_transfer_target_world = np.array(
                  [basket_world[0], basket_world[1], z_keep], dtype=np.float32
                )
                ctrl.post_grasp_lift_active = False
                task1_step = 2
                ik_hold_active = False
                ik_hold_ready_count = 0
                ik_hold_target_world = None
                print("[TASK1] Transfer phase -> moving to blue basket")
            elif task1_step == 2:
              if pos_err_norm <= task1_pos_thresh:
                ctrl.set_grip_state(False)
                task1_step = 3
                task1_drop_open_time = time.time()
                print("[TASK1] At basket -> open grip")
            elif (task1_step == 3) and (task1_drop_open_time is not None):
              if (time.time() - task1_drop_open_time) >= task1_retarget_cube_delay_s:
                current_grasp_rot_world = roll_local_x.astype(np.float32)
                task1_step = 4
                print("[TASK1] Retarget -> moving to blue cube")
            elif task1_step == 4:
              if pos_err_norm <= task1_cube_grasp_pos_thresh:
                ctrl._cache_finger_actuators("blue_cube")
                ctrl.set_grip_state(True)
                task1_close_time = time.time()
                task1_step = 5

                ik_hold_active = False
                ik_hold_ready_count = 0
                ik_hold_target_world = None
                print("[TASK1] At blue cube -> switched grip profile, close and settle grip")
            elif (task1_step == 5) and (task1_close_time is not None):
              if (time.time() - task1_close_time) >= task1_cube_grip_settle_delay_s:
                lift_start_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                lift_final_world = lift_start_world.copy()
                lift_final_world[2] += post_grasp_lift_z_world
                ctrl.post_grasp_lift_start_world = lift_start_world
                ctrl.post_grasp_lift_final_world = lift_final_world
                ctrl.post_grasp_lift_start_time = time.time()
                ctrl.post_grasp_lift_target_world = lift_start_world.copy()
                ctrl.post_grasp_lift_active = True
                task1_step = 6
                print("[TASK1] Firm grip wait done -> lifting cube vertically")
            elif (task1_step == 6) and (ctrl.post_grasp_lift_start_time is not None):
              lift_elapsed = time.time() - ctrl.post_grasp_lift_start_time
              lift_done_by_time = lift_elapsed >= post_grasp_lift_duration_s
              lift_z_goal = None
              if ctrl.post_grasp_lift_final_world is not None:
                lift_z_goal = float(ctrl.post_grasp_lift_final_world[2]) + 0.08
              current_z = float(mj_data.site_xpos[ctrl.right_palm_site_id][2])
              lift_done_by_height = (
                (lift_z_goal is not None)
                and (abs(current_z - lift_z_goal) <= task1_cube_lift_z_tolerance)
              )
              lift_timeout = lift_elapsed >= (post_grasp_lift_duration_s + task1_cube_lift_max_wait_s)

              if ((lift_elapsed >= post_grasp_lift_duration_s and lift_done_by_height) or 
              (lift_elapsed >= (post_grasp_lift_duration_s + task1_cube_lift_max_wait_s))):
                
                basket_world = mj_data.xpos[task1_red_basket_body_id].copy()
                basket_world[0] -= 0.06 # adjustment for cube drop off
                if lift_z_goal is not None:
                  z_keep = lift_z_goal
                else:
                  z_keep = current_z
                task1_second_transfer_target_world = np.array(
                  [basket_world[0], basket_world[1], z_keep], dtype=np.float32
                )
                ctrl.post_grasp_lift_active = False
                task1_step = 7
                ik_hold_active = False
                ik_hold_ready_count = 0
                ik_hold_target_world = None
                print("[TASK1] Transfer phase -> moving to red basket")
            elif task1_step == 7:
              if pos_err_norm <= task1_pos_thresh2:
                ctrl.set_grip_state(False)
                task1_step = 8
                task1_drop_open_time = time.time()
                print("[TASK1] At red basket -> open grip")
            elif (task1_step == 8) and (task1_drop_open_time is not None):
              if (time.time() - task1_drop_open_time) >= task1_retarget_cube_delay_s:
                lift_start_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                lift_final_world = lift_start_world.copy()
                lift_final_world[2] += post_grasp_lift_z_world
                ctrl.post_grasp_lift_start_world = lift_start_world
                ctrl.post_grasp_lift_final_world = lift_final_world
                ctrl.post_grasp_lift_start_time = time.time()
                ctrl.post_grasp_lift_target_world = lift_start_world.copy()
                ctrl.post_grasp_lift_active = True
                task1_step = -1
                print("[TASK1] End")

          elif is_scene2_task:
            if task2_step == 0:
              if (pos_err_norm <= task2_pos_thresh) and (theta_hold <= task2_rot_thresh):
                ctrl._cache_finger_actuators("mug_object")
                ctrl.set_grip_state(True)
                task2_close_time = time.time()
                task2_hold_target_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                task2_pre_grasp_palm_z = float(task2_hold_target_world[2])

                # more graceful grip
                task2_hold_target_world[1] += 0.015
                task2_step = 1
                print("[TASK2] Mug grasp condition met -> close grip and stabilize")
            elif (
              (task2_step == 1)
              and (task2_close_time is not None)
              and (task2_hold_target_world is not None)
              and ((time.time() - task2_close_time) >= task2_grip_settle_delay_s)
            ):
                lift_start_world = task2_hold_target_world.copy()
                lift_final_world = lift_start_world.copy()
                lift_final_world[2] += post_grasp_lift_z_world
                ctrl.post_grasp_lift_start_world = lift_start_world
                ctrl.post_grasp_lift_final_world = lift_final_world
                ctrl.post_grasp_lift_start_time = time.time()
                ctrl.post_grasp_lift_target_world = lift_start_world.copy()
                ctrl.post_grasp_lift_active = True
                task2_step = 2
                print("[TASK2] Grip stabilized -> lifting vertically")
            elif (task2_step == 2) and (ctrl.post_grasp_lift_start_time is not None):
              lift_elapsed = time.time() - ctrl.post_grasp_lift_start_time
              if lift_elapsed >= post_grasp_lift_duration_s:
                if ctrl.post_grasp_lift_final_world is not None:
                  task2_hold_target_world = ctrl.post_grasp_lift_final_world.copy()
                ctrl.post_grasp_lift_active = False
                task2_step = 3
                task2_lift_done_time = time.time()
                ik_hold_active = False
                ik_hold_ready_count = 0
                ik_hold_target_world = None
                print("[TASK2] Lift done -> holding elevated pose")
            elif (
              (task2_step == 3)
              and (task2_lift_done_time is not None)
              and ((time.time() - task2_lift_done_time) >= task2_post_lift_wait_s)
            ):
                current_grasp_rot_world = yaw_local_z_after_transport.astype(np.float32)
                coaster_world = mj_data.xpos[task2_coaster_body_id].copy()
                coaster_world[0] -= 0.04
                coaster_world[1] -= 0.085
                if task2_hold_target_world is not None:
                  z_keep = float(task2_hold_target_world[2])
                else:
                  z_keep = float(mj_data.site_xpos[ctrl.right_palm_site_id][2])
                task2_transfer_target_world = np.array(
                  [coaster_world[0], coaster_world[1], z_keep], dtype=np.float32
                )
                task2_step = 4
                ik_hold_active = False
                ik_hold_ready_count = 0
                ik_hold_target_world = None
                print("[TASK2] Post-lift wait done -> reorienting and moving to coaster XY at fixed Z")
            elif task2_step == 4:
              if pos_err_norm <= task2_descend_pos_thresh:
                coaster_world = mj_data.xpos[task2_coaster_body_id].copy()
                coaster_world[0] -= 0.04
                coaster_world[1] -= 0.085
                if task2_pre_grasp_palm_z is not None:
                  release_z = task2_pre_grasp_palm_z
                elif task2_hold_target_world is not None:
                  release_z = float(task2_hold_target_world[2])
                else:
                  release_z = float(mj_data.site_xpos[ctrl.right_palm_site_id][2])
                task2_release_target_world = np.array(
                  [coaster_world[0], coaster_world[1], release_z - 0.01], dtype=np.float32
                )
                task2_step = 5
                ik_hold_active = False
                ik_hold_ready_count = 0
                ik_hold_target_world = None
                print("[TASK2] At coaster XY -> descending to release pose")
            elif task2_step == 5:
              if pos_err_norm <= task2_release_pos_thresh:
                mug_release_open_targets = {
                  "right_hand_thumb_0_joint": 0.0,   # curl thumb inward
                  "right_hand_thumb_1_joint": -0.3,  # flex thumb
                  "right_hand_thumb_2_joint": -0.9,  # curl thumb tip
                  "right_hand_index_0_joint": 0.7,   # curl index
                  "right_hand_index_1_joint": 0.7,   # curl index tip
                  "right_hand_middle_0_joint": 0.7,  # curl middle
                  "right_hand_middle_1_joint": 0.7,  # curl middle tip
                }
                ctrl._cache_finger_actuators("mug_object", open_targets=mug_release_open_targets)
                ctrl.grip_transition_duration_s = 2.5 # slow partial opening to avoid mug tipping (normally 1.0s)
                task2_opening_target_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                task2_opening_target_world[0] -= 0.01
                task2_opening_target_world[1] -= 0.013
                ctrl.set_grip_state(False)
                task2_lift_done_time = time.time()
                task2_step = 6
                print("[TASK2] Release pose reached -> slow partial opening grip")
            elif task2_step == 6:
              if ((pos_err_norm <= task2_distancing_pos_thresh)
                  and ((time.time() - task2_lift_done_time) >= task2_post_descent_wait_s)):
                task2_distancing_target_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                task2_distancing_target_world[0] -= 0.1
                task2_distancing_target_world[1] -= 0.005

                task2_step = 7
                ik_hold_active = False
                ik_hold_ready_count = 0
                ik_hold_target_world = None
                print("[TASK2] End")

          if is_scene1_task:
            ik_hold_active = False
            ik_hold_ready_count = 0
            ik_hold_target_world = None
          elif is_scene2_task:
            ik_hold_active = False
            ik_hold_ready_count = 0
            ik_hold_target_world = None
          elif ctrl.post_grasp_lift_active:
            ik_hold_active = False
            ik_hold_ready_count = 0
            ik_hold_target_world = None
          else:
            hold_ready = (pos_err_norm <= ik_hold_distance) and (theta_hold <= ik_hold_rot_rad)
            if not ik_hold_active:
              if hold_ready:
                ik_hold_ready_count += 1
              else:
                ik_hold_ready_count = 0

              if ik_hold_ready_count >= ik_hold_min_cycles:
                ik_hold_active = True
                ik_hold_target_world = target_world.copy()
                print(
                  f"[IK] HOLD latched at |pos_err|={pos_err_norm:.4f} m "
                  f"|rot_err|={np.rad2deg(theta_hold):.2f} deg"
                )

          jacp, jacr = ctrl._get_palm_jacobian_in_pelvis()
          jacobian_6x7 = np.vstack((jacr, jacp)).astype(np.float32)
          x_dot = ctrl.compute_ee_cartesian_velocity(
            ik_goal_pos_pelvis,
            ik_goal_rot_pelvis,
            k_l=ik_pos_gain,
            k_a=ik_rot_gain,
          )

          dq = _solve_damped_pseudoinverse_dq(jacobian_6x7, x_dot, ik_damping)
          dq = np.clip(dq, -ik_max_joint_speed, ik_max_joint_speed)
          right_arm_q_des = right_arm_q_des + dq * ik_dt
          right_arm_q_des = np.clip(right_arm_q_des, right_arm_q_min, right_arm_q_max)

          if control_step % 200 == 0:
            if ik_hold_active:
              print(
                f"[IK] HOLD active |pos_err|={pos_err_norm:.4f} "
                f"|rot_err|={np.rad2deg(theta_hold):.2f} deg"
              )
            else:
              print(
                f"[IK] |pos_err|={pos_err_norm:.4f} "
                f"|rot_err|={np.rad2deg(theta_hold):.2f} deg "
                f"|x_dot|={float(np.linalg.norm(x_dot)):.4f} "
                f"|dq|={float(np.linalg.norm(dq)):.4f}"
              )
        elif control_step % 200 == 0:
          print("[IK] Waiting startup settle before enabling IK...")

        # Inject IK targets into target_pos
        for i, full_idx in enumerate(ctrl.right_arm_indices):
          target_pos[full_idx] = float(right_arm_q_des[i])
      ctrl.apply_pd_control(target_pos)
      mujoco.mj_step(mj_model, mj_data)

      locker.release()

      control_step += 1
      sim_time += mj_model.opt.timestep


def PhysicsViewerThread():
    right_shoulder_body_id = mujoco.mj_name2id(
      mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link"
    )
    configured_camera = False
    while viewer.is_running():

        locker.acquire()
        if not configured_camera:
          # Set initial camera aiming at right arm/hand, then keep it fixed.
          palm_world = mj_data.site_xpos[ctrl.right_palm_site_id]
          if right_shoulder_body_id >= 0:
            shoulder_world = mj_data.xpos[right_shoulder_body_id]
            shoulder_world[0] = shoulder_world[0] + 0.2 # offset along x axis world
            cam_focus = 0.65 * palm_world + 0.35 * shoulder_world
          else:
            cam_focus = palm_world
          viewer.cam.lookat[:] = cam_focus
          viewer.cam.distance = 1.7
          viewer.cam.azimuth = 150.0
          viewer.cam.elevation = -35.0
          configured_camera = True
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
