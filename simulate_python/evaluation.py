import json
import os
import sys
import time
import threading
import numpy as np
from threading import Thread
from pathlib import Path
from datetime import datetime

import mujoco
from mujoco import viewer

from script_files.ONNXPolicy import ONNXPolicy
from script_files.G1Controller import G1Controller
import script_files.utilities as utilities 
import evaluation_config

# Providing fallback for SCRIPT_PATH and SCRIPT_DIR in case __file__ is not unavailable.
# This way both instructions work.
try:
  SCRIPT_PATH = Path(__file__).resolve()
except NameError:
  SCRIPT_PATH = Path.cwd() / "evaluation.py"
SCRIPT_DIR = SCRIPT_PATH.parent

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

locker = threading.Lock() # MuJoCo viewer and simulation run in separate threads,
                          # need to ensure consistency of shared variables (e.g. mjData)

# Read current evaluation instruction from the last non-empty log line.
EVAL_LOG_PATH = SCRIPT_DIR / "quantitative_evaluation/evaluations.log"
with open(EVAL_LOG_PATH, "r", encoding="utf-8") as f:
  log_lines = [line.strip() for line in f if line.strip()]
if not log_lines:
  raise RuntimeError(f"No evaluation instructions found in {EVAL_LOG_PATH}")
current_instruction = log_lines[-1]
current_instruction = current_instruction.split(" ")[-1]

# Selects scene to load based on instructions from evaluations.log (infers benchmark selection).
selected_scene = evaluation_config.get_robot_scene(current_instruction)
print(f"[CONFIG] Using scene: {selected_scene}")
mj_model = mujoco.MjModel.from_xml_path(selected_scene)

# Load robot configuation for walker policy(specific order to coordinate with ONNX model) and armature setup.
config_path = SCRIPT_DIR / "policy_resources/model_config.json"
with open(config_path) as f:
    config_robot = json.load(f)
joint_names = config_robot["joint_names"]

mj_model.opt.timestep = evaluation_config.SIMULATE_DT
utilities.set_armature(mj_model, joint_names)

mj_data = mujoco.MjData(mj_model)

# Init robot pose -> spawn behind the table, facing it
# (pelvis position and orientation)
mj_data.qpos[0] = -0.6  # x: back from table
mj_data.qpos[2] = 0.76  # z: height
mj_data.qpos[3:7] = [1, 0, 0, 0]  # quaternion for orientation
for name, value in config_robot["default_joint_pos"].items():
    if name in joint_names:
        mj_data.qpos[7 + joint_names.index(name)] = value

# Startup right-arm posture preload: elbow flexed, forearm and hand lifted
# (avoids poor right arm control at simulation start, and provides a more natural starting pose for the right arm).
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

# Snapshot full initial simulation state to restore before each new evaluation trial.
initial_sim_state = {
  "qpos": mj_data.qpos.copy(),
  "qvel": mj_data.qvel.copy(),
  "act": mj_data.act.copy(),
  "ctrl": mj_data.ctrl.copy(),
  "qacc_warmstart": mj_data.qacc_warmstart.copy(),
  "mocap_pos": mj_data.mocap_pos.copy(),
  "mocap_quat": mj_data.mocap_quat.copy(),
  "time": float(mj_data.time),
}

# Load walker policy
print("Loading ONNX policies...")
walker = ONNXPolicy(str(SCRIPT_DIR / "policy_resources/walker.onnx"))

# ------------ Creating right arm controller ------------ #

# Predefined grasp poses for the two benchmark objects, defined as local rotations 
# to be applied to the right palm site. 
# (all orientations are to be considered as applied to the right hand with its palm facing the robot body)

# Desired palm orientation in world frame for top-down grasping (Blue cube).
# A local roll (if negative -> inward rotation of the palm) is applied to the wrist joint.
wrist_roll_deg = -90.0
roll = np.deg2rad(wrist_roll_deg)
cr1, sr1 = np.cos(roll), np.sin(roll)

roll_local_x = np.array([
  [1.0, 0.0, 0.0],
  [0.0, cr1, -sr1],
  [0.0, sr1, cr1],
], dtype=np.float32)

# Desired palm orientation in world frame for tilted grasping (White mug).
# A local yaw (negative -> flex of the wrist to the right) is applied to the wrist joint.
wrist_yaw_deg = 20.0
yaw = np.deg2rad(wrist_yaw_deg)
cr2, sr2 = np.cos(yaw), np.sin(yaw)

yaw_local_z = np.array([
  [cr2, -sr2, 0.0],
  [sr2, cr2,  0.0],
  [0.0, 0.0,  1.0],
], dtype=np.float32)

# Palm orientation after transportation to ease object relocation (White mug).
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

# Predefined palm position offsets for grasping different objects (only those that are grabbed as first item).
target_body_name = None
for candidate in ("red_cylinder", "mug_object"):
  # Retrieving first object to grasp depending on task chosen.
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
      ik_target_x_offset_world = 0.01 # offset along x direction mug center
      ik_target_y_offset_world = -0.14 # no offset along y direction mug center
      ik_target_z_offset_world = 0.0  # no offset above mug center
      grasp_rot_world = (yaw_local_z).astype(np.float32)
    break
else:
  raise RuntimeError("Body not found in model: red_cylinder/mug_object")
print(f"[IK] Tracking target body: {target_body_name}")

ctrl = G1Controller(mj_model, mj_data, walker, config_robot, target_body_name)
initial_grasp_rot_world = grasp_rot_world.copy()

# Setting up triggering task flags based on the selected benchmark.
is_scene1_task = (target_body_name == "red_cylinder")
is_scene2_task = (target_body_name == "mug_object")

# Warm up ONNX models (first call triggers JIT compilation)
_dummy99 = np.zeros((1, 99), dtype=np.float32)
walker(_dummy99)

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

# -------------------------------------------------------------------- #
#          Evaluation parameters (resetted when changing task)
# -------------------------------------------------------------------- #

# (defined as global variables to be accessible from both simulation and measurement threads)

# Lists of elements with 8 values each (left hip, left knee, left ankle, right hip, right knee, right ankle,
#  waist, torso) to store maximum roll and pitch measured during each trial (used for evaluation type 1 and 2)
# (computed in a separate thread for highest frequency of measurements)
n_trials_per_task = 20 # Number of trials to execute for each task and evaluation type
max_roll_measured = [np.zeros(8, dtype=np.float32) for _ in range(n_trials_per_task)]
max_pitch_measured = [np.zeros(8, dtype=np.float32) for _ in range(n_trials_per_task)]
trial_index = 0 # Shared current trial slot/counter for measurement and simulation threads.
all_trials_done = threading.Event() # Signaled by SimulationThread when all trials finish; PhysicsViewerThread handles viewer.close() on its own thread to avoid GLFW thread-safety violations.


# Task logic runs separate from the physics and rendering loop in order to avoid impacting simulation performance with Python code execution time.
def SimulationThread():

  # -------------------------------------------------------------------- #
  #                   Evaluation parameters (continued)
  # -------------------------------------------------------------------- #

  # (defined as local variables since they need to be accessed only by this thread)
  # (variables for task 1 and 2 are not separated, since the executions of this script to evaluate
  #  the two tasks are done in two separate runs, and the variables are reset at the beginning of each run)

  evaluation_type = 1 # 1: simple repetition of the task, 2: randomisation of object initial positions
  position_error_eval = [] # List to store position errors of mug placement step for each trial
  # (since only step where target position must be precisely respected)(used for evaluation type 1 and 2 with randomised initial positions)
  task_completion_times = [] # List to store task completion times for each trial (used for evaluation type 1 and 2)
  task_start_time = None # Variable to store the start time of the current trial (used to compute task completion time)
  task_success = [] # List to store integer values indicating successful steps for each trial of task 1/2 (used for evaluation type 1 and 2)
  randomisation_range = 0.015 # Range of randomisation for object initial positions in evaluation type 2 (applied on x and y axes)

  # Variable shared with other thread declared as global to be accessible and modifiable from both threads.
  # This variable is used to keep track of the current trial index for storing measurements and results,
  #  as well as counting the trials executed.
  global trial_index, max_roll_measured, max_pitch_measured

  # Defining initial right-arm joint target positions.
  right_arm_joint_refs = utilities._joint_state_refs(mj_model, ctrl.right_arm_joint_names)
  right_arm_q_des = np.array([
    float(mj_data.qpos[right_arm_joint_refs[name][0]])
    for name in ctrl.right_arm_joint_names
  ], dtype=np.float32)

  # Defining right-arm joint limits for IK clamping.
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
  # this way joint positions are kept until initial movement.
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

  # IK parameters and state variables (can be changed more easily than hardcoded values in control loop)
  ik_enabled = True
  ik_damping = 0.08 # for pseudoinverse computation
  ik_pos_gain = 2.0 # gain applied on position error to compute desired task velocity
  ik_rot_gain = 2.0 # gain applied on orientation error to compute desired task velocity
  ik_max_joint_speed = 1.8  # rad/s
  ik_startup_delay_s = 0.25  # keep startup posture briefly, then start IK pursuit
  post_grasp_lift_z_world = 0.07 # post-grasp z offset for lifting phase
  post_grasp_lift_duration_s = 1.2 # wait after grasping before lifting the object (to ensure firm grip)
  current_grasp_rot_world = grasp_rot_world.copy() # current desired palm orientation for grasping (can be changed during task execution)

  # Scene1 task state machine #
  # Phases: 1. grasp red cylinder 
  #         2. transfer and release it in blue basket 
  #         3. grasp from cube 4.  
  #         4. transfer and release it in blue basket
  task1_step = 0
  task1_cylind_grasp_pos_thresh = 0.005 # position error threshold to get in position to grasp the cylinder (used to enter step 1)
  task1_cylind_drop_pos_thresh = 0.005 # position error threshold to align cylinder with the blue basket (used to enter step 2)
  task1_cube_drop_pos_thresh = 0.01 # position error threshold to align cube with the red basket (used to enter step 7)
  task1_cube_grasp_pos_thresh = 0.0125 # position error threshold to get in position to grasp the cube (used to enter step 4)
  task1_cylind_rot_thresh = 0.08 # orientation error threshold to get in position to grasp the cylinder (used to enter step 0)
  task1_cylind_transfer_delay_s = 2.0 # wait after grasping the cylinder before moving to the blue basket (used to enter step 1)
  task1_retarget_cube_delay_s = 1.0 # wait after releasing the cylinder before retargeting the cube (used to enter step 3)
  task1_cube_grip_settle_delay_s = 2.0 # wait after grasping the cube before lifting it to secure firm grip(used to enter step 5)
  task1_cube_lift_z_tolerance = 0.008 # z position tolerance to consider the cube lifted (used to enter step 5)
  task1_cube_lift_max_wait_s = 2.0 # maximum wait time to consider the cube lifted (used to enter step 5)
  task1_close_time = None # Used as instant in time as reference to know when the grip is closed (used to enforce grasp delays in task progression)
  task1_drop_open_time = None # Used as instant in time as reference to know when the grip is opened (used to enforce grasp delays in task progression)
  task1_transfer_target_world1 = None # Used to store target palm position for the cylinder transfer to blue basket (used to enter step 1)
  task1_transfer_target_world2 = None # Used to store target palm position for the cube transfer to red basket (used to enter step 6)
  task1_blue_basket_body_id = -1 # Auxiliary variable to store body ID of the blue basket (used to enter step 1)
  task1_red_basket_body_id = -1 # Auxiliary variable to store body ID of the red basket (used to enter step 6)
  task1_blue_cube_body_id = -1 # Auxiliary variable to store body ID of the blue cube (used to enter step 4)
  # Fixed offsets from the cube center to apply an efficient grasp. (determined in a separate simulation)
  # (for the cylinder and the mug the information are provided during initialisation of the IK controller, 
  # as they are the first objects to grasp in each task)
  task1_cube_x_offset = 0.019
  task1_cube_y_offset = 0.0
  task1_cube_z_offset = 0.08
  if is_scene1_task:
    # Allocating object IDs for later use in the task logic.
    task1_blue_basket_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "blue_basket")
    if task1_blue_basket_body_id < 0:
      raise RuntimeError("Body not found in model: blue_basket")
    task1_red_basket_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "red_basket")
    if task1_red_basket_body_id < 0:
      raise RuntimeError("Body not found in model: red_basket")
    task1_blue_cube_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube")
    if task1_blue_cube_body_id < 0:
      raise RuntimeError("Body not found in model: blue_cube")

  # Scene1 task state machine #
  # Phases: 1. grasp mug handle 
  #         2. lift mug
  #         3. transfer mug on coaster
  #         4. release mug on coaster
  task2_step = 0
  task2_pos_thresh = 0.01 # position error threshold to reach mug handle (used to enter step 0)
  task2_rot_thresh = np.deg2rad(0.2) # orientation error threshold to reach mug handle (used to enter step 0)
  task2_grip_settle_delay_s = 2.0 # wait after grasping the mug before lifting it to secure firm grip (used to enter step 1)
  task2_post_lift_wait_s = 2.0 # wait after lifting the mug before proceeding (used to enter step 2)
  task2_post_descent_wait_s = 3.0 # wait after descending the mug before releasing (used to enter step 3)
  task2_release_pos_thresh = 0.025 # position error threshold to release the mug (used to enter step 4)
  task2_descend_pos_thresh = 0.01 # position error threshold to descend the mug (used to enter step 5)
  task2_distancing_pos_thresh = 0.02 # position error threshold to distance the mug (used to enter step 6)
  task2_step0_max_wait_s = 4.0 # fallback timeout to avoid getting stuck before initial mug grasp in evaluation type 2
  task2_step0_fallback_pos_thresh = 0.03 # relaxed position threshold used by step-0 timeout fallback
  task2_step5_max_wait_s = 2.5 # fallback timeout to avoid getting stuck in step 5 when threshold is narrowly missed
  task2_close_time = None # Used as instant in time as reference to know when the grip is closed (used to enforce grasp delays in task progression)
  task2_lift_done_time = None # Used as instant in time as reference to know when the lift is done (used to enforce grasp delays in task progression)
  task2_step0_entry_time = None # Used as time reference to enable timeout-based fallback in step 0
  task2_step5_entry_time = None # Used as time reference to enable timeout-based fallback in step 5
  task2_hold_target_world = None # Used to store palm target positions throught the steps
  task2_pre_grasp_palm_z = None # Used to store palm height for grasping and releasing mug handle
  task2_transfer_target_world = None # Used to store  target palm position to transfer the mug on the coaster
  task2_release_target_world = None # Used to store target palm position to release the mug on the coaster
  task2_opening_target_world = None # Used to store target palm position during mug release to avoid tipping the mug (delicate motion)
  task2_coaster_body_id = -1 # Auxiliary variable to store the body ID for the coaster (used in scene 2 task)
  if is_scene2_task:
    # Allocating object ID for later use in the task logic.
    task2_coaster_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "coaster")
    if task2_coaster_body_id < 0:
      raise RuntimeError("Body not found in model: coaster")

  task_ended = False # Flag to indicate when the task is completed (used to stop IK updates and print final message)

  # ----------------------------------------------------------- #
  #                 Function to reset simulation
  # ----------------------------------------------------------- #

  def reset_simulation():
    """Resets the simulation to the initial state and reinitializes task variables."""

    # Declare nonlocal variables to reset them using this nested function within the simulation loop at the end of each trial.
    nonlocal right_arm_q_des, current_grasp_rot_world
    nonlocal task1_step, task1_close_time, task1_drop_open_time
    nonlocal task1_transfer_target_world1, task1_transfer_target_world2
    nonlocal task2_step, task2_close_time, task2_lift_done_time
    nonlocal task2_step0_entry_time, task2_step5_entry_time
    nonlocal task2_hold_target_world, task2_pre_grasp_palm_z
    nonlocal task2_transfer_target_world, task2_release_target_world, task2_opening_target_world
    nonlocal task_ended, task_start_time

    right_arm_q_des, current_grasp_rot_world = utilities.reset(
      mj_model,mj_data,ctrl,right_arm_q_min,right_arm_q_max,
      initial_sim_state,initial_grasp_rot_world,evaluation_type,
      is_scene1_task,is_scene2_task,randomisation_range)
    task1_step = 0
    task1_close_time = None
    task1_drop_open_time = None
    task1_transfer_target_world1 = None
    task1_transfer_target_world2 = None

    task2_step = 0
    task2_close_time = None
    task2_lift_done_time = None
    task2_step0_entry_time = None
    task2_step5_entry_time = None
    task2_hold_target_world = None
    task2_pre_grasp_palm_z = None
    task2_transfer_target_world = None
    task2_release_target_world = None
    task2_opening_target_world = None
    task_ended = False
    task_start_time = None

  # ------------------------------------------------------------------- #
  # Simulation loop using launch_passive (MuJoCo's built-in viewer)
  # ------------------------------------------------------------------- #

  # Simulation parameters
  decimation = 4 # Number of physics steps per control step (controls the action update frequency)
  ik_dt = float(mj_model.opt.timestep * decimation) # Time step to use for desired joint positions computation from IK velocity (integration)
  control_step = 0 # Counter to keep track of control steps for decimation (physics/informative updates)
  target_pos = ctrl.default_joint_pos.copy() # Initial target joint positions (will be updated by the control loop)
  sim_time = 0.0 # Simulation time, updated in real time with the wall clock to allow time-based task logic and IK updates

  print("Launching MuJoCo viewer...")

  # Reset clock AFTER viewer opens — prevents catchup lag burst on startup
  t0 = time.time()

  # Time after which IK updates are allowed to start (allows to maintain the initial posture briefly
  # for better visualisation and to avoid poor right arm control at simulation start).
  ik_enable_time = t0 + ik_startup_delay_s
  while viewer.is_running():

    # Step physics in real time (cap catchup to avoid jitter snowball)
    wall = time.time() - t0
    max_catchup = 0.05  # Never try to catch up more than 50ms per frame

    if wall - sim_time > max_catchup:
      # Simulation is lagging behind real time, skipping frames to catch up (prevents long catchup bursts that cause jitter).
      sim_time = wall - max_catchup
    while sim_time < wall:

      locker.acquire()
      # --- critical section start (accessing shared variables) --- #

      if control_step % decimation == 0:
        target_pos = ctrl.step() # Updating target joint positions (walker policy + arm controller)
        ik_can_run = ik_enabled and (time.time() >= ik_enable_time)
        if ik_can_run:
          ee_pos = ctrl._get_palm_pos_in_pelvis().astype(np.float32)

          # Defining target palm position for IK depending on the current task and step to achieve 
          # the desired behavior in each phase of the tasks. If no specific target is defined for the current step,
          # the default behavior is to keep the current palm position.
          if is_scene1_task and (task1_step == 2) and (task1_transfer_target_world1 is not None):
            # Lifting cylinder after grasping and moving to the blue basket.
            target_world = task1_transfer_target_world1.copy()
          elif is_scene1_task and (task1_step in (4, 5)):
            # Target position for cube grasping. The goal is defined here instead of being applied 
            # in the corresponding steps (4, 5) to use the most up to date cube position at each cycle.
            # This ensures even tiny adjustments (e.g., due to the hand) are accounted for, providing a firmer grip.
            cube_world = mj_data.xpos[task1_blue_cube_body_id].copy()
            cube_world[0] -= task1_cube_x_offset
            cube_world[1] += task1_cube_y_offset
            cube_world[2] += task1_cube_z_offset
            target_world = cube_world
          elif is_scene1_task and (task1_step in (7, 8)) and (task1_transfer_target_world2 is not None):
            # Transferring cube to the red basket after grasping it.
            target_world = task1_transfer_target_world2.copy()
          elif is_scene2_task and (task2_step in (1, 3)) and (task2_hold_target_world is not None):
            # Holding the mug after grasping and before releasing it on the coaster.
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
            # Handling post-grasp lift by blending post_grasp_lift_start_world and post_grasp_lift_final_world to provide 
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
          else:
            # Target palm position for grasping the first object (cylinder or mug)
            #  (defined in the controller initialisation)
            target_world = mj_data.xpos[target_body_id].copy()
            target_world[0] -= ik_target_x_offset_world
            target_world[1] += ik_target_y_offset_world
            target_world[2] += ik_target_z_offset_world

          # Computing the desired palm position and orientation in the pelvis frame to be used for IK.
          base_pos_world, base_quat_world = ctrl._get_base_pose()
          ik_goal_pos_pelvis = ctrl._quat_apply_inverse(
            base_quat_world,
            target_world - base_pos_world,
          ).astype(np.float32)
          world_to_pelvis = ctrl._quat_to_rotmat(base_quat_world).T
          ik_goal_rot_pelvis = (world_to_pelvis @ current_grasp_rot_world).astype(np.float32)

          # Computing variable for position error check.
          pos_err = ik_goal_pos_pelvis - ee_pos
          pos_err_norm = float(np.linalg.norm(pos_err))

          # Computing variable for orientation error check.
          bRe = ctrl._get_palm_rot_in_pelvis_mat().astype(np.float64)
          R_err_hold = bRe.T @ ik_goal_rot_pelvis.astype(np.float64)
          _, theta_hold = ctrl._rot_to_angle_axis(R_err_hold)

          # ------ Scene1 task progression ------ #
          if is_scene1_task:
            if task1_step == 0:
              # Approaching cylinder to grasp it. Once the position and orientation thresholds are met,
              # the grip is closed and the lift phase is initiated.
              if (pos_err_norm <= task1_cylind_grasp_pos_thresh) and (theta_hold <= task1_cylind_rot_thresh):
                ctrl.set_grip_state(True)
                task1_close_time = time.time()
                lift_start_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                lift_final_world = lift_start_world.copy()
                lift_final_world[2] += post_grasp_lift_z_world
                ctrl.post_grasp_lift_start_world = lift_start_world
                ctrl.post_grasp_lift_final_world = lift_final_world
                ctrl.post_grasp_lift_start_time = time.time()
                ctrl.post_grasp_lift_target_world = lift_start_world.copy()
                ctrl.post_grasp_lift_active = True
                task1_step = 1
                task_start_time = time.time()  # Record the start time of the task
                print("[TASK1] Grasp condition met -> close grip and lift")

            elif (task1_step == 1) and (task1_close_time is not None):
              # Waiting for the grip to settle, then lifting the cylinder and moving to the blue basket.
              if (time.time() - task1_close_time) >= task1_cylind_transfer_delay_s:
                basket_world = mj_data.xpos[task1_blue_basket_body_id].copy()
                basket_world[1] -= 0.05
                z_keep = float(mj_data.site_xpos[ctrl.right_palm_site_id][2]) + 0.06
                task1_transfer_target_world1 = np.array(
                  [basket_world[0], basket_world[1], z_keep], dtype=np.float32
                )
                ctrl.post_grasp_lift_active = False
                task1_step = 2
                print("[TASK1] Transfer phase -> moving to blue basket")

            elif task1_step == 2:
              # Waiting for the cylinder to reach the blue basket, then opening the grip to release it.
              if pos_err_norm <= task1_cylind_drop_pos_thresh:
                ctrl.set_grip_state(False)
                task1_step = 3
                task1_drop_open_time = time.time()
                print("[TASK1] At basket -> open grip")

            elif (task1_step == 3) and (task1_drop_open_time is not None):
              # Waiting for the retarget delay before moving to the blue cube 
              # (slows down the transition preventing errors).
              if (time.time() - task1_drop_open_time) >= task1_retarget_cube_delay_s:
                current_grasp_rot_world = roll_local_x.astype(np.float32)
                task1_step = 4
                print("[TASK1] Retarget -> moving to blue cube")

            elif task1_step == 4:
              # Approaching blue cube to grasp it. Once the position and orientation thresholds are met,
              #  the grip is closed and the lift phase is initiated.
              if pos_err_norm <= task1_cube_grasp_pos_thresh:
                ctrl._cache_finger_actuators("blue_cube")
                # Re-prime grip interpolation state so cube closing always starts from a known open baseline.
                ctrl.grip_transition_duration_s = 1.0
                ctrl.grip_closed = False
                ctrl.grip_close_time = None
                ctrl.grip_transition_start_time = None
                ctrl.grip_alpha_start = 0.0
                ctrl.grip_alpha_goal = 0.0
                ctrl.set_grip_state(True)
                task1_close_time = time.time()
                task1_step = 5
                print("[TASK1] At blue cube -> switched grip profile, close and settle grip")

            elif (task1_step == 5) and (task1_close_time is not None):
              # Waiting for the grip to settle, then lifting the cube.
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
              # Waiting for the cube to be lifted, then moving to the red basket.
              lift_elapsed = time.time() - ctrl.post_grasp_lift_start_time
              lift_z_goal = None
              if ctrl.post_grasp_lift_final_world is not None:
                # Defining the target z position to check if cube has been lifted enough,
                # and to use it as reference for the red basket transfer target (keeping the same height during transfer).
                lift_z_goal = float(ctrl.post_grasp_lift_final_world[2]) + 0.08

              current_z = float(mj_data.site_xpos[ctrl.right_palm_site_id][2])

              # Triple check for lifting completion: by height, by time, and by timeout 
              # (to avoid getting stuck in case of unexpected issues).
              lift_done_by_height = ((lift_z_goal is not None) and 
                                    (abs(current_z - lift_z_goal) <= task1_cube_lift_z_tolerance))
              lift_done_by_time = lift_elapsed >= post_grasp_lift_duration_s
              lift_timeout = lift_elapsed >= (post_grasp_lift_duration_s + task1_cube_lift_max_wait_s)

              if ((lift_done_by_time and lift_done_by_height) or lift_timeout):
                # If the cube has been lifted enough, we can move to the red basket.
                basket_world = mj_data.xpos[task1_red_basket_body_id].copy()
                basket_world[0] -= 0.06 # adjustment for cube drop off
                if lift_z_goal is not None:
                  z_keep = lift_z_goal
                else:
                  z_keep = current_z
                task1_transfer_target_world2 = np.array(
                  [basket_world[0], basket_world[1], z_keep], dtype=np.float32
                )
                ctrl.post_grasp_lift_active = False
                task1_step = 7
                print("[TASK1] Transfer phase -> moving to red basket")

            elif task1_step == 7:
              # Waiting for the cube to reach the red basket, then opening the grip to release it.
              if pos_err_norm <= task1_cube_drop_pos_thresh:
                ctrl.set_grip_state(False)
                task1_step = 8
                task1_drop_open_time = time.time()
                print("[TASK1] At red basket -> open grip")

            elif (task1_step == 8) and (task1_drop_open_time is not None):
              # Final movmenent after releasing the cube to avoid collisions with the basket. 
              # After this delay, the task is considered completed.
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
                task_ended = True
                print("[TASK1] End")

          # ------ Scene2 task progression ------ #
          elif is_scene2_task:
            if task2_step == 0:
              # Approaching mug handle to grasp it. Once the position and orientation thresholds are met,
              # the grip is closed. Setting a timeout to avoid getting stuck in case of unexpected issues
              # due to mug position randomisation (evaluation type 2) (e.g., mug handle is not reachable).
              if task2_step0_entry_time is None:
                task2_step0_entry_time = time.time()

              step0_timeout = (
                (evaluation_type == 2)
                and ((time.time() - task2_step0_entry_time) >= task2_step0_max_wait_s)
              )

              can_grasp_normally = ((pos_err_norm <= task2_pos_thresh) and (theta_hold <= task2_rot_thresh))
              can_grasp_with_timeout = (step0_timeout and (pos_err_norm <= task2_step0_fallback_pos_thresh))

              if can_grasp_normally or can_grasp_with_timeout:
                ctrl._cache_finger_actuators("mug_object")
                ctrl.set_grip_state(True)
                task2_close_time = time.time()
                task2_hold_target_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                task2_hold_target_world[1] += 0.015 # better alignment for grasping the mug handle
                task2_pre_grasp_palm_z = float(task2_hold_target_world[2])
                task2_step = 1
                task_start_time = time.time()  # Record the start time of the task
                if can_grasp_with_timeout:
                  print(f"[TASK2][SAFE] Step 0 timeout ({task2_step0_max_wait_s}s) -> forcing grasp transition (pos_err={pos_err_norm:.4f}, rot_err={theta_hold:.4f})")
                print("[TASK2] Mug grasp condition met -> close grip and stabilize")

            elif (
              (task2_step == 1)
              and (task2_close_time is not None)
              and (task2_hold_target_world is not None)
              and ((time.time() - task2_close_time) >= task2_grip_settle_delay_s)
            ):
                # After the grip has settled, we can lift the mug vertically to avoid collisions with the table.
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
              # Waiting for the mug to be lifted, then holding it in an elevated pose before moving to the coaster.
              lift_elapsed = time.time() - ctrl.post_grasp_lift_start_time
              if lift_elapsed >= post_grasp_lift_duration_s:
                if ctrl.post_grasp_lift_final_world is not None:
                  task2_hold_target_world = ctrl.post_grasp_lift_final_world.copy()
                ctrl.post_grasp_lift_active = False
                task2_step = 3
                task2_lift_done_time = time.time()
                print("[TASK2] Lift done -> holding elevated pose")

            elif (
              (task2_step == 3)
              and (task2_lift_done_time is not None)
              and ((time.time() - task2_lift_done_time) >= task2_post_lift_wait_s)
            ):
                # After the post-lift wait, we can reorient the palm and move to the coaster XY position 
                # maintaining the same Z height after lift.
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
                print("[TASK2] Post-lift wait done -> reorienting and moving to coaster XY at fixed Z")

            elif task2_step == 4:
              # Waiting for the mug to reach the coaster XY position, then descending to the release pose.
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
                task2_step5_entry_time = time.time()
                print("[TASK2] At coaster XY -> descending to release pose")

            elif task2_step == 5:
              # Waiting for the mug to reach the release pose, then slowly opening the grip to avoid
              #  tipping the mug. Considering a timeout to avoid getting stuck in this step 
              #  during evaluation if the position threshold is missed (for evaluation type 2).
              step5_timeout = (
                (task2_step5_entry_time is not None)
                and ((time.time() - task2_step5_entry_time) >= task2_step5_max_wait_s)
              )
              if (pos_err_norm <= task2_release_pos_thresh) or step5_timeout:
                mug_release_open_targets = {
                  "right_hand_thumb_0_joint": 0.0,   # curl thumb inward
                  "right_hand_thumb_1_joint": -0.3,  # flex thumb
                  "right_hand_thumb_2_joint": -0.9,  # curl thumb tip
                  "right_hand_index_0_joint": 0.7,   # curl index
                  "right_hand_index_1_joint": 0.7,   # curl index tip
                  "right_hand_middle_0_joint": 0.7,  # curl middle
                  "right_hand_middle_1_joint": 0.7,  # curl middle tip
                }

                # Applying a slow partial opening of the grip to avoid tipping the mug.
                ctrl._cache_finger_actuators("mug_object", open_targets=mug_release_open_targets)
                ctrl.grip_transition_duration_s = 2.5 # slow partial opening to avoid mug tipping (normally 1.0s)
                task2_opening_target_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                task2_opening_target_world[0] -= 0.01
                task2_opening_target_world[1] -= 0.013
                ctrl.set_grip_state(False)
                task2_lift_done_time = time.time()
                task2_step = 6
                if step5_timeout:
                  print(f"[TASK2][SAFE] Step 5 timeout ({task2_step5_max_wait_s}s) -> forcing release transition (pos_err={pos_err_norm:.4f})")
                print("[TASK2] Release pose reached -> slow partial opening grip")

            elif task2_step == 6:
              # Waiting for the mug to be released on the coaster, then moving to a safe position.
              # After this check is cleared, the task is considered completed.
              if ((pos_err_norm <= task2_distancing_pos_thresh)
                  and ((time.time() - task2_lift_done_time) >= task2_post_descent_wait_s)):
                task2_distancing_target_world = mj_data.site_xpos[ctrl.right_palm_site_id].copy()
                task2_distancing_target_world[0] -= 0.1
                task2_distancing_target_world[1] -= 0.005
                
                # Computing distance between mug and coaster for evaluation purposes (error).
                coaster_pos = mj_data.xpos[task2_coaster_body_id].copy()
                mug_pos = mj_data.xpos[target_body_id].copy()
                mug_coaster_distance = np.linalg.norm(mug_pos - coaster_pos)
                position_error_eval.append(mug_coaster_distance)
                task2_step = 7
                task_ended = True
                print("[TASK2] End")

          # ------ Inverse Kinematics (IK) update ------ #
          # Retrieving the palm Jacobian in the pelvis frame to compute the desired joint velocities
          #  from the desired end-effector velocity.
          jacp, jacr = ctrl._get_palm_jacobian_in_pelvis()
          jacobian_6x7 = np.vstack((jacr, jacp)).astype(np.float32)
          x_dot = ctrl.compute_ee_cartesian_velocity(
            ik_goal_pos_pelvis,
            ik_goal_rot_pelvis,
            k_l=ik_pos_gain,
            k_a=ik_rot_gain,
          )

          # Computing the desired joint velocities using the damped pseudoinverse of the Jacobian.
          dq = utilities._solve_damped_pseudoinverse_dq(jacobian_6x7, x_dot, ik_damping)
          dq = np.clip(dq, -ik_max_joint_speed, ik_max_joint_speed)

          # Integrating the desired joint velocities to compute the desired joint positions.
          right_arm_q_des = right_arm_q_des + dq * ik_dt
          right_arm_q_des = np.clip(right_arm_q_des, right_arm_q_min, right_arm_q_max)

          if task_ended:

            # Compute task completion time for evaluation purposes.
            task_end_time = time.time()
            if task_start_time is not None:
              task_completion_times = utilities._compute_task_completion_time(
                task_start_time,
                task_end_time,
                task_completion_times
              )
            
            # Compare object positions with basket positions to determine the degree of success of the task.
            # (in task 1 two objects must reach their respective baskets, while in task 2 only one object
            #  must reach its target position)
            task_success = utilities._is_task_successful(
              is_scene1_task, is_scene2_task,
              mj_data, task1_blue_cube_body_id,
              task1_red_basket_body_id, task1_blue_basket_body_id,
              target_body_id, task2_coaster_body_id,
              task_success
            )

            trial_index += 1  # Move to the next trial slot (shared across threads).
            if trial_index >= n_trials_per_task:
              # Writing a compact aligned report block in the evaluation log.
              evaluation_log_path = str(EVAL_LOG_PATH)
              utilities._write_evaluation_log(
                n_trials_per_task, is_scene1_task, is_scene2_task, evaluation_type,
                task_completion_times, task_success,
                max_pitch_measured, max_roll_measured, position_error_eval, evaluation_log_path
              )

              if is_scene1_task:
                if evaluation_type == 1:
                  print(f"[INFO] Evaluation completed for task 1 with evaluation type 1.")
                  print(f"[INFO] Next evaluation will be again with task 1 but type 2.")
                  evaluation_type = 2
                  trial_index = 0
                  max_roll_measured = [np.zeros(8, dtype=np.float32) for _ in range(n_trials_per_task)]
                  max_pitch_measured = [np.zeros(8, dtype=np.float32) for _ in range(n_trials_per_task)]
                  reset_simulation()
                  print(f"[INFO] Starting trial {trial_index + 1} of {n_trials_per_task}...")
                elif evaluation_type == 2:
                  print(f"[INFO] Evaluation completed for task 1 with evaluation type 2.")
                  print(f"[INFO] Next evaluation will start with task 2 with evaluation type 1.")
                  # Restarting the script to switch to task 2 with evaluation type 1.
                  os.execv(sys.executable, [sys.executable, str(SCRIPT_PATH)])
              elif is_scene2_task:
                if evaluation_type == 1:
                  print(f"[INFO] Evaluation completed for task 2 with evaluation type 1.")
                  print(f"[INFO] Next evaluation will start with task 2 with evaluation type 2.")
                  evaluation_type = 2
                  trial_index = 0
                  max_roll_measured = [np.zeros(8, dtype=np.float32) for _ in range(n_trials_per_task)]
                  max_pitch_measured = [np.zeros(8, dtype=np.float32) for _ in range(n_trials_per_task)]
                  reset_simulation()
                elif evaluation_type == 2:
                  print(f"[INFO] Evaluation completed for task 2 with evaluation type 2.")
                  print(f"[INFO] All evaluations for both tasks have been completed. Exiting the simulation.")
                  locker.release()  # Release lock before signaling: PhysicsViewerThread must be able to acquire it to reach viewer.close().
                  all_trials_done.set()  # Signal PhysicsViewerThread to close the viewer (GLFW must be closed from the thread that owns sync).
                  return  # Exit SimulationThread entirely (also exits the outer while viewer.is_running() loop).
                
            else:
              # Resetting the simulation for the next trial. In case of evaluation type 2, 
              # the initial positions of the objects are randomized for each trial.
              reset_simulation()
              print(f"[INFO] Starting trial {trial_index + 1} of {n_trials_per_task}...")
            
        elif control_step % 200 == 0:
          print("[IK] Waiting startup settle before enabling IK...")
          if (is_scene1_task and trial_index == 0 and task1_step == 0) or \
             (is_scene2_task and trial_index == 0 and task2_step == 0):
            print(f"[INFO] Starting trial 1 of {n_trials_per_task}...")

        # Injecting IK joint target positions into target_pos
        for i, full_idx in enumerate(ctrl.right_arm_indices):
          target_pos[full_idx] = float(right_arm_q_des[i])
      ctrl.apply_pd_control(target_pos)
      mujoco.mj_step(mj_model, mj_data)

      # --- exiting critical section --- #
      locker.release()

      control_step += 1
      sim_time += mj_model.opt.timestep
  
  # Viewer has been stopped, checking if last message printed in evaluations.py is an order for executing next task.
  with open(EVAL_LOG_PATH, "a+", encoding="utf-8") as log_file:
    log_file.seek(0)  # go to beginning of file

    lines = log_file.readlines() # read all lines in the file
    last_line = lines[-1].strip() if lines else ""

    if (
        "[ORDER] Execute task_1" not in last_line
        and "[ORDER] Execute task_2" not in last_line
    ):
      # Artificially write order for next execution (by default task 1)
      log_file.write(f"[ORDER] Execute task_1\n")


# Main thread for rendering and synchronizing with the physics thread, also used to set up the camera view.
def PhysicsViewerThread():
    
    # Retrieving right shoulder position for camera setup. 
    # If the body is not found, the camera will fallback to aiming at the palm.
    right_shoulder_body_id = mujoco.mj_name2id(
      mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link"
    )
    configured_camera = False
    while viewer.is_running():

        locker.acquire()
        # --- critical section start (accessing shared variables) --- #

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

        # --- exiting critical section --- #
        locker.release()

        if all_trials_done.is_set():
          viewer.close()  # Closed here (on the viewer thread) to avoid GLFW thread-local-storage assertion.
          break
        time.sleep(evaluation_config.VIEWER_DT) # Sleep to limit viewer thread loop frequency (and avoid excessive CPU usage, since sync is not blocking in this setup).

def RollPitchReaderThread():
    # Thread to read roll and pitch of joints from the IMU sensor and save the highest value for later evaluation.
    # Joints read are (8): left hip, left knee, left ankle, right hip, right knee, right ankle, waist, torso
    global trial_index

    # Mapping order of joints is fixed and matches the layout of arrays used to store the roll and pitch maxima values:
    # [left hip, left knee, left ankle, right hip, right knee, right ankle, waist, torso]
    # (stored values as pair of roll and pitch to match pair of arrays. 
    # However certain joints do not have either roll or pitch, so None is used in those cases)
    tracked = [
      ("left_hip_roll_joint", "left_hip_pitch_joint"),
      (None, "left_knee_joint"),
      ("left_ankle_roll_joint", "left_ankle_pitch_joint"),
      ("right_hip_roll_joint", "right_hip_pitch_joint"),
      (None, "right_knee_joint"),
      ("right_ankle_roll_joint", "right_ankle_pitch_joint"),
      ("waist_roll_joint", "waist_pitch_joint"),
      (None, None),  # torso is read from base (pelvis) quaternion
    ]

    tracked_idx = [(utilities._joint_qpos_index(mj_model, r), utilities._joint_qpos_index(mj_model, p)) for r, p in tracked]

    while viewer.is_running():
        locker.acquire()
        # --- critical section start (accessing shared variables) --- #

        ti = int(trial_index)
        if 0 <= ti < n_trials_per_task:
          # Valid trial index, read roll and pitch values and update maxima.

          # Read roll and pitch from joint states (radians).
          # We track maxima in absolute value over each trial.
          roll_vals = np.zeros(8, dtype=np.float32)
          pitch_vals = np.zeros(8, dtype=np.float32)

          for i, (r_idx, p_idx) in enumerate(tracked_idx):
            if r_idx is not None:
              roll_vals[i] = float(mj_data.qpos[r_idx])
            if p_idx is not None:
              pitch_vals[i] = float(mj_data.qpos[p_idx])

          # Torso roll/pitch from base(pelvis) orientation quaternion.
          torso_roll, torso_pitch = utilities._quat_to_roll_pitch(mj_data.qpos[3:7])
          roll_vals[7] = torso_roll
          pitch_vals[7] = torso_pitch

          max_roll_measured[ti] = np.maximum(
            max_roll_measured[ti],
            np.abs(roll_vals).astype(np.float32),
          )
          max_pitch_measured[ti] = np.maximum(
            max_pitch_measured[ti],
            np.abs(pitch_vals).astype(np.float32),
          )

        # --- exiting critical section --- #
        locker.release()
        time.sleep(0.01) # Sleep to limit IMU reading frequency (and avoid excessive CPU usage).


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    roll_pitch_reader_thread = Thread(target=RollPitchReaderThread)

    viewer_thread.start()
    sim_thread.start()
    roll_pitch_reader_thread.start()