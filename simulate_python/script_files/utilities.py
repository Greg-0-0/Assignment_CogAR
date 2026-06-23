from time import time
from datetime import datetime

import numpy as np

import mujoco

# --------------------------------------------------------------------------- #
# Armature setup
# --------------------------------------------------------------------------- #
def set_armature(model, joint_names):
  """Sets fixed armature values for joints inertia based on naming conventions."""

  ARM_5020 = 0.00360972
  ARM_7520_14 = 0.01017752
  ARM_7520_22 = 0.02510192
  ARM_4010 = 0.00425000
  ARM_2x5020 = 0.00721945

  for i, name in enumerate(joint_names):
    dof = 6 + i
    if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
      model.dof_armature[dof] = ARM_5020
    elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
      model.dof_armature[dof] = ARM_7520_14
    elif "hip_roll" in name or "knee" in name:
      model.dof_armature[dof] = ARM_7520_22
    elif "wrist_pitch" in name or "wrist_yaw" in name:
      model.dof_armature[dof] = ARM_4010
    elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
      model.dof_armature[dof] = ARM_2x5020
    else:
      model.dof_armature[dof] = ARM_5020

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _joint_state_refs(model, joint_names):
  """Returns a dictionary mapping joint names to (qpos, dof) indices."""
  refs = {}
  for joint_name in joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
      raise RuntimeError(f"Joint not found in model: {joint_name}")
    refs[joint_name] = (
      int(model.jnt_qposadr[joint_id]),
      int(model.jnt_dofadr[joint_id]),
    )
  return refs


def _solve_damped_pseudoinverse_dq(jacobian, task_velocity, damping):
  """Implements inverse kinematics: solves desired joint velocities 
    from task cartesian velocities using a damped pseudoinverse."""

  J = np.asarray(jacobian, dtype=np.float64)
  x_dot = np.asarray(task_velocity, dtype=np.float64)
  rows, cols = J.shape
  lam2 = float(damping) * float(damping)

  # Accept bidimensional vector and convert to a task vector.
  if x_dot.ndim == 2:
    if x_dot.shape == (rows, rows):
      # If a diagonal gain matrix was multiplied element-wise by position and orientation errors,
      # the desired vector would be on the diagonal.
      x_dot = np.diag(x_dot)
    elif x_dot.shape == (rows, 1):
      x_dot = x_dot[:, 0]
    else:
      raise ValueError(
        f"task_velocity has invalid shape {x_dot.shape}; expected ({rows},)"
      )
  elif x_dot.ndim != 1:
    raise ValueError(
      f"task_velocity must be a 1D vector; got ndim={x_dot.ndim}"
    )

  if x_dot.shape[0] != rows:
    raise ValueError(
      f"task_velocity length {x_dot.shape[0]} does not match Jacobian rows {rows}"
    )

  # Computing damped pseudoinverse using the formula:
  #   dq = J^T * (J * J^T + lam^2 * I)^-1 * x_dot  if rows <= cols
  #   dq = (J^T * J + lam^2 * I)^-1 * J^T * x_dot  if rows > cols
  if rows <= cols:
    regularized = J @ J.T + lam2 * np.eye(rows, dtype=np.float64)
    dq = J.T @ np.linalg.solve(regularized, x_dot)
  else:
    regularized = J.T @ J + lam2 * np.eye(cols, dtype=np.float64)
    dq = np.linalg.solve(regularized, J.T @ x_dot)

  return dq.astype(np.float32)

# --------------------------------------------------------------------------- #
#               Functions used only in evaluation.py
# --------------------------------------------------------------------------- #

def reset(mj_model, mj_data, ctrl, right_arm_q_min, right_arm_q_max, initial_sim_state, initial_grasp_rot_world,
            evaluation_type, is_scene1_task, is_scene2_task,randomisation_range):
  """Resets variables and structures that have been modified during simulation.
     Necessary for executing multiple evaluation runs in the same process.
     (Used only in evaluation.py)"""

  # Always restore full MuJoCo data state first so every trial starts from the same baseline.
  np.copyto(mj_data.qpos, initial_sim_state["qpos"])
  np.copyto(mj_data.qvel, initial_sim_state["qvel"])
  np.copyto(mj_data.act, initial_sim_state["act"])
  np.copyto(mj_data.ctrl, initial_sim_state["ctrl"])
  np.copyto(mj_data.qacc_warmstart, initial_sim_state["qacc_warmstart"])
  np.copyto(mj_data.mocap_pos, initial_sim_state["mocap_pos"])
  np.copyto(mj_data.mocap_quat, initial_sim_state["mocap_quat"])
  mj_data.time = float(initial_sim_state["time"])

  if evaluation_type == 2:
    # Randomize initial object positions only on x/y while preserving z and orientation.
    dx, dy = np.random.uniform(-randomisation_range, randomisation_range, size=2)
    if is_scene1_task:
      # Keep cylinder and cube relative placement fixed by using the same offset for both.
      _apply_xy_offset_to_free_body(mj_model, mj_data, "blue_cube", dx, dy)
      _apply_xy_offset_to_free_body(mj_model, mj_data, "red_cylinder", dx, dy)
    elif is_scene2_task:
      # In scene 2 randomize the mug x/y spawn.
      mug_name = "mug_object"
      if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, mug_name) < 0:
        mug_name = "mug"
      _apply_xy_offset_to_free_body(mj_model, mj_data, mug_name, dx, dy)

  mujoco.mj_forward(mj_model, mj_data)

  # Defining initial right-arm joint target positions.
  right_arm_joint_refs = _joint_state_refs(mj_model, ctrl.right_arm_joint_names)
  right_arm_q_des = np.array([
    float(mj_data.qpos[right_arm_joint_refs[name][0]])
    for name in ctrl.right_arm_joint_names
  ], dtype=np.float32)

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

  # Reset controller-side transient state used by task orchestration.
  ctrl._cache_finger_actuators(ctrl.obj_name)
  ctrl.grip_transition_duration_s = 1.0
  ctrl.grip_closed = False
  ctrl.grip_close_time = None
  ctrl.grip_transition_start_time = None
  ctrl.grip_alpha_start = 0.0
  ctrl.grip_alpha_goal = 0.0
  ctrl.post_grasp_lift_active = False
  ctrl.post_grasp_lift_start_world = None
  ctrl.post_grasp_lift_final_world = None
  ctrl.post_grasp_lift_start_time = None
  ctrl.post_grasp_lift_target_world = None
  for act_id, open_val, _ in ctrl.right_finger_actuators:
    ctrl.data.ctrl[act_id] = open_val

  # Restore default grasp orientation used at task start.
  current_grasp_rot_world = initial_grasp_rot_world.copy()

  return right_arm_q_des, current_grasp_rot_world

def _apply_xy_offset_to_free_body(mj_model, mj_data, body_name, dx, dy):
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
      raise RuntimeError(f"Body not found in model: {body_name}")
    first_joint_id = int(mj_model.body_jntadr[body_id])
    if first_joint_id < 0 or mj_model.jnt_type[first_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
      raise RuntimeError(f"Body {body_name} does not have a free joint for position randomization")
    qpos_adr = int(mj_model.jnt_qposadr[first_joint_id])
    mj_data.qpos[qpos_adr + 0] += float(dx)
    mj_data.qpos[qpos_adr + 1] += float(dy)

def _compute_task_completion_time(task_start_time, task_end_time, task_completion_times):
  """Computes the task completion time in seconds given the start time and end time."""

  if task_start_time is None or task_end_time is None:
    return None
  task_completion_times.append(float(task_end_time - task_start_time))
  return task_completion_times

def _is_task_successful(is_scene1_task, is_scene2_task, mj_data, task1_blue_cube_body_id,
                        task1_red_basket_body_id, task1_blue_basket_body_id,
                        target_body_id, task2_coaster_body_id,
                        task_success):
  """Determines the type of task success based on the the final positions of objects with respect 
    to their target position (manages both tasks)."""

  if is_scene1_task:
    blue_cube_pos = mj_data.xpos[task1_blue_cube_body_id]
    red_basket_pos = mj_data.xpos[task1_red_basket_body_id]
    cube_in_red_basket = (
      abs(blue_cube_pos[0] - red_basket_pos[0]) < 0.05 and
      abs(blue_cube_pos[1] - red_basket_pos[1]) < 0.05 and
      abs(blue_cube_pos[2] - red_basket_pos[2]) < 0.05
    )

    red_cylinder_pos = mj_data.xpos[target_body_id]  # Assuming target_body_id corresponds to the red cylinder
    blue_basket_pos = mj_data.xpos[task1_blue_basket_body_id]
    cylinder_in_blue_basket = (
      abs(red_cylinder_pos[0] - blue_basket_pos[0]) < 0.05 and
      abs(red_cylinder_pos[1] - blue_basket_pos[1]) < 0.05 and
      abs(red_cylinder_pos[2] - blue_basket_pos[2]) < 0.05
    )
    if cube_in_red_basket and cylinder_in_blue_basket:
      print("[TASK1] Success-Success: Cube and cylinder are in the correct baskets.")
      task_success.append("Success-Success")
    elif cube_in_red_basket:
      print("[TASK1] Success-Failure: Cube is in the red basket, but cylinder is NOT in the blue basket.")
      task_success.append("Success-Failure")
    elif cylinder_in_blue_basket:
      print("[TASK1] Failure-Success: Cube is NOT in the red basket, but cylinder is in the blue basket.")
      task_success.append("Failure-Success")
    else:
      print("[TASK1] Failure-Failure: Neither cube nor cylinder are in the correct baskets.")
      task_success.append("Failure-Failure")
    
    return task_success
    
  elif is_scene2_task:
    mug_pos = mj_data.xpos[target_body_id]  # Assuming target_body_id corresponds to the mug
    coaster_pos = mj_data.xpos[task2_coaster_body_id]
    mug_on_coaster = (
      abs(mug_pos[0] - coaster_pos[0]) < 0.1 and
      abs(mug_pos[1] - coaster_pos[1]) < 0.1 and
      abs(mug_pos[2] - coaster_pos[2]) < 0.1
    )
    if mug_on_coaster:
      print("[TASK2] Success: Mug is on the coaster.")
      task_success.append("Success")
    else:
      print("[TASK2] Failure: Mug is NOT on the coaster.")
      task_success.append("Failure")

    return task_success

def _write_evaluation_log(n_trials_per_task, is_scene1_task, is_scene2_task, evaluation_type,
                          task_completion_times, task_success,
                          max_pitch_measured, max_roll_measured, position_error_eval, EVAL_LOG_PATH):
  """Writes a message to the log file with a timestamp."""
  col_w = 14
  label_w = 22
  joint_labels = [
    "L_HIP",
    "L_KNE",
    "L_ANK",
    "R_HIP",
    "R_KNE",
    "R_ANK",
    "WAIST",
    "TORSO",
  ]

  ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  trial_cols = "".join(f"{i:>{col_w}d}" for i in range(1, n_trials_per_task + 1))
  total_width = label_w + col_w * n_trials_per_task

  with open(EVAL_LOG_PATH, "a", encoding="utf-8") as log_file:
    log_file.write("\n")

    print(f"[INFO] All {n_trials_per_task} trials for task {1 if is_scene1_task else 2} with evaluation type {evaluation_type} completed.")
    print(f"[INFO] Printing evaluation results to log file: quantitative_evaluation/evaluations.log")
    log_file.write(
      f"[{ts}] [INFO] Evaluation results for task {1 if is_scene1_task else 2} with evaluation type {evaluation_type}:\n"
    )
    
    log_file.write(f"{'Trial number':<{label_w}}{trial_cols}\n")
    log_file.write(f"{'-' * total_width}\n")
    _write_row(log_file, "Completion time", task_completion_times, n_trials_per_task, col_w, label_w, decimals=4)
    log_file.write("\n")
    _write_row(log_file, "Success value", task_success, n_trials_per_task, col_w, label_w, decimals=0)
    log_file.write("\n")

    for j, joint_name in enumerate(joint_labels):
      left_label = f"Max pitch {joint_name}" if j == 0 else f"{'':10}{joint_name}"
      per_trial = [float(max_pitch_measured[t][j]) for t in range(n_trials_per_task)]
      _write_row(log_file, left_label, per_trial, n_trials_per_task, col_w, label_w, decimals=4)

    log_file.write("\n")

    for j, joint_name in enumerate(joint_labels):
      left_label = f"Max roll  {joint_name}" if j == 0 else f"{'':10}{joint_name}"
      per_trial = [float(max_roll_measured[t][j]) for t in range(n_trials_per_task)]
      _write_row(log_file, left_label, per_trial, n_trials_per_task, col_w, label_w, decimals=4)

    if is_scene2_task:
      log_file.write("\n")
      _write_row(log_file, "Position error(mug)", position_error_eval, n_trials_per_task, col_w, label_w, decimals=4)

    # Computing mean values of completion time, max pitch, max roll, and position error(when applicable) +
    #  rate of each success case for the current evaluation (set of trials).
    log_file.write("\n")
    average_completion_time = (
      sum(task_completion_times) / len(task_completion_times)
      if task_completion_times
      else None
    )
    left_label1 = f"Average completion time: {_fmt_val(average_completion_time, 4)} - "
    left_label2 = f"Average max pitch: {_fmt_val(np.mean(max_pitch_measured), 4)} - "
    left_label3 = f"Average max roll: {_fmt_val(np.mean(max_roll_measured), 4)} - "
    left_label4 = f"Average position error(mug): {_fmt_val(np.mean(position_error_eval), 4)} " if is_scene2_task else ""
    left_label5 = ""
    if is_scene2_task:
      left_label5 = f"Success rate: {task_success.count('Success') / len(task_success) * 100:.2f}% - " if task_success else "Success rate: n/a - "
      left_label6 = f"Failure rate: {task_success.count('Failure') / len(task_success) * 100:.2f}% " if task_success else "Failure rate: n/a"
      left_label5 = f"{left_label5}{left_label6}"
    elif is_scene1_task:
      left_label6 = f"Success-Success rate: {task_success.count('Success-Success') / len(task_success) * 100:.2f}% - " if task_success else "Success-Success rate: n/a"
      left_label7 = f"Success-Failure rate: {task_success.count('Success-Failure') / len(task_success) * 100:.2f}% - " if task_success else "Success-Failure rate: n/a"
      left_label8 = f"Failure-Success rate: {task_success.count('Failure-Success') / len(task_success) * 100:.2f}% - " if task_success else "Failure-Success rate: n/a"
      left_label9 = f"Failure-Failure rate: {task_success.count('Failure-Failure') / len(task_success) * 100:.2f}% " if task_success else "Failure-Failure rate: n/a"
      left_label5 = f"{left_label6}{left_label7}{left_label8}{left_label9}"
    left_label0 = f"{left_label1}{left_label2}{left_label3}{left_label4}"
    log_file.write(f"{left_label0}\n")
    log_file.write(f"{left_label5}\n")

    if evaluation_type == 2:
      if is_scene1_task:
        # Executing evaluation on task 2.
        log_file.write("\n")
        log_file.write("[ORDER] Execute task_2")
      elif is_scene2_task:
        # Evaluation completed for both tasks. Next evaluation will start with task 1 again (executed by user).
        log_file.write("\n")
        log_file.write("[INFO] Evaluation completed for both tasks")
        log_file.write("\n")
        log_file.write("[ORDER] Execute task_1")

def _fmt_val(v, decimals=4):
  """ Formats a value for logging, handling None and NaN cases."""
  
  if v is None:
    return "n/a"
  try:
    if isinstance(v, (float, np.floating)) and np.isnan(v):
      return "n/a"
    return f"{float(v):.{decimals}f}"
  except (TypeError, ValueError):
    return str(v)

def _write_row(log_file, left_label, values, n_trials_per_task, col_w, label_w, decimals=4):
  """Writes a row of values to the log file with a left-aligned label."""

  clipped = list(values)[:n_trials_per_task]
  while len(clipped) < n_trials_per_task:
    clipped.append(None)
  row = "".join(f"{_fmt_val(v, decimals):>{col_w}}" for v in clipped)
  log_file.write(f"{left_label:<{label_w}}{row}\n")

# ------------------------------------------------------------------------------------- #
# Joint state and orientation utilities (used by evaluation.py in RollPitchReaderThread)
# ------------------------------------------------------------------------------------- #

def _joint_qpos_index(mj_model, joint_name):
  """ Returns the index of the joint's qpos in mj_data.qpos provided its name, or None if the joint is not found. """

  if joint_name is None:
    return None
  jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  if jid < 0:
    return None
  return int(mj_model.jnt_qposadr[jid])

def _quat_to_roll_pitch(quat_wxyz):
  """ Converts a quaternion in wxyz format to roll and pitch angles (in radians). """

  w, x, y, z = [float(v) for v in quat_wxyz]
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = np.arctan2(sinr_cosp, cosr_cosp)

  sinp = 2.0 * (w * y - z * x)
  sinp = float(np.clip(sinp, -1.0, 1.0))
  pitch = np.arcsin(sinp)
  return float(roll), float(pitch)

