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


# Fucntion used only in evaluation.py to reset the simulation state between evaluation runs.
def reset(mj_model, mj_data, ctrl, right_arm_q_min, right_arm_q_max,
          initial_sim_state, initial_grasp_rot_world):
  """Resets variables and structures that have been modified during simulation.
     Necessary for executing multiple evaluation runs in the same process.
     (Used only in evaluation.py)"""

  # Restore full MuJoCo data state so free bodies (objects) return to their original pose.
  np.copyto(mj_data.qpos, initial_sim_state["qpos"])
  np.copyto(mj_data.qvel, initial_sim_state["qvel"])
  np.copyto(mj_data.act, initial_sim_state["act"])
  np.copyto(mj_data.ctrl, initial_sim_state["ctrl"])
  np.copyto(mj_data.qacc_warmstart, initial_sim_state["qacc_warmstart"])
  np.copyto(mj_data.mocap_pos, initial_sim_state["mocap_pos"])
  np.copyto(mj_data.mocap_quat, initial_sim_state["mocap_quat"])
  mj_data.time = float(initial_sim_state["time"])
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

