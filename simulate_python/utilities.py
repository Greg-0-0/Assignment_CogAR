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

