import time
import numpy as np

import mujoco

# --------------------------------------------------------------------------- #
# G1 Controller (walker + IK arm/gripper support)
# --------------------------------------------------------------------------- #
class G1Controller:
  """Full G1 controller with locomotion mode switching and arm control."""
  
  KEY_COMMA_GRIP = 44  # , = Grip toggle

  def __init__(self, model, data, walker, config, obj_name):
    self.model = model
    self.data = data
    self.walker_policy = walker
    self.config = config
    self.obj_name = obj_name

    # Walk state
    self.lin_vel_x = 0.0
    self.lin_vel_y = 0.0
    self.ang_vel_z = 0.0
    self.vel_step_linear = 0.2
    self.vel_step_angular = 0.2
    self.vel_max_linear = 2.0
    self.vel_max_angular = 1.0

    # Frozen arm position — holds the last commanded arm position when IK is idle.
    self.frozen_arm_pos = None  # None = use defaults, array = hold position

    self.last_action = np.zeros(29, dtype=np.float32)

    # Right hand grip state
    self.grip_closed = False
    self.grip_close_time = None
    self.grip_transition_duration_s = 1.0
    self.grip_transition_start_time = None
    self.grip_alpha_start = 0.0
    self.grip_alpha_goal = 0.0
    self.manual_grip_enabled = True
    self.post_grasp_lift_active = False
    self.post_grasp_lift_target_world = None
    self.post_grasp_lift_start_time = None
    self.post_grasp_lift_start_world = None
    self.post_grasp_lift_final_world = None

    # These joints are commanded by local direct ctrl writes.
    self.dds_controlled_joints = {
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    }

    self._build_joint_mappings()
    self._build_arm_mappings()
    self._compute_pd_gains()
    self._cache_actuator_ids()
    self._cache_finger_actuators(self.obj_name)

  def _build_joint_mappings(self):
    self.joint_names = self.config["joint_names"]
    self.num_joints = len(self.joint_names)
    self.joint_qpos_indices = {n: 7 + i for i, n in enumerate(self.joint_names)}
    self.joint_qvel_indices = {n: 6 + i for i, n in enumerate(self.joint_names)}

    self.default_joint_pos = np.zeros(self.num_joints, dtype=np.float32)
    for name, value in self.config["default_joint_pos"].items():
      if name in self.joint_names:
        self.default_joint_pos[self.joint_names.index(name)] = value

    self.action_scales = np.array(
      [self.config["action_scales"][n] for n in self.joint_names], dtype=np.float32
    )

    arm_patterns = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    self.arm_indices = []
    for i, name in enumerate(self.joint_names):
      if any(p in name for p in arm_patterns):
        self.arm_indices.append(i)

  def _build_arm_mappings(self):
    self.right_arm_joint_names = [
      "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint", "right_elbow_joint",
      "right_wrist_roll_joint", "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ]
    self.right_arm_indices = [
      self.joint_names.index(n) for n in self.right_arm_joint_names
      if n in self.joint_names
    ]
    # Arm defaults are simply the defaults from the main model config.
    self.arm_default_pos = np.array([
      self.default_joint_pos[self.joint_names.index(n)]
      for n in self.right_arm_joint_names
    ], dtype=np.float32)
    self.right_palm_site_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm"
    )

  def _compute_pd_gains(self):
    S5020, D5020, E5020 = 14.2506, 0.9072, 25.0
    S7520_14, D7520_14, E7520_14 = 40.1792, 2.5579, 88.0
    S7520_22, D7520_22, E7520_22 = 99.0984, 6.3088, 139.0
    S4010, D4010, E4010 = 16.7783, 1.0681, 5.0

    self.kp = np.zeros(self.num_joints, dtype=np.float32)
    self.kd = np.zeros(self.num_joints, dtype=np.float32)
    self.effort_limit = np.zeros(self.num_joints, dtype=np.float32)

    for i, name in enumerate(self.joint_names):
      if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
        self.kp[i], self.kd[i], self.effort_limit[i] = S5020, D5020, E5020
      elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
        self.kp[i], self.kd[i], self.effort_limit[i] = S7520_14, D7520_14, E7520_14
      elif "hip_roll" in name or "knee" in name:
        self.kp[i], self.kd[i], self.effort_limit[i] = S7520_22, D7520_22, E7520_22
      elif "wrist_pitch" in name or "wrist_yaw" in name:
        self.kp[i], self.kd[i], self.effort_limit[i] = S4010, D4010, E4010
      elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
        self.kp[i], self.kd[i], self.effort_limit[i] = S5020 * 2, D5020 * 2, E5020 * 2
      else:
        self.kp[i], self.kd[i], self.effort_limit[i] = S5020, D5020, E5020

  # --- State helpers ---
  def _get_base_pose(self):
    '''
    Return base(pelvis) position and orientation (quaternion) in world frame
    '''
    return self.data.qpos[:3].copy(), self.data.qpos[3:7].copy()

  @staticmethod
  def _quat_apply_inverse(quat, vec):
    '''
    Computes rotation on provided vector based on quaternions in input
    '''
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

  @staticmethod
  def _quat_to_rotmat(quat):
    '''
    Computes rotation matrix from quaternions
    '''
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)

  def _get_base_velocities(self):
    '''
    Returns base(pelvis) linear and angular velocities in world frame
    '''
    lin_vel_world = self.data.qvel[:3].copy()
    ang_vel_body = self.data.qvel[3:6].copy()
    _, quat = self._get_base_pose()
    return self._quat_apply_inverse(quat, lin_vel_world), ang_vel_body

  def _get_projected_gravity(self):
    '''
    Returns base(pelvis) rotation vector in world frame
    '''
    _, quat = self._get_base_pose()
    return self._quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0]))

  def _get_joint_positions(self):
    '''
    Returns array with positions of all joints defined and ordered as in model_config.json
    '''
    pos = np.zeros(self.num_joints, dtype=np.float32)
    for i, n in enumerate(self.joint_names):
      pos[i] = self.data.qpos[self.joint_qpos_indices[n]] - self.default_joint_pos[i]
    return pos

  def _get_joint_velocities(self):
    '''
    Returns array with velocities of all joints defined and ordered as in model_config.json
    '''
    vel = np.zeros(self.num_joints, dtype=np.float32)
    for i, n in enumerate(self.joint_names):
      vel[i] = self.data.qvel[self.joint_qvel_indices[n]]
    return vel

  def _get_arm_joint_positions(self):
    '''
    Returns array with positions of right arm joints defined and ordered as in model_config.json
    '''
    pos = np.zeros(len(self.right_arm_indices), dtype=np.float32)
    for i, idx in enumerate(self.right_arm_indices):
      n = self.joint_names[idx]
      pos[i] = self.data.qpos[self.joint_qpos_indices[n]] - self.arm_default_pos[i]
    return pos

  def _get_arm_joint_velocities(self):
    '''
    Returns array with velocities of right arm joints defined and ordered as in model_config.json
    '''
    vel = np.zeros(len(self.right_arm_indices), dtype=np.float32)
    for i, idx in enumerate(self.right_arm_indices):
      vel[i] = self.data.qvel[self.joint_qvel_indices[self.joint_names[idx]]]
    return vel

  def _get_palm_pos_in_pelvis(self):
    '''
    Returns position of right palm frame realtive to base(pelvis)
    '''
    palm_world = self.data.site_xpos[self.right_palm_site_id].copy() # palm postion realtive to world frame
    pos, quat = self._get_base_pose()
    return self._quat_apply_inverse(quat, palm_world - pos)

  def _get_palm_orientation_in_pelvis(self):
    '''
    Returns orientation of right palm frame realtive to base(pelvis)
    '''
    mat = self.data.site_xmat[self.right_palm_site_id].reshape(3, 3) # palm orientation as a rotation matrix
    palm_q = np.zeros(4)
    mujoco.mju_mat2Quat(palm_q, mat.flatten()) # palm orientation as quaternions
    _, pelvis_q = self._get_base_pose()
    pinv = np.array([pelvis_q[0], -pelvis_q[1], -pelvis_q[2], -pelvis_q[3]])
    w1, x1, y1, z1 = pinv
    w2, x2, y2, z2 = palm_q
    rel = np.array([
      w1*w2 - x1*x2 - y1*y2 - z1*z2,
      w1*x2 + x1*w2 + y1*z2 - z1*y2,
      w1*y2 - x1*z2 + y1*w2 + z1*x2,
      w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    w, x, y, z = rel # orientation of right palm wrt base(pelvis) in quaternions
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = np.clip(2*(w*y - z*x), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float32)

  def _get_palm_jacobian_in_pelvis(self):
    '''
    Returns linear and angular jacobian matrices of right palm(EE) relative to pelvis(base)
    (comprises basic robot jacobian of last joint relative to base and rigid body jacobian of palm realtive to last joint)
    '''
    jacp_world = np.zeros((3, self.model.nv), dtype=np.float64)
    jacr_world = np.zeros((3, self.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
      self.model,
      self.data,
      jacp_world,
      jacr_world,
      self.right_palm_site_id,
    ) # retrieves from robot model linear and angular jacobian matrices of right palm(site) relative to world

    # Use MuJoCo's true dof addresses for robustness (independent of config ordering).
    arm_dof_indices = []
    for name in self.right_arm_joint_names:
      joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if joint_id < 0:
        raise RuntimeError(f"Joint not found in model: {name}")
      arm_dof_indices.append(int(self.model.jnt_dofadr[joint_id])) # right arm is 7DOF
    # building jacobian matrices dimensions
    jacp_arm_world = jacp_world[:, arm_dof_indices]
    jacr_arm_world = jacr_world[:, arm_dof_indices]

    _, pelvis_q = self._get_base_pose() # quaternions of base(pelvis) relative to world
    pelvis_rot_world = self._quat_to_rotmat(pelvis_q) # rotation matrix from base(pelvis) to world
    world_to_pelvis = pelvis_rot_world.T  # rotation matrix from world to base(pelvis)

    # linear and angular jacobians of all right arm joints up to right palm(EE) realtive to base(pelvis)
    jacp_pelvis = world_to_pelvis @ jacp_arm_world
    jacr_pelvis = world_to_pelvis @ jacr_arm_world
    return jacp_pelvis.astype(np.float32), jacr_pelvis.astype(np.float32)

  @staticmethod
  def _rot_to_angle_axis(R):
    """Convert a 3x3 rotation matrix to angle-axis with SO(3) checks."""

    # Checking matrix validity
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
      raise ValueError(f"Rotation matrix must be 3x3, got {R.shape}")

    tol = 1e-3
    det_R = np.linalg.det(R)
    ortho_err = np.linalg.norm(np.eye(3, dtype=np.float64) - (R.T @ R))
    if abs(det_R - 1.0) >= tol or ortho_err >= tol:
      # Non-orthonormal matrix
      raise ValueError(
        f"Invalid rotation matrix: det={det_R:.6f}, ortho_err={ortho_err:.6e}"
      )

    # Conversion in angle-axis vector
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < tol:
      # theta = 0 -> arbitrary unitary h
      h = np.array([1.0, 0.0, 0.0], dtype=np.float64) 
    elif theta > (np.pi - tol) and theta < np.pi:
      # theta = pi
      h = np.array([
        np.sqrt(max((R[0, 0] + 1.0) / 2.0, 0.0)),
        np.sqrt(max((R[1, 1] + 1.0) / 2.0, 0.0)),
        np.sqrt(max((R[2, 2] + 1.0) / 2.0, 0.0)),
      ], dtype=np.float64)
      h_norm = np.linalg.norm(h)
      if h_norm < 1e-12:
        h = np.array([1.0, 0.0, 0.0], dtype=np.float64)
      else:
        h = h / h_norm
    else:
      # generica theta
      S = 0.5 * (R - R.T)
      vex_S = np.array([
        S[2, 1],
        -S[2, 0],
        S[1, 0],
      ], dtype=np.float64)
      sin_theta = np.sin(theta)
      if abs(sin_theta) < 1e-12:
        h = np.array([1.0, 0.0, 0.0], dtype=np.float64)
      else:
        h = vex_S / sin_theta

    return h, theta

  def _get_palm_rot_in_pelvis_mat(self):
    """Return the 3x3 rotation matrix of the palm in the pelvis frame."""
    wRe = self.data.site_xmat[self.right_palm_site_id].reshape(3, 3)
    _, pelvis_q = self._get_base_pose()
    world_to_pelvis = self._quat_to_rotmat(pelvis_q).T # pelvis with respect to world
    return world_to_pelvis @ wRe

  def compute_ee_cartesian_velocity(self, goal_pos_pelvis, goal_rot_pelvis, k_l=1.0, k_a=1.0):
    """Compute desired 6D EE Cartesian velocity to reach a goal pose.

    Follows the resolved-rate formulation:
      r    = p_goal - p_ee                        (translational error between goal and EE in pelvis frame)
      bRe  = current EE rotation in pelvis frame
      bRg  = goal rotation in pelvis frame
      R_err = bRe.T @ bRg                         (goal orientation relative to EE)
      (h, theta) = RotToAngleAxis(R_err)
      rho  = bRe @ (h * theta)                    (angular error[rotation vector] in pelvis frame)
      error = [rho; r]
      x_dot = diag(k_a*I3, k_l*I3) @ error

    Args:
      goal_pos_pelvis : (3,) target position in pelvis frame.
      goal_rot_pelvis : (3,3) target rotation matrix in pelvis frame.
      k_l             : linear gain (scalar).
      k_a             : angular gain (scalar).

    Returns:
      x_dot : (6,) desired Cartesian velocity [angular (3); linear (3)] in pelvis frame.
    """
    p_ee = self._get_palm_pos_in_pelvis().astype(np.float64)
    bRe  = self._get_palm_rot_in_pelvis_mat()
    bRg  = np.asarray(goal_rot_pelvis, dtype=np.float64)

    # Translational error
    r = np.asarray(goal_pos_pelvis, dtype=np.float64) - p_ee

    # Rotational error via angle-axis
    R_err = bRe.T @ bRg
    h, theta = self._rot_to_angle_axis(R_err)
    rho = bRe @ (h * theta)           # angular error expressed in pelvis frame

    error = np.concatenate([rho, r])  # [angular (3); linear (3)]

    delta = np.diag([k_a, k_a, k_a, k_l, k_l, k_l])

    x_dot = delta @ error             # element-wise: equivalent to diag(delta) @ error
    return x_dot.astype(np.float32)

  # --- Step ---
  def step(self) -> np.ndarray:
    # Build walker observation (always runs — keeps legs stable)
    lin_vel, ang_vel = self._get_base_velocities()
    proj_gravity = self._get_projected_gravity()
    joint_pos = self._get_joint_positions()
    joint_vel = self._get_joint_velocities()

    cmd = np.array([self.lin_vel_x, self.lin_vel_y, self.ang_vel_z], dtype=np.float32)

    obs = np.concatenate([
      lin_vel, ang_vel, proj_gravity, joint_pos, joint_vel, self.last_action, cmd,
    ]).astype(np.float32)

    # Walker policy (handles legs, waist and torso for standing)
    action = self.walker_policy(obs)
    target_pos = self.default_joint_pos + action * self.action_scales

    # Arms: left arm always at default, right arm holds last reach position
    for idx in self.arm_indices:
      target_pos[idx] = self.default_joint_pos[idx]

    # Right arm: if we have a frozen position from IK/startup, hold it.
    if self.frozen_arm_pos is not None:
      for i, full_idx in enumerate(self.right_arm_indices):
        target_pos[full_idx] = self.frozen_arm_pos[i]

    self.last_action = action.copy()
    return target_pos

  def _cache_actuator_ids(self):
    """Cache actuator IDs once at init instead of looking up every step."""
    self.actuator_ids = []
    for name in self.joint_names:
      self.actuator_ids.append(self._resolve_actuator_id(name))

  @staticmethod
  def _actuator_name_candidates(joint_name: str):
    if joint_name.endswith("_joint"):
      return [joint_name, joint_name[:-6]]
    return [joint_name, f"{joint_name}_joint"]

  def _resolve_actuator_id(self, joint_name: str) -> int:
    for candidate in self._actuator_name_candidates(joint_name):
      actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate)
      if actuator_id >= 0:
        return actuator_id
    return -1

  def _cache_finger_actuators(self, obj_name, open_targets=None):
    """Cache right hand finger actuator IDs and their closed targets."""
    # (actuator_id, open_position, closed_position)
    self.right_finger_actuators = []
    '''
    for blue cube
    finger_closed = {
      "right_hand_thumb_0_joint":  0.0,     # curl thumb inward
      "right_hand_thumb_1_joint": -0.3,     # flex thumb
      "right_hand_thumb_2_joint": -0.6,     # curl thumb tip
      "right_hand_index_0_joint":  0.9,     # curl index
      "right_hand_index_1_joint":  0.9,     # curl index tip
      "right_hand_middle_0_joint": 0.9,     # curl middle
      "right_hand_middle_1_joint": 0.9,     # curl middle tip
    }
    with:
    ik_target_x_offset_world = 0.02 # offset along x direction cube center
    ik_target_y_offset_world = 0.01 # offset along y direction cube center
    ik_target_z_offset_world = 0.08  # offset above cube center
    finger_closed = {
      "right_hand_thumb_0_joint":  0.0,     # curl thumb inward
      "right_hand_thumb_1_joint": -0.4,     # flex thumb
      "right_hand_thumb_2_joint": -0.7,     # curl thumb tip
      "right_hand_index_0_joint":  0.85,     # curl index
      "right_hand_index_1_joint":  0.85,     # curl index tip
      "right_hand_middle_0_joint": 0.85,     # curl middle
      "right_hand_middle_1_joint": 0.85,     # curl middle tip
    }
    with:
    ik_target_x_offset_world = 0.01 # offset along x direction cube center
    ik_target_y_offset_world = 0.0 # offset along y direction cube center
    ik_target_z_offset_world = 0.08  # offset above cube center
    and:
    roll_local_x = np.array([
    [1.0, 0.0, 0.0],
    [0.0, cr, -sr],
    [0.0, sr,  cr],
    ], dtype=np.float32)

    for red cylinder
    finger_closed = {
      "right_hand_thumb_0_joint":  0.2,     # curl thumb inward
      "right_hand_thumb_1_joint": -0.75,     # flex thumb
      "right_hand_thumb_2_joint": -1.4,     # curl thumb tip
      "right_hand_index_0_joint":  1.5,     # curl index
      "right_hand_index_1_joint":  1.5,     # curl index tip
      "right_hand_middle_0_joint": 1.5,     # curl middle
      "right_hand_middle_1_joint": 1.5,     # curl middle tip
    }
    with:
    ik_target_x_offset_world = 0.02 # offset along x direction cylinder center
    ik_target_y_offset_world = -0.05 # no offset along y direction cylinder center
    ik_target_z_offset_world = 0.0  # no offset above cylinder center
    and:
    no_rotation = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    for frustum:
    finger_closed = {
      "right_hand_thumb_0_joint":  0.0,     # curl thumb inward
      "right_hand_thumb_1_joint": -0.1,     # flex thumb
      "right_hand_thumb_2_joint": -0.9,     # curl thumb tip
      "right_hand_index_0_joint":  0.6,     # curl index
      "right_hand_index_1_joint":  1.46,     # curl index tip
      "right_hand_middle_0_joint": 0.6,     # curl middle
      "right_hand_middle_1_joint": 1.44,     # curl middle tip
    }
    with:
    ik_target_x_offset_world = 0.0245 # offset along x direction cylinder center
    ik_target_y_offset_world = 0.0 # no offset along y direction cylinder center
    ik_target_z_offset_world = 0.105  # no offset above cylinder center
    and: 
    roll_local_x = np.array([
    [1.0, 0.0, 0.0],
    [0.0, cr, -sr],
    [0.0, sr,  cr],
    ], dtype=np.float32)
    '''
    if obj_name == "red_cylinder":
      # Red cylinder
      finger_closed = {
        "right_hand_thumb_0_joint":  0.2,     # curl thumb inward
        "right_hand_thumb_1_joint": -0.75,     # flex thumb
        "right_hand_thumb_2_joint": -1.4,     # curl thumb tip
        "right_hand_index_0_joint":  1.5,     # curl index
        "right_hand_index_1_joint":  1.5,     # curl index tip
        "right_hand_middle_0_joint": 1.5,     # curl middle
        "right_hand_middle_1_joint": 1.5,     # curl middle tip
      }
    elif obj_name == "blue_cube":
      # Blue cube
      finger_closed = {
      "right_hand_thumb_0_joint":  0.0,     # curl thumb inward
      "right_hand_thumb_1_joint": -0.4,     # flex thumb
      "right_hand_thumb_2_joint": -0.7,     # curl thumb tip
      "right_hand_index_0_joint":  0.85,     # curl index
      "right_hand_index_1_joint":  0.85,     # curl index tip
      "right_hand_middle_0_joint": 0.85,     # curl middle
      "right_hand_middle_1_joint": 0.85,     # curl middle tip
    }
    elif obj_name == "mug_object":
      # White mug
      finger_closed = {
      "right_hand_thumb_0_joint":  0.0,     # curl thumb inward
      "right_hand_thumb_1_joint": -0.8,     # flex thumb
      "right_hand_thumb_2_joint": -1.2,     # curl thumb tip
      "right_hand_index_0_joint":  1.4,     # curl index
      "right_hand_index_1_joint":  1.4,     # curl index tip
      "right_hand_middle_0_joint": 1.4,     # curl middle
      "right_hand_middle_1_joint": 1.4,     # curl middle tip
    }
    
    for name, closed_val in finger_closed.items():
      aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      if aid >= 0:
        open_val = 0.0
        if open_targets is not None:
          open_val = float(open_targets.get(name, 0.0))
        self.right_finger_actuators.append((aid, open_val, closed_val))

  def apply_pd_control(self, target_pos):
    for i, act_id in enumerate(self.actuator_ids):
      if self.joint_names[i] in self.dds_controlled_joints:
        continue
      if act_id >= 0:
        self.data.ctrl[act_id] = target_pos[i]

    # Apply grip
    grip_alpha = self._get_grip_alpha()
    for act_id, open_val, closed_val in self.right_finger_actuators:
      self.data.ctrl[act_id] = (1.0 - grip_alpha) * open_val + grip_alpha * closed_val

  def _get_grip_alpha(self) -> float:
    """Return smoothed [0..1] grip command interpolation."""
    if self.grip_transition_start_time is None:
      return float(self.grip_alpha_goal)

    elapsed = time.time() - self.grip_transition_start_time
    if self.grip_transition_duration_s <= 1e-6:
      alpha = float(self.grip_alpha_goal)
      self.grip_transition_start_time = None
      self.grip_alpha_start = alpha
      self.grip_alpha_goal = alpha
      return alpha

    t = float(np.clip(elapsed / self.grip_transition_duration_s, 0.0, 1.0))
    alpha = (1.0 - t) * self.grip_alpha_start + t * self.grip_alpha_goal
    if t >= 1.0:
      self.grip_transition_start_time = None
      self.grip_alpha_start = float(alpha)
      self.grip_alpha_goal = float(alpha)
    return float(alpha)

  def _start_grip_transition(self, close_target: bool) -> None:
    current_alpha = self._get_grip_alpha()
    self.grip_alpha_start = current_alpha
    self.grip_alpha_goal = 1.0 if close_target else 0.0
    self.grip_transition_start_time = time.time()

  def set_grip_state(self, close_target: bool) -> None:
    """Set right hand grip state explicitly and reset lift state when toggled."""
    if self.grip_closed == close_target:
      return

    self.grip_closed = close_target
    self._start_grip_transition(self.grip_closed)
    if self.grip_closed:
      self.grip_close_time = time.time()
    else:
      self.grip_close_time = None

    self.post_grasp_lift_active = False
    self.post_grasp_lift_target_world = None
    self.post_grasp_lift_start_time = None
    self.post_grasp_lift_start_world = None
    self.post_grasp_lift_final_world = None
    print(f"[GRIP] Right hand: {'CLOSED' if self.grip_closed else 'OPEN'}")

  # --- Keyboard ---
  def key_callback(self, key: int) -> None:
    # Grip toggle (works in any mode)
    if key == self.KEY_COMMA_GRIP:
      if not self.manual_grip_enabled:
        return
      self.set_grip_state(not self.grip_closed)
      return
