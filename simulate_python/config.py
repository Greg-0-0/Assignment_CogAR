import os
import sys


ROBOT = "g1" # Robot name

# Startup scene selection:
# - Set SCENE_PRESET to "scene1" or "scene2" to skip prompt.
# - Keep None to ask interactively at launch.
SCENE_PRESET = None

# Optional environment variable override (highest priority):
# export UNITREE_SCENE=scene1   or   export UNITREE_SCENE=scene2
SCENE_ENV_VAR = "UNITREE_SCENE"


def _scene_path(scene_file: str) -> str:
	return "../unitree_robots/" + ROBOT + "/" + scene_file


def _normalize_scene_name(value: str):
	v = value.strip().lower()
	if v in ("1", "scene1", "scene1.xml"):
		return "scene1.xml"
	if v in ("2", "scene2", "scene2.xml"):
		return "scene2.xml"
	return None


def get_robot_scene() -> str:
	"""Resolve scene path before MuJoCo model initialization."""
	env_value = os.environ.get(SCENE_ENV_VAR)
	if env_value:
		env_scene = _normalize_scene_name(env_value)
		if env_scene is not None:
			return _scene_path(env_scene)

	if SCENE_PRESET is not None:
		preset_scene = _normalize_scene_name(SCENE_PRESET)
		if preset_scene is not None:
			return _scene_path(preset_scene)

	if not sys.stdin or not sys.stdin.isatty():
		print("[CONFIG] Non-interactive session detected, defaulting to scene1.xml")
		return _scene_path("scene1.xml")

	while True:
		choice = input("Select task scene [1=scene1, 2=scene2] (default 1): ").strip()
		if choice == "":
			return _scene_path("scene1.xml")
		chosen = _normalize_scene_name(choice)
		if chosen is not None:
			return _scene_path(chosen)
		print("Invalid choice. Please enter 1 or 2.")
		
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer
