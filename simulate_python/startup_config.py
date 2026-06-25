import sys


ROBOT = "g1" # Robot name

# Startup scene selection:
# - Set SCENE_PRESET to "scene1" or "scene2" to skip prompt.
# - Keep None to ask interactively at launch.
SCENE_PRESET = None

def _scene_path(scene_file: str) -> str:
	return "../unitree_robots/" + ROBOT + "/" + scene_file


def _normalize_scene_name(value: str):
	"""Normalizes scene value for benchmark selection."""

	v = value.strip().lower()
	if v in ("1", "scene1", "scene1.xml"):
		return "scene1.xml"
	if v in ("2", "scene2", "scene2.xml"):
		return "scene2.xml"
	return None


def get_robot_scene() -> str:
	"""Resolves scene path before MuJoCo model initialization."""

	if SCENE_PRESET is not None:
		preset_scene = _normalize_scene_name(SCENE_PRESET)
		if preset_scene is not None:
			return _scene_path(preset_scene)

	if not sys.stdin or not sys.stdin.isatty():
		print("[CONFIG] Non-interactive session detected, defaulting to scene1.xml")
		return _scene_path("scene1.xml")

	while True:
		choice = input("Select task scene [1=task1, 2=task2] (default 1): ").strip()
		if choice == "":
			return _scene_path("scene1.xml")
		chosen = _normalize_scene_name(choice)
		if chosen is not None:
			return _scene_path(chosen)
		print("Invalid choice. Please enter 1 or 2.")

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer
