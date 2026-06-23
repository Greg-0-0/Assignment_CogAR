import sys


ROBOT = "g1" # Robot name

def _scene_path(scene_file: str) -> str:
    return "../unitree_robots/" + ROBOT + "/" + scene_file

def get_robot_scene(current_instruction: str) -> str:
    """Resolves scene path before MuJoCo model initialization."""

    instr = current_instruction.strip().lower()
    # Support both canonical tokens and log-style lines such as
    # "[ORDER] Execute task_1".
    if instr.startswith("[order]"):
        parts = instr.split()
        instr = parts[-1] if parts else instr

    if instr in ("task1", "task_1"):
        print("[CONFIG] Current evaluation instruction: task1")
        return _scene_path("scene1.xml")
    elif instr in ("task2", "task_2"):
        print("[CONFIG] Current evaluation instruction: task2")
        return _scene_path("scene2.xml")
    else:
        print(f"[CONFIG] Unrecognized evaluation instruction: {current_instruction}. Defaulting to task1.")
        return _scene_path("scene1.xml")

DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer
