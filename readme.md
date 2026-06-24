# Assignment 9: Subgroup D2
G1 EDU Grasping and Manipulation Controller (SIMULATION)
Student id: 5658523

## Assignment instructions
- Develop and evaluate basic reaching, grasping, and manipulation capabilities for the Unitree G1 EDU humanoid robot in MuJoCo.
    1. Set up the G1 EDU model in MuJoCo with focus on torso, arms, and hands/end-effectors.
    2. Configure simple object interaction scenarios using: cubes, cylinders, tabletop objects
    3. Implement controllers for: reaching, inverse kinematics or operational-space control, grasp execution, object transport and release
    4. Define at least two benchmark tasks, such as: tabletop pick-and-place, object relocation, simple handover scenario
    5. Evaluate performance using metrics such as: grasp success rate, task completion time, final placement accuracy, body stability during manipulation
    6. If feasible, compare two manipulation strategies, for example: purely kinematic control, feedback-based or learned control
- Software needed: MuJoCo, Python, inverse kinematics libraries, Unitree G1 EDU model resources, NumPy, Matplotlib
- Research needed: Humanoid manipulation, reaching and grasping control, inverse kinematics for humanoid robots, whole-body coordination, manipulation benchmarking in simulation
- Deliverables: Working G1 EDU manipulation setup in MuJoCo, reaching and grasping controller, benchmark task scenarios, quantitative evaluation report, demo videos
- The robot itself features 29 degrees of freedom (DOF). Equipped with the Dex 3-1 hands, the total DOF increases to 43. The Dex 3-1 version is F-1515-214.

# Assignment explanation

The software was implemented by expanding and modifying the code structure of the git hub repository https://github.com/unitreerobotics/unitree_mujoco, focusing on the robot model G1 EDU from Unitree. The project consists in two benchmark tasks, each one with a different scenario and goal. Both scenarios feature a G1 robot standing in front of a table on which various objects are placed. The items displayed on the table vary depending on the objective of the task.
## Benchmark tasks definition
- Tabletop pick-and-drop: in the center of desk there is a blue cube with a red cylinder on top of it, while on the right-hand side (robot point of view) there are two baskets, one blue and the other red. The robot has to drop the blue cube inside the red basket, thus first needing to move the cylinder out of the way, in order to grab the cube. To make the task more engaging, and since the cylinder needs to be moved in any case, the robot, after grasping the cylinder, has also to drop it inside the blue basket. This action represents a secondary goal that the machine has to fulfill in order to complete the task successfully.
- Object relocation: the robot has to move a mug from an initial position to a coaster, grabbing its handle. Although it may seem a brief task to implement, the challenge of this sequence of actions lies in the delicacy the robot has to show while manipulating the object, especially when placing it on top of the coaster, to avoid tipping it. Indeed, given the particular shape of the item to grasp, the robot needs more precision to carry out the goal than it does for the previous problem.