# Introduction
G1 EDU Grasping and Manipulation Controller (SIMULATION)
Student id: 5658523

## Assignment instructions
Develop and evaluate basic reaching, grasping, and manipulation capabilities for the Unitree G1 EDU humanoid robot in MuJoCo. 
Steps to follow:
1. Set up the G1 EDU model in MuJoCo with focus on torso, arms, and hands/end-effectors.
2. Configure simple object interaction scenarios using: cubes, cylinders, tabletop objects.
3. Implement controllers for: reaching, inverse kinematics or operational-space control, grasp execution, object transport and release.
4. Define at least two benchmark tasks, such as: tabletop pick-and-place, object relocation, simple handover scenario.
5. Evaluate performance using metrics such as: grasp success rate, task completion time, final placement accuracy, body stability during manipulation.
6. If feasible, compare two manipulation strategies, for example: purely kinematic control, feedback-based or learned control.

Additional information:
- Software needed: MuJoCo, Python, inverse kinematics libraries, Unitree G1 EDU model resources, NumPy, Matplotlib.
- Research needed: Humanoid manipulation, reaching and grasping control, inverse kinematics for humanoid robots, whole-body coordination, manipulation benchmarking in simulation.
- Deliverables: Working G1 EDU manipulation setup in MuJoCo, reaching and grasping controller, benchmark task scenarios, quantitative evaluation report, demo videos.
- The G1 EDU robot model features 29 degrees of freedom (DOF). Equipped with the Dex 3-1 hands, the total DOF increases to 43. The Dex 3-1 version used is F-1515-214.

## Assignment explanation
The software was implemented by expanding and modifying the code structure of the git hub repository https://github.com/unitreerobotics/unitree_mujoco, focusing on the robot model **G1 EDU from Unitree**. The project consists in two benchmark tasks, each one with a different scenario and goal. Both scenarios feature a G1 robot standing in front of a table on which various objects are placed. The items displayed on the table vary depending on the objective of the task.

### Benchmark tasks description
The tasks are carried out using only one hand, since it is enough to reach the goals, and the rest of the joints actively contribute to balancing the robot.

1. **Tabletop pick-and-drop**: in this scenario there is a blue cube with a red cylinder on top of it located in the center of the desk, while on the right-hand side (robot point of view) there are two baskets, one blue and the other red. The robot has to drop the blue cube inside the red basket, thus first needing to move the cylinder out of the way, in order to grab the cube. To make the task more engaging, and since the cylinder needs to be moved in any case, the robot, after grasping the cylinder, has also to drop it inside the blue basket. This action represents a secondary goal that the machine has to fulfill in order to complete the task successfully.

2. **Object relocation**: the robot has to move a mug from an initial position to a coaster, grabbing its handle. Both items are placed on the table. Although it may seem a brief task to implement, the challenge of this sequence of actions lies in the delicacy the robot has to show while manipulating the object, to avoid tipping it, especially when placing it on top of the coaster. Indeed, given the particular shape of the item to grasp, the robot needs more precision to carry out the goal than it does in the previous problem.

# Software requirements
The code is written in python language using multiple modules and the physics simulator `mujoco` with its libraries. It is advised to use a python environment, as a lot of software is required to run the application.

## Dependencies
- Python >= 3.8
- cyclonedds == 0.10.2
- numpy
- opencv-python
- onnxruntime

The code to install the dependencies:
```bash
cd ~
sudo apt install python3-pip
pip3 install mujoco cyclonedds numpy opencv-python onnxruntime
```
## Installation
```bash
git clone https://github.com/Greg-0-0/Assignment_CogAR
```

# Execution guidlines
## Running the benchmark tasks
The application can be run with the command:
```bash
cd ~/simulate_python
python3 startup.py
```
This prompts a request in the terminal asking to choose either the first or the second task (`1` or `2`).
Once the task is choosen, the simulation starts and prints various information in the terminal during the execution such as:
- norm of the **position error** (meters) and **rotation error** (degrees) of the right hand palm pose (End-Effector) relative to the target pose.
- norm of both the **cartesian velocity vector** and the **joint velocity vector**.
- cleared **steps** of the task and next ones to follow.
- information relative to **right hand status** (`OPEN` or `CLOSE`).

## Running the evaluations
The repository also provides a script to run a series of tests to evaluate certain **metrics**. The evaluation consists of **40 trials** for each task, half of which run with fixed object positions, while the other half applying a randomisation to their spawning locations. These variations are applied only to the positons of the objects that must be grasped by the robot.
It is possible to increase the number of trials in the corresponding script. However, changing any other parameter may significantly impact performance and correct execution.
The command to run evaluation:
```bash
cd ~/simulate_python
python3 evaluation.py
```
The evaluation always follows this order:
- 20 trials of task 1 scenario without randomised object positions.
- 20 trials of task 1 scenario with randomised object positions.
- 20 trials of task 2 scenario without randomised object positions.
- 20 trials of task 2 scenario with randomised object positions.

If the evaluation gets interrupted before it reaches the end, the next execution will start again from scenario 1 without randomised object positions.
The results of the tests are printed at [path-to-results](simulate_python/quantitative_evaluation/evaluations.log) as the evaluation gets executed.
A round of results is already present in the file, since it was used by myself to test the evaluation process in the first place.
The log file **must not** be modified, the application automatically sets it up correctly to execute the next evaluation, even in case of abrupt interruptions of the execution.

### Metrics analyzed
The testing process provides different information for evaluating the simulation. The type of data can be checked from the exemple already present in [evaluations.log](simulate_python/quantitative_evaluation/evaluations.log):

1. **Completion time** -> time taken to complete the corresponding trial.
2. **Success value**:
    - in case of task 1: 
        1. *Suc-Suc* -> Both the cube and the cylinder have been placed correctly in their corresponding baskets.
        2. *Suc-Fail* -> Only the cube has been placed in its corresponding basket.
        3. *Fail-Suc* -> Only the cylinder has been placed in its corresponding basket.
        4. *Fail-Fail* -> Neither cube or cylinder placed in baskets.
    - in case of task 2: 
        1. *Success* -> the mug has been placed on  the coaster.
        2. *Failure* -> The mug has not been placed on the coaster.
3. **Maximum pitch** measured as an absoulte value for each joint actively stabilyzing the robot during the corresponding trial:
    - *L_HIP* -> left hip
    - *L_KNE* -> left knee
    - *L_ANK* -> left ankle
    - *R_HIP* -> right hip
    - *R_KNE* -> right knee
    - *R_ANK* -> right ankle
    - *WAIST*
    - *TORSO*
3. **Maximum roll** measured as an absoulte value for each joint actively stabilyzing the robot during the corresponding trial:
    - *L_HIP* -> left hip
    - *L_KNE* -> left knee
    - *L_ANK* -> left ankle
    - *R_HIP* -> right hip
    - *R_KNE* -> right knee
    - *R_ANK* -> right ankle
    - *WAIST*
    - *TORSO*
4. **Position error** which is only computed for the final mug placement in the second task, since the first task does not require a precise final position check for the cube nor for the cylinder.

Finally, some statistical data computed from each trial results:
1. **Average completion time**
2. **Average max pitch**
3. **Average max roll**
4. in case of the first scenario:
    - **Success-Success rate**
    - **Success-Failure rate**
    - **Failure-Success rate**
    - **Failure-Failure rate**

   in case of the second scenario:
    - **Success rate**
    - **Failure rate**
5. only for the second scenario:
    - **Average position error** (mug final placement)

# Project structure
The behavior of the robot depends on different parts of the assignment, each one with a precise role.
For a more detalied analysis of the code, it is advised to directly consult the comments in the various files.

<!-- 
- inside [g1](unitree_robots/g1/) there are defined the physical model for the G1 EDU Unitree robot and the scenarios for both tasks.
- [simulate_python](simulate_python/) which comprises many sub parts:
    - the root folder itself where there are the scripts to execute the simulation and the evaluation process as well as their corresponding configuration files.
    - [script_files](simulate_python/script_files/) in which there are the definitions of the classes for controlling the G1 robot movments, managing the balancing policy and additional useful functions.
    - [policy_resources](simulate_python/policy_resources/) that provides the necessary data to correcty load the balancing policy.
    - [quantitative_evaluation](simulate_python/quantitative_evaluation/) which, as stated before, contains the log file with all the results of the carried out evaluations.
    -->

## G1 Robot model
Inside [g1](unitree_robots/g1/) folder there are defined the physical model for the G1 EDU Unitree robot and the scenarios for both tasks:
- [assets](unitree_robots/g1/assets/) provides the 3D models for each joint of the robot, including the hands.
- [g1.xml](unitree_robots/g1/g1.xml) defines the physical links between the different joints, their geometry, the type of actuator used for each on of them and sensors for measuring joint positions, veocities and inclinations. The file uses *position actuators* which differ from *motor actuators*, since the former excpets as input target joint positions and delegates to the `mujoco` engine the computation of the necessary control signals to reach that goal, while the latter interprets the signals recevied as torques.
- [scene1.xml](unitree_robots/g1/scene1.xml) and [scene2.xml](unitree_robots/g1/scene2.xml) implements respectively the scenarios for the first and second task, like the definition of the items to be manipulated by the robot and the table.

## Balancing policy
The [policy_resources](simulate_python/policy_resources/) folder provides the necessary data to correcty load the *walker* policy, which balances the robot, maintaining its standing posture. This policy was imported from the git hub repository https://github.com/luckyrobots/g1-manipulation-challenge/tree/main and adapted to the assignment needs.
In particular, the model takes in input information from sensors like joint positions, gyroscope values and foot pressure, and returns velocity commands for the 29 joints defined in [model_config.json](simulate_python/policy_resources/model_config.json) and mapped to the corresponding joint names decleared in [g1.xml](unitree_robots/g1/g1.xml).
 The model is split into two files:
- *walker.onnx* : stores the layers of the neural network with weights, other than metadata.
- *walker.onnx.data* : the actual values of the parameters that represent the outputs (torque, angles applied to the joints)

In the previous project, the "walking" behavior was achieved by tilting the torso either backward or foreward, causing the policy to balance the robot, thus making it effectively walk in the process. However, in this assignment the policy is exclusively used to maintain the robot in a crouched standing position and to stabilise its posture during object manipulation.
