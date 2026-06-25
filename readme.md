# Contents
- [Introduction](#introduction)
    - [Assignment instructions](#assignment-instructions)
    - [Assignment explanation](#assignment-explanation)
        - [Benchmark tasks description](#benchmark-tasks-description)
- [Software requirements](#software-requirements)
    - [Dependencies](#dependencies)
    - [Installation](#installation)
- [Execution guidlines](#execution-guidlines) 
    - [Running the benchmark tasks](#running-the-benchmark-tasks)
    - [Running the evaluations](#running-the-evaluations)
        - [Metrics measured](#metrics-measured)
- [Project structure](#project-structure) 
    - [G1 Robot model](#g1-robot-model)
    - [Balancing policy](#balancing-policy)
    - [G1 controller and auxiliary functions](#g1-controller-and-auxiliary-functions)
    - [Utilities](#utilities)
    - [Simulation files](#simulation-files)
        - [Simulation thread](#simulation-thread)
        - [Physical thread](#physical-thread)
        - [Roll and pitch reader thread](#roll-and-pitch-reader-thread)
    - [Configuration files](#configuration-files)

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
The code is written in python language using multiple modules and the physics simulator MuJoCo` with its libraries. It is advised to use a python environment, as a lot of software is required to run the application.

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
A round of results is already present in the file, since it was used by the author to test the evaluation process in the first place. There is an additional set of results for task 2, since the evaluation was carried out trying two different randomisation factors with the goal to improve the performance.
The log file **must not** be modified, the application automatically sets it up correctly to execute the next evaluation, even in case of abrupt interruptions of the execution.

### Metrics measured
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

## G1 Robot model
Inside [g1](unitree_robots/g1/) folder there are defined the physical model for the G1 EDU Unitree robot and the scenarios for both tasks:
- [assets](unitree_robots/g1/assets/) provides the 3D models for each joint of the robot, including the hands.
- [g1.xml](unitree_robots/g1/g1.xml) defines the physical links between the different joints, their geometry, the type of actuator used for each on of them and sensors for measuring joint positions, veocities and inclinations. The file uses *position actuators* which differ from *motor actuators*, since the former excpets as input target joint positions and delegates to the MuJoCo engine the computation of the necessary control signals to reach that goal, while the latter interprets the signals recevied as torques.
- [scene1.xml](unitree_robots/g1/scene1.xml) and [scene2.xml](unitree_robots/g1/scene2.xml) implements respectively the scenarios for the first and second task, like the definition of the items to be manipulated by the robot and the table.

## Balancing policy
The [policy_resources](simulate_python/policy_resources/) folder provides the necessary data to correcty load the *walker* policy, which balances the robot, maintaining its standing posture. This policy was imported from the git hub repository [g1-manipulation-challenge](https://github.com/luckyrobots/g1-manipulation-challenge/tree/main) and adapted to the assignment needs.
In particular, the model takes in input information from sensors like joint positions, gyroscope values and foot pressure, and returns velocity commands for the 29 joints defined in [model_config.json](simulate_python/policy_resources/model_config.json) and mapped to the corresponding joint names decleared in [g1.xml](unitree_robots/g1/g1.xml).
 The model is split into two files:
- *walker.onnx* : stores the layers of the neural network with weights, other than metadata.
- *walker.onnx.data* : the actual values of the parameters that represent the outputs (torque, angles applied to the joints)

The policy is loaded thanks to the *ONNXPolicy* class defined at [path-to-resource](simulate_python/script_files/ONNXPolicy.py).

In the previous project, the "walking" behavior was achieved by tilting the torso either backward or foreward, causing the policy to balance the robot, thus making it effectively walk in the process. However, in this assignment the policy is exclusively used to maintain the robot in a crouched standing position and to stabilise its posture during object manipulation.

## G1 controller and auxiliary functions
As explained previously in this document, the robot receives two different control signals:
- the *walker* policy that stabilise its posture.
- inverse kinematics to compute the target positions for the right arm and hand joints, which allow to achieve the necessary motions for actions like reaching and grasping.

The two control strategies drive different joints, so there is no contrast between them. However, both are applied on the robot thanks to the methods of the class [G1Controller](simulate_python/script_files/G1Controller.py).
In particular, this class provides:
- methods to build mappings from joint names to indeces to access important information like joint positions and velocities.
- helper functions to retrieve necessary information for the inverse kinematics:
    - position and orientation of the base frame, which is identified by the robot pelvis.
    - positions and velocities of all joints.
    - position and orientation of the robot end-effector, which is the right hand palm.
    - complete jacobian matrices of the end-effector with respect to the base.
- method to compute the necessary cartesian velocities to apply on the end-effector to reach the goal position and orientation
- functions to control the joints of the right hand to implement the closing and opening action.
- step function and actual control function to apply on each joint the target positions during the simulation.

The implementation of the inverse kinematics required to the user is only partial, since MuJoCo itself already computes the geometric and kinematic model of the robot thanks to the information provided in [g1.xml](unitree_robots/g1/g1.xml), thus providing the jacobian matrices of any joint relative to a world frame.
The missing parts consist in:
- the computation of the *misalignment problem* and the *distance zeroing problem* to retrieve the cartesian velocities of the end-effector.
- the application of the *inverse jacobian relation* to obtain the desired joint velocities.
- integration over time of the joint velocities obtained to retrieve the target joint positions.

## Utilities
In the file [Utilities](simulate_python/script_files/utilities.py) are present useful functions for various applications such as:
- setting up an additional inertia for the robot joints to achieve a more realistic and less twitchy behavior when aggressive commands are applied (imported from [g1-manipulation-challenge](https://github.com/luckyrobots/g1-manipulation-challenge/tree/main) repo).
- computing dumped pseudoinverse of a jacobian matrix to implement the *inverse jacobian relation*.
- resetting the simulation and retrieving its information, which is used during evaluation.

## Simulation files
The core of the project are the [startup.py](simulate_python/startup.py) and [evaluation.py](simulate_python/evaluation.py) files, which both implement the simulation logic and the physical rendering of the robot.
The only difference is that the code inside [evaluation.py](simulate_python/evaluation.py) has been agumented to allow the execution of multiple consequent trials for both scenarios, whereas [startup.py](simulate_python/startup.py) can only execute once per user input, providing an alternative lighter computation process.

The execution of scripts is divided in threads that run concurrently, thus a guarding mechanism is present to ensure consistency of shared data.

### Simulation thread
This thread represents the pivotal part of the project: task logic.
Both tasks are broken into multiple steps, and each one is carried out inside this section making use of the resources provided by all the other folders and files in the project.
For both benchmark tasks, the robot spawns at the beginning of the simulation with its right arm and hand in a specific position ready to grasp the first object, then the position of its right arm and hand joints are adjusted, using inverse kinematics, to reach the final objective. The manipulation processes follow sepcific sequence of actions, each one with a separate sub-goal.
Steps of task 1:
1. grasping the red cylinder and lifiting the right arm.
2. moving the arm above the blue basket.
3. opening the right hand, thus dropping the red cylinder inside the blue basket.
4. moving the hand on top of the blue cube.
5. grasping the blue cub and stabilising the grip.
6. lifting the cube.
7. trasnfering the cube over the red basket.
8. realising the cube inside the red basket.

Steps of task 2:
1. grasping the handle of the mug without making it fall.
2. lifting the arm with the grabbed mug.
3. translating the arm upon the coaster.
4. lowering the arm and the mug on top of the coaster.
5. slowly releasing the grasp on the mug handle.

For either task, each step that involves reaching a certain target position, especially to grasp an object, is cleared only when a certain position and occasionally orientation error threshold is surpassed. These values have been tweaked accordingly, so that these conditions do not take too much time to be verified, but they still ensure that the objects are grabbed with a firm grip.
After a condition is met, and so a new step gets accessed, the target position gets updated and saved in specific variables, which are then used to apply commands to the various robot actuators.

### Physical thread
This thread updates the visual physics of the robot model, synchronizing itself with the simulation loop through a shared *viewer* of the scene. Indeed, to execute this operation, a guarding mechanism is necessary.

The code also applies a custom camera configuration make the observation of the task executions more comfortable.

### Roll and pitch reader thread
This execution unit is only present inside [evaluation.py](simulate_python/evaluation.py), since it computes data necessary for the evaluation.

More precisely, it measures at each cycle the roll and pitch of the joints previously mentioned in the section [Metrics measured](#metrics-measured).

## Configuration files
These files are used to properly link the definition of the task scenarios with the logic of the simualtion:
- [startup_config.py](simulate_python/startup_config.py) is used to configure the simulation environment for [startup.py](simulate_python/startup.py).
- [evaluation_config.py](simulate_python/evaluation_config.py) prepares the scene every time [evaluation.py](simulate_python/evaluation.py) starts a set of trials to evaluate a different task.

# Author
Daneri Gregorio - Robotics student at UNIGE.