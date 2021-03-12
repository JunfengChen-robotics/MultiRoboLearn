# MultiRoboLearn
<!-- omit in toc -->
 
**This is an unified framework which can used by researchers to study multi-robot reinforcement learning in multi-robot systems.**

``MultiRoboLearn`` builds an open-source framework for multi-robot systems. This framework builds a unified setup of simulation and real-world applications. It aims to provide standard, easy-to-use simulated scenarios that can also be easily deployed to real-world multi-robot environments. Also, the framework provides researchers with a benchmark system for comparing the performance of different reinforcement learning algorithms. 

![](https://github.com/JunfengChen-robotics/MultiRoboLearn/blob/main/gif/sac.gif) 

**Test mode in both simulation and real world for multi-sac algorithm**


![](https://github.com/JunfengChen-robotics/MultiRoboLearn/blob/main/gif/dqn.gif)

**Test mode in both simulation and real world for multi-dqn algorithm**


**Note:** The whole structure of this framework and tranining process and test process can be obtained in this [**video**](https://github.com/JunfengChen-robotics/MultiRoboLearn/raw/main/MultiRoboLearn.mp4).

# Main Features
- It mainly focus on multi-robot system. It provides the community with an open-source framework for MADRL simulations and their corresponding deployment on real robots. 
- It can be used in continous and discrete action space
- It inherits [OpenAI Gym](https://gym.openai.com) interface for all the the environments, easily matching different kinds of algorithms
- It has been successfully deployed to train two kinds of MADRL algorithms to solve two different tasks in simulation as well as real-world tasks without any further training in  the real world

**Warning:** This framework is ongoing developing,  and thus some changes might slight changes occur. Please don't mind modification!

# Requirements

Our experimental platform can be well installed on both Ubuntu 16.04 and 18.04, in the future, our framework will support higher veirsion of Ubuntu. At the same time, this framework is tested in Python2.7 and Python3.5(higher Python3 versions are supported). 


Our framework has some packages and libraries as follows:
- **ROS**(Kinetic @ Melodic). For melodic version, we also bulid some dependencies:  **ros-melodic-pid**, **ros-melodic-joy**
-  **Python2.7** & **Python3.6**
-  **Gym**
-  **Virtualenv** (Due to ROS only supports Python2.x, however, multiagent reinforcement learning algorithms must be trained in Python3.x, so we build virtual environment.): so we need to build some dependencies:  **python-pip**, **numpy**, **python-catkin-tools**, **python3-dev**, **python3-pip**, **python3-yaml**, **rospkg**, **catkin_pkg**, **geometry**, **geometry2**, **empy** 
-   **Tensorflow**
-   **Pytorch**
-   **Keras**
-   **Pandas**
-   **seaborn**


# Code Structure
- `./scenarios/gym_construct`: folder where some different kinds of taks scenarios were stored.
- `./MultiRoboLearn/src`: folder where the core code of this framework were stored and users can set different MADRL environment(including: reward, state, action)
   1) `./MultiRoboLearn/src/task_envs`: this folder mainly can be used to set different MADRL environment according to specific task.
   2) `./MultiRoboLearn/src/robot_envs`: this folder mainly can be extended into diffrent kinds of robots combination supported ROS
- `./algorithms/algorithms_example/`: this mainly contains of diffent algorithms,users can develop their desired algorithms in this folder.

# Installation & How to use?
There are two ways to install this framework:
   1. using a virtual environment and pip
   2. how to use through code structure

## Virtualenv @ pip
First download the **pip** Python package manager and create a virtual environment for Python.

On Ubuntu, you can install **pip** and **virtualenv** by typing in the terminal:

-In Python 3.6:
```
sudo apt install python3-pip
sudo pip install virtualenv
```
You can then create the virtual environment by typing:

```
virtualenv -p /usr/bin/python3.6 <virtualenv_name>
activate the virtual environment
source <virtualenv_name>/bin/activate
```
To deactivate the virtual environment, just type:
```
deactivate
```

## How to use?
- Firstly, users can download the code by typing:
```
 git clone https://github.com/JunfengChen-robotics/MultiRoboLearn.git
 ```
 
- Secondly, users can make virtual enviroment through above command, just type:

```
cd ~/<virtualenv_name>/bin
source activate
```

- Thirdly, users can load taks sceniaors through gazebo, by typing:
```
cd  ~/scenarios/gym_construct
roslaunch multiagent_main.launch
```

- Fourth, users can load algorthims to tranin, just type:
```
cd ~/algorithms/algorithms_example/launch/turtle2_openai_ros_example
roslaunch start_multiagent_training.launch
```

# Contributing
- New task environments and new robots combination implementations are welcome!
- If you encounter troubles running MultiRoboLearn or if you have questions please submit a new issue.

