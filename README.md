# Waterberry Farms

Waterberry Farms is an experimental framework for developing and testing algorithms that control mobile robots that investigate and monitor an environment.

It contains the following major components:

* __Environmental models__ The framework provides abstract classes and several full implementations for specific environment models. The currently implemented models assume that the information takes the form of a _scalar field_. Environments modeling other scenarios are planned for future releases. We are especially interested in environments that change at timescales sufficiently fast that to affect the information collection. We implemented two environmental models:
  * __Pollution__ - randomly occuring polution events introduce pollution into the environment at specific locations, which spread and dissipate in time
  * __Epidemic spread__ - an implementation of the SIR epidemiological model for a 2D grid.
  
* __Information models__ The model that the robot builds an maintain about the environment. This information model is updated through _observations_. The current implementation model provides a model that stores the posted observations and allow, at a later time the creation of an estimative for the _value_ and the _uncertainty_.
  * The model provides point based, disk based and Gaussian process based estimators.

* __Robot__ This class represents the physical properties of the mobile robot (ground based or aerial) such as location and altitude. We assume that the evolution of the robot happens through a series of _actions_. Several actions can happen at each timestep, such as movement, observation and communication. To accomodate this, the robot's behavior is divided into two phases: the establishment of a set of actions (pre-wired or policy driven) and the enactment of the queue of pending actions. 
  * The movement control of the robot can be performed in terms of location, velocity or acceleration, allowing for various control laws.
  * The framework provides functionality for tracking the fuel/battery use, as well as the value of information collected by the robot.

* __Policy__: A policy describes an aspect of the behavior of a robot (such as movement). The framework provides several policies that can serve as the starting point for more complex policies the user might experiment with: move to location, follow path and random waypoint.
  * Sophisticated policies, especially in scenarios with limited communication, might maintain their own information model. 

* __World__: A world for MREM is composed of an environmental model, a global information model and a collection of robots. Processing the evolution of time through the world object allows the modeling of certain scenarios. 

* __User interface__: The framework provides an ipywidget + bokeh based user interface, runnable in a Jupyter notebook. It allows the visualization of the environment, the robot location and information model. It also provides a control panel for limited interactive control of the robots.

## How to use

* MREM is implemented in Python 3.x. It relies on numpy, scipy, sklearn, ipywidgets and bokeh. 
* In order to learn the operation of various components, you might want to run the Jupyter notebooks Environment, InformationModel, Robot and Policy. These notebooks contain the respective classes as well as sample code for testing.
* The corresponding Python files ```Environment.py``` etc. contain only the implementations. 
* To see a visualized scenario that integrates all the components, use the ```Informative path planning.ipynb``` notebook. 
