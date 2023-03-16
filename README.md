# About The Project
This is a framework for creating simple traffic environment simulations. Here is an example simulation (red is the agent):

![Example](example.gif)
<!--- (<img src="example.gif"  width="1200" height="300">) --->

# Requirements
numpy, gymnasium, cv2

# Usage
Overwrite **BaseSpawner** and **BaseTrafficEnvironment** classes in **TrafficEnvironments\base_models.py** to create your own custom environments. Read the documentation in **BaseTrafficEnvironment** to see how. You can also see the example **TrafficEnvironments\traffic_env.py**.


# Contact
kaan.buyukdemirci@ug.bilkent.edu.tr, kaanbuyukdemirci2023@gmail.com

# License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# TODO
* Better documentation
* Parallel computation
* Profiling
* Using continuous y-axis locations instead of digital lanes
* A separate log window