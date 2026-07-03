# 🚁 Drone Swarm Navigation: A Numerical Methods Approach

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Vectors-lightblue.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3D_Rendering-orange.svg)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-brightgreen.svg)

## 📌 Project Overview
The rapid advancement of autonomous aviation introduces complex engineering challenges, particularly in multi-agent navigation. This project presents a **3D computational model** that simulates the autonomous, collision-free navigation of a drone swarm in an urban environment. 

Rather than relying on simple static distance checks or pre-built physics engines, this simulation acts as a virtual LiDAR system that mathematically predicts collisions before they occur. The entire physics and flight dynamics engine was built from scratch to bridge theoretical numerical mathematics with practical aerospace engineering.

## 🧮 Numerical Methods Implemented
This project heavily relies on numerical algorithms to handle different stages of flight:

* **4th Order Runge-Kutta (RK4):** Integrated to solve the high-dimensional system of Ordinary Differential Equations (ODEs) that dictate the drone's kinematic state and flight dynamics.
* **Newton-Raphson Method:** Deployed as the core predictive root-finding algorithm to foresee and actively evade static building collisions (Virtual LiDAR).
* **Cubic Spline Interpolation:** Used as a post-flight data processing tool to mathematically smooth the drone's discrete point-by-point path history into a continuous flight curve.
* **Floating-Point Error Analysis:** Conducted to track and visualize the accumulation of machine round-off errors over continuous simulation time.
* **Linear Systems & LU Decomposition:** Utilized to efficiently solve aerodynamic drag matrices and calculate environmental wind disturbances.
* **Numerical Optimization:** Applied to calculate the most energy-efficient cruise velocity prior to takeoff.

## ✨ Core Features
* **Real-Time 3D Rendering:** Custom procedural geometry calculation for hexacopter bodies and spinning propellers using trigonometric transformations.
* **Interactive Control Panel:** Modern UI built with `CustomTkinter` to manage swarm size, toggle trajectories, and switch between dynamic camera modes.
* **Mathematical Console:** A live-updating terminal that displays RK4 steps, Newton-Raphson collision alerts, and real-time numerical stability checks.
* **Post-Flight Academic Diagnostics:** Automatically generates Matplotlib figures to analyze path interpolation and machine precision loss once the drone reaches its target.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/drone-numerical-simulation.git](https://github.com/yourusername/drone-numerical-simulation.git)
   cd drone-numerical-simulation
