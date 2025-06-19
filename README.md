# 🛣️ Frenet Corridor Planner (FCP)

**An Optimal Local Path Planning Framework for Autonomous Driving**
> 📌 **Code coming soon!**  
> ⭐ Star this repo to get notified as soon as it's released.
---

## 📘 Overview

The **Frenet Corridor Planner (FCP)** is an optimization-based local path planner designed for real-time trajectory generation in autonomous vehicles. Leveraging the Frenet reference frame and a space-domain bicycle model, FCP defines safe, smooth, drivable corridors around static and dynamic obstacles—then optimizes paths for smoothness, clearance, and risk reduction, before integrating with a speed planner.

- Noise-resilient planning
![Dynamic obstacles with perception noises](01-dyn-obs.gif)
- Smooth update with occlusions
![Planning with occlusions](04-occlusion.gif)
- Dynamic and efficient updates
![Crowd navgiation](03-crowd-nav.gif)

---

## 🧩 Key Features

- **Frenet-based obstacle modeling**  
  Vehicles are represented as safety-augmented bounding boxes and pedestrians as convex hulls in Frenet space.

- **Adaptive drivable corridors**  
  Dynamically selects lateral deviations to create safe passage zones around detected obstacles.

- **Optimization with bicycle kinematics**  
  A modified space-domain bicycle model ensures kinematic feasibility, smoothness, and safety during optimization.

- **Decoupled path-speed planning**  
  Separates path and speed stages for efficient computation: FCP handles geometry, while an existing speed planner completes trajectory generation.

---

## 📊 Evaluation

- Tested in **simulation** and on **physical hardware** with realistic sensor noise and dynamic obstacle scenarios.
- Demonstrated **smooth, safe** maneuvers around static and moving objects with kinematic consistency, even in tight spaces.

---

## 📚 Citation

If you find this work useful, please cite:
```
@article{tariq2025frenet,
  title = {Frenet Corridor Planner: An Optimal Local Path Planning Framework for Autonomous Driving},
  author = {Tariq, Faizan M. and Yeh, Zheng-Hang and Singh, Avinash and Isele, David and Bae, Sangjae},
  journal = {arXiv preprint arXiv:2505.03695},
  year = {2025}
}
```



