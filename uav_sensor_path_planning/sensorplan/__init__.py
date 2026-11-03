"""sensorplan - information-theoretic UAV sensor path planning for multi-target tracking.

Python reproduction of Adurthi, Singla & Majji,
"Information Theoretic Sensor Path Planning: Applications to UAV Routing".
"""
from .grid import NodeGrid, enumerate_trajectories
from .models import CVTargetModel, CircularFOVSensor, CoverageMap
from .planners import seq_uav, seq_time, joint_exhaustive, gaussian_mi
from .simulate import SimConfig, run, metrics, summary, prob_ratio_2sigma

__all__ = ["NodeGrid", "enumerate_trajectories", "CVTargetModel", "CircularFOVSensor",
           "CoverageMap", "seq_uav", "seq_time", "joint_exhaustive", "gaussian_mi",
           "SimConfig", "run", "metrics", "summary", "prob_ratio_2sigma"]
