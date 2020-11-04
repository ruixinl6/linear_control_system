
from BuggySimulator import *
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from util import *
import scipy.signal
    
class controller():
    
    def __init__(self,traj,vehicle):
        self.vehicle=vehicle
        self.traj=traj

        # Add additional member variables according to your need here.
        self.old_delta_error = 0
        self.old_xd_error = 0
        self.count = 0

    def control_update(self):

        traj=self.traj
        vehicle=self.vehicle 
        
        lr = vehicle.lr
        lf = vehicle.lf
        Ca = vehicle.Ca
        Iz = vehicle.Iz
        f = vehicle.f
        m = vehicle.m
        g = vehicle.g

        delT = 0.05   # The simulator runs at a constant fps.

        #reading current vehicle states
        X = vehicle.state.X
        Y = vehicle.state.Y
        xdot = vehicle.state.xd
        ydot = vehicle.state.yd
        phi = vehicle.state.phi
        phidot = vehicle.state.phid
        delta = vehicle.state.delta
        
        stride = 100
        distance, idx = closest_node(X,Y,traj)
        forward_idx = min(traj.shape[0]-1,idx+stride)

        # ---------------|Lateral Controller|-------------------------
        # find the closest point in the trajectory
        X_desired = traj[forward_idx][0]
        Y_desired = traj[forward_idx][1]
        delta_error = np.arctan2(Y_desired - Y, X_desired - X)
        delta_error = wrap2pi(delta_error - phi - delta)
        
        deltad_Kp = 500
        deltad_Ki = 0.02
        deltad_Kd = 0.3
        deltad = deltad_Kp * delta_error + deltad_Ki * delta_error * delT + deltad_Kd * (delta_error - self.old_delta_error) / delT

        #--------|Longitudinal Controller|------------------------------
        xd_desired = (traj[forward_idx][0]-traj[idx][0])/(delT*stride)
        yd_desired = (traj[forward_idx][1]-traj[idx][1])/(delT*stride)
        xd_error = 56 - (xdot)
        yd_error = ydot - yd_desired
        
        F_Kp = 5
        F_Ki = 0.02
        F_Kd = 0.1
        F = F_Kp * xd_error + F_Ki * xd_error * delT + F_Kd * (xd_error - self.old_xd_error) / delT


        self.old_delta_error = delta_error
        self.old_xd_error = xd_error
        
        self.count = self.count + 1
        # Communicating the control commands with the BuggySimulator
        controlinp = vehicle.command(F,deltad)
        Vx = xdot
        # F: Force
        # deltad: desired rate of steering command

        return controlinp,Vx



