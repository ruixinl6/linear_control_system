
from BuggySimulator import *
import numpy as np
import scipy
import cmath
from scipy.ndimage import gaussian_filter1d
from util import *
from scipy import signal
import control

import matplotlib.pyplot as plt
    
class controller():
    
    def __init__(self,traj,vehicle):
        self.vehicle=vehicle
        self.traj=traj
        self.prev_vx_error=0
        self.integral_vx_error=0
        self.curv=self.compute_curvature()
        self.old_xd_error = 0
        
        self.K1 = -4    
        self.K2 = -11
        self.K3 = -850
        
    def compute_curvature(self):
        
        traj=self.traj
        
        traj_X = traj[:,0]
        traj_Y = traj[:,1]
        
        # first derivatives
        dx = np.gradient(traj_X) 
        dy = np.gradient(traj_Y)
        
        #second derivatives
        d2x = np.gradient(dx) 
        d2y = np.gradient(dy)
        
        # curvature
        cur = np.abs(d2y) / (np.sqrt(1 + dy ** 2)) ** 1.5  
        cur = gaussian_filter1d(cur, sigma=100)
# =============================================================================
#         plt.plot(np.arange(cur.shape[0]),cur)
#         plt.show()
# =============================================================================
        
        return cur
    
    def updateK(self,k1,k2,k3):
        self.K1 = k1
        self.K2 = k2
        self.K3 = k3
    
    
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

        delT = 0.05

        #reading current vehicle states
        X = vehicle.state.X
        Y = vehicle.state.Y
        xdot = vehicle.state.xd
        ydot = vehicle.state.yd
        phi = vehicle.state.phi
        phidot = vehicle.state.phid
        delta = vehicle.state.delta


        # ---------------|Lateral Controller|-------------------------
        stride = 100
        distance, idx = closest_node(X,Y,traj)
        forward_idx = min(traj.shape[0]-1,idx+stride)
        
        X_desired = traj[forward_idx][0]
        Y_desired = traj[forward_idx][1]
        phidot_desired = wrap2pi(xdot/self.curv[forward_idx]) # ?
        phi_desired = (np.arctan2(Y_desired - Y, X_desired - X))
# =============================================================================
#         print(phi_desired.type)
# =============================================================================
        e1 = Y-Y_desired
        e1_dot = ydot+xdot*(phi-phi_desired)
        e2 = (phi - phi_desired)
        e2_dot = (phidot-phidot_desired)
        e_matrix = np.array([e1,e1_dot,e2,e2_dot]).T
        
        A = np.array([[0, 1, 0, 0],[ 0, -4*Ca/(m*xdot), 4*Ca/m, 2*Ca*(lr-lf)/(m*xdot)],
               [0, 0, 0, 1],[0, 2*Ca*(lr-lf)/(Iz*xdot), 2*Ca*(lf-lr)/Iz, -2*(lf**2+lr**2)*Ca/(Iz*xdot)]])
        B = np.array([0, 2*Ca/m, 0, 2*Ca*lf/Iz]).T
        C = np.eye(4,4)
# =============================================================================
#         control.StateSpace(A, np.atleast_2d(B).T, C, np.zeros((C.shape[0],np.atleast_2d(B).T.shape[1])))
#         
#         desired_poles = np.array([-0.5,-2,-4,-100])
#         results = scipy.signal.place_poles(A,np.atleast_2d(B).T,desired_poles)
# =============================================================================
        K = np.array([self.K1,self.K2,self.K3,0]).T
        delta_desired = K@e_matrix
        deltad = delta_desired - delta
        

        #--------|Longitudinal Controller|------------------------------
        xd_desired = (traj[forward_idx][0]-traj[idx][0])/(delT*stride)
        yd_desired = (traj[forward_idx][1]-traj[idx][1])/(delT*stride)
        xd_error = 56 - (xdot)
        yd_error = ydot - yd_desired
        
        F_Kp = 5
        F_Ki = 0.02
        F_Kd = 0.1
        F = F_Kp * xd_error + F_Ki * xd_error * delT + F_Kd * (xd_error - self.old_xd_error) / delT
        # -----------------------------------------------------------------
        self.old_xd_error = xd_error

        # Communicating the control commands with the BuggySimulator
        controlinp = vehicle.command(F,deltad)
        # F: Force
        # deltad: desired rate of steering command
       

        return controlinp



