from __future__ import division, print_function
from BuggySimulator import *

import scipy.linalg
import numpy as np
import scipy 
import cmath 
from scipy.ndimage import gaussian_filter1d 
from util import *
from scipy import signal


class controller():

    def __init__(self,traj, vehicle):

        self.vehicle=vehicle
        self.traj=traj
        self.prev_vx_error=0
        self.integral_vx_error=0
        self.curv=self.compute_curvature()
        self.K = 0
        self.curv_normal = self.curv/(max(self.curv)-min(self.curv))
        
        # Kalman Filter 1
        self.A = 0
        self.B = 0
        self.C = 0
        self.D = 0
        self.Ad = 0
        self.Bd = 0
        self.P = 0
        self.error_kalman = 0
        self.W = np.array([[3,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.V = np.array([[16,0,0,0],[0,3,0,0],[0,0,1.5,0],[0,0,0,0.55]])
        
        # Kalman Filter 2
        self.A2 = 0
        self.B2 = 0
        self.C2 = 0
        self.D2 = 0
        self.A2d = 0
        self.B2d = 0
        self.P2 = 0
        self.error_kalman2 = 0
        self.W2 = np.eye(2)
        self.V2 = np.array([[1,0],[0,0.5]])
        
        
    def compute_curvature(self): 
        """ 
        Function to compute and return the curvature of trajectory. 
        """
        sigma_gaus = 10 
        traj=self.traj 
        xp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,0], sigma=sigma_gaus,order=1)
        xpp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,0], sigma=sigma_gaus,order=2) 
        yp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,1], sigma=sigma_gaus,order=1) 
        ypp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,1], sigma=sigma_gaus,order=2) 
        curv=np.zeros(len(traj))
        for i in range(len(xp)): 
            curv[i] = (xp[i]*ypp[i] - yp[i]*xpp[i])/(xp[i]**2 + yp[i]**2)**1.5
            
        return curv
    
    def dlqr(self,A,B,Q,R):

        S = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
 
        #compute the LQR gain
        K = -np.matrix(scipy.linalg.inv(B.T*S*B+R)*(B.T*S*A))
 
        eigVals, eigVecs = scipy.linalg.eig(A+B*K)
 
        return K, eigVals
    
    def kalman(self,u,y):
        A = self.Ad
        B = self.Bd
        C = self.C
        P_k_1_k_1 = self.P
        x_k_1_k_1 = self.error_kalman
        
        x_k_k_1 = A @ x_k_1_k_1 + B @ np.atleast_2d(u)
        P_k_k_1 = A @ P_k_1_k_1 @ A.T + self.W
        L = P_k_k_1 @ C.T @ np.linalg.inv(C @ P_k_k_1 @ C.T + self.V)
        x_k_k = x_k_k_1 + L @ (y - C @ x_k_k_1)
        P_k_k = (np.eye(4) - L @ C) @ P_k_k_1
        
        return x_k_k, P_k_k
    
    def kalman2(self,u,y):
        A = self.A2d
        B = self.B2d
        C = self.C2
        P_k_1_k_1 = self.P2
        x_k_1_k_1 = self.error_kalman2
        x_k_k_1 = A @ x_k_1_k_1 + B @ np.atleast_2d(u)
        P_k_k_1 = A @ P_k_1_k_1 @ A.T + self.W2
        L = P_k_k_1 @ C.T @ np.linalg.inv(C @ P_k_k_1 @ C.T + self.V2)
        x_k_k = x_k_k_1 + L @ (y - C @ x_k_k_1)
        P_k_k = (np.eye(2) - L @ C) @ P_k_k_1
        
        return x_k_k, P_k_k

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
        X = vehicle.observation.X
        Y = vehicle.observation.Y 
        xdot = vehicle.observation.xd
        ydot = vehicle.observation.yd 
        phi = vehicle.observation.phi
        phidot = vehicle.observation.phid
        delta = vehicle.observation.delta


        mindist, index = closest_node(X, Y, traj)
        if index<len(traj)-100: 
            idx_fwd = 100 
        else: 
            idx_fwd = len(traj)-index-1
        
        
        Vx = 7
        # Computing the curvature of trajectory 
        curv=self.curv
        # ---------------|Lateral Controller|-------------------------
        if self.prev_vx_error==0: # the very first loop
            # For lateral
            self.error_kalman = np.zeros((4,1))
            self.P = np.zeros((4,4))
            self.A = np.array([[0,1,0,0],[0,-4*Ca/(m*Vx),4*Ca/m,2*Ca*(lr-lf)/(m*Vx)],[0,0,0,1], [0,2*Ca*(lr-lf)/(Iz*Vx),2*Ca*(lf-lr)/Iz,-2*Ca*(lr*lr+lf*lf)/(Iz*Vx)]]) 
            self.B = np.array([[0],[2*Ca/m],[0],[2*Ca*lf/Iz]])
            self.C = np.identity(4)
            self.D = [[0],[0],[0],[0]]
            
            #State space system (continuous) 
            syscont = signal.StateSpace(self.A,self.B,self.C,self.D)
            #Discretizing state space system 
            sysdisc = syscont.to_discrete(delT) 
            self.Ad = sysdisc.A 
            self.Bd = sysdisc.B
            #Computing the Feedback control matrix by pole placement
            # poles=np.array([-1, -0.5, 0.1, 1]) 
            Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,10,0],[0,0,0,1]])
            R = 1
            K, eigVals = self.dlqr(self.Ad,self.Bd,Q,R)
            self.K = -K
            
            # For longitudinal
            self.error_kalman2 = np.zeros((2,1))
            self.P2 = np.zeros((2,2))
            self.A2 = np.array([[0,1],[0,0]])
            self.B2 = np.array([[0],[1/m]])
            self.C2 = np.eye(2)
            self.D2 = [[0],[0]]
            syscont2 = signal.StateSpace(self.A2,self.B2,self.C2,self.D2)
            sysdisc2 = syscont2.to_discrete(delT) 
            self.A2d = sysdisc2.A 
            self.B2d = sysdisc2.B
        
        phides = np.arctan2((traj[index+idx_fwd][1]-Y),(traj[index+idx_fwd][0]-X)) 
        phidesdot = xdot*curv[index+idx_fwd]
        e = np.zeros(4)
        
        #Ref p34 Vehicle Dynamics and Control by Rajesh Rajamani 
        e[0] = (Y - traj[index+idx_fwd][1])*np.cos(phides) - (X - traj[index+ idx_fwd][0])*np.sin(phides) 
        e[2] = wrap2pi(phi - phides) 
        e[1] = ydot + xdot*e[2] 
        e[3] = phidot - phidesdot
        error_measure = np.matrix(e).T
        
        deltades = float(-self.K*(self.error_kalman)) 
        #deltades = float(-K[0,:]*np.transpose(error)) 
        self.error_kalman, self.P = self.kalman(deltades,error_measure)
        # print(deltades) 
        deltad = (deltades - delta)/0.05

        #--------|Longitudinal Controller|------------------------------
        kp=400 
        kd=300 
        ki=-0.1
        # Computing the errors 
        vx_error=Vx-xdot 
        self.integral_vx_error+=vx_error 
        derivative_error=vx_error-self.prev_vx_error 
        F=kp*vx_error + ki*self.integral_vx_error*delT + kd*derivative_error/delT
        self.error_kalman2, self.P2 = self.kalman2(F,vx_error)
        # -----------------------------------------------------------------
        
        # Communicating the control commands with the BuggySimulator
        controlinp = vehicle.command(F,deltad)
        # F: Force
        # deltad: desired rate of steering command
        self.prev_vx_error=vx_error
       

        return controlinp
