from BuggySimulator import *
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from util import *
import scipy.signal
import math

class controller():
    
    def __init__(self,traj,vehicle):
        self.vehicle=vehicle
        self.traj=traj
        # Additional member variable
        self.vErroror_p = 0
        self.xdot_p = 0;
        self.vIndex = 0

    def control_update(self):
        traj = self.traj
        vehicle = self.vehicle
        vErroror_p = self.vErroror_p
        xdot_p = self.xdot_p
        vIndex = self.vIndex

        lr = vehicle.lr
        lf = vehicle.lf
        Ca = vehicle.Ca
        Iz = vehicle.Iz
        vFavor = vehicle.f
        m = vehicle.m
        g = vehicle.g

        delT = 0.05  # The simulator runs at a constant fps.

        # reading current vehicle states
        X = vehicle.state.X
        Y = vehicle.state.Y
        xdot = vehicle.state.xd
        ydot = vehicle.state.yd
        phi = vehicle.state.phi
        phidot = vehicle.state.phid
        delta = vehicle.state.delta

        # ---------------|Lateral Controller|-------------------------
        """
        Design your lateral controller here. 
        """
        vIntegral = 0
        vKp = 500
        vKi = 0
        vKd = 0
        vbias = 0

        vError, mid = closest_node(X, Y, traj)

        vIndex = mid + 100
        if (vIndex > 8202):
            vIndex = 8202

        vNumber = (traj[vIndex][1] - Y)
        vDenom_ = (traj[vIndex][0] - X)

        vErroror = math.atan2(vNumber, vDenom_)
        vErroror = wrap2pi(vErroror - delta - phi)
        vIntegral = vErroror * delT
        vDev = (vErroror - vErroror_p) / delT
        deltad = vKp * vErroror + vKi * vIntegral + vKd * vDev

        # ---------------|Longitudinalal Controller|-------------------------
        """
        Desing your longitudinal controller here. 
        """

        vKp = 6
        vKi = 0.05
        vKd = 0.01
        vbias = 0
        vErroror = 54 - xdot
        vIntegral = vErroror * delT
        vDev = (vErroror - vErroror_p) / delT
        vFavor = vKp * vErroror + vKi * vIntegral + vKd * vDev
        # vFavor represents Force
        # deltad represents desired rate of steering command

        Vx = xdot

        # Communicating the control commands with the BuggySimulator
        controlinp = vehicle.command(vFavor,deltad)

        return controlinp,Vx



