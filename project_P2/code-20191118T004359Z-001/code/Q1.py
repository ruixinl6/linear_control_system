import numpy as np
import matplotlib.pyplot as plt
import control

Ca = 15000
Iz = 3344
m = 2000
lr = 1.7
lf = 1.1

# Calculating A
x_dot = 2
A1 = np.array([[0, 1, 0, 0],[ 0, -4*Ca/(m*x_dot), 4*Ca/m, 2*Ca*(lr-lf)/(m*x_dot)],
               [0, 0, 0, 1],[0, 2*Ca*(lr-lf)/(Iz*x_dot), 2*Ca*(lf-lr)/Iz, -2*(lf**2+lr**2)*Ca/(Iz*x_dot)]])
x_dot = 5
A2 = np.array([[0, 1, 0, 0],[ 0, -4*Ca/(m*x_dot), 4*Ca/m, 2*Ca*(lr-lf)/(m*x_dot)],
               [0, 0, 0, 1],[0, 2*Ca*(lr-lf)/(Iz*x_dot), 2*Ca*(lf-lr)/Iz, -2*(lf**2+lr**2)*Ca/(Iz*x_dot)]])
x_dot = 8
A3 = np.array([[0, 1, 0, 0],[ 0, -4*Ca/(m*x_dot), 4*Ca/m, 2*Ca*(lr-lf)/(m*x_dot)],
               [0, 0, 0, 1],[0, 2*Ca*(lr-lf)/(Iz*x_dot), 2*Ca*(lf-lr)/Iz, -2*(lf**2+lr**2)*Ca/(Iz*x_dot)]])

# B & C
B = np.array([0, 2*Ca/m, 0, 2*Ca*lf/Iz]).T
C = np.eye(4,4)

# Calculating Q & P
Q1 = np.array([C,C@A1,C@A1@A1,C@A1@A1@A1]).reshape((16,4))
Q2 = np.array([C,C@A2,C@A2@A2,C@A2@A2@A2]).reshape((16,4))
Q3 = np.array([C,C@A3,C@A3@A3,C@A3@A3@A3]).reshape((16,4))

P1 = np.array([B,A1@B,A1@A1@B,A1@A1@A1@B]).T
P2 = np.array([B,A2@B,A2@A2@B,A2@A2@A2@B]).T
P3 = np.array([B,A3@B,A3@A3@B,A3@A3@A3@B]).T

# Calculating rank
Qrank1 = np.linalg.matrix_rank(Q1)
Qrank2 = np.linalg.matrix_rank(Q2)
Qrank3 = np.linalg.matrix_rank(Q3)

Prank1 = np.linalg.matrix_rank(P1)
Prank2 = np.linalg.matrix_rank(P2)
Prank3 = np.linalg.matrix_rank(P3)

n = 4
results = []
poles_total = []
for Vx in range(40):
    x_dot = Vx + 1
    A = np.array([[0, 1, 0, 0],[ 0, -4*Ca/(m*x_dot), 4*Ca/m, 2*Ca*(lr-lf)/(m*x_dot)],
               [0, 0, 0, 1],[0, 2*Ca*(lr-lf)/(Iz*x_dot), 2*Ca*(lf-lr)/Iz, -2*(lf**2+lr**2)*Ca/(Iz*x_dot)]])
    P = np.array([B,A@B,A@A@B,A@A@A@B]).T
    _,s,_ = np.linalg.svd(P)
    result_log = np.log10((s[0]/s[-1]))
    results.append(result_log)
    
    sys = control.StateSpace(A, np.atleast_2d(B).T, C, np.zeros((C.shape[0],np.atleast_2d(B).T.shape[1])))
    poles = np.real(control.pole(sys))
    poles_total.append(poles)
    
plt.plot(np.arange(40)+1,results)
plt.show()

for i in range(1,5):
    plt.subplot(2,2,i)
    poles = np.array([poles_total[j][i-1]for j in range(40)])
    plt.plot(np.arange(40)+1,poles)
plt.show()