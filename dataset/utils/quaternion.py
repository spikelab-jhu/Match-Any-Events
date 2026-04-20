import numpy as np
import math

class Quaternion:
    def __init__(self, scalar=1, vec=[0,0,0]): 
        self.q = np.array([scalar, 0., 0., 0.])
        self.q[1:4] = vec

    def normalize(self):
        self.q = self.q/np.linalg.norm(self.q)

    def scalar(self):
        return self.q[0]

    def vec(self):
        return self.q[1:4]

    def axis_angle(self):
        theta = 2*np.arccos(self.scalar())
        vec = self.vec()
        if (np.linalg.norm(vec) == 0):
            return np.zeros(3)
        vec = vec/np.linalg.norm(vec)
        return vec*theta

    def euler_angles(self):
        phi = math.atan2(2*(self.q[0]*self.q[1]+self.q[2]*self.q[3]), \
                1 - 2*(self.q[1]**2 + self.q[2]**2))
        theta = math.asin(2*(self.q[0]*self.q[2] - self.q[3]*self.q[1]))
        psi = math.atan2(2*(self.q[0]*self.q[3]+self.q[1]*self.q[2]), \
                1 - 2*(self.q[2]**2 + self.q[3]**2))
        return np.array([phi, theta, psi])
    
    def to_rotm(self):
        w = self.q[0]
        x = self.q[1]
        y = self.q[2]
        z = self.q[3]

        return np.array(
            [
                [1 - 2*(y**2 + z**2),   2*(x*y - w*z),          2*(x*z + w*y)],
                [2*(x*y + w*z),         1- 2*(x**2 + z**2),    2*(y*z - w*x)],
                [2*(x*z - w*y),         2*(y*z + w*x),          1- 2*(x**2 + y**2)]
            ]
        )

    def from_axis_angle(self, a):
        angle = np.linalg.norm(a)
        if angle != 0:
            axis = a/angle
        else:
            axis = np.array([1,0,0])
        if np.isnan(axis[0] * axis[1] * axis[2]):
            axis = np.array([0,0,0])
        self.q[0] = math.cos(angle/2)
        self.q[1:4] = axis*math.sin(angle/2)

    def from_rotm(self, R):
        theta = math.acos((np.trace(R)-1)/2)
        if math.sin(theta)==0:
            omega = np.array([1, 0, 0])
        else:
            omega_hat = (R - np.transpose(R))/(2*math.sin(theta))
            omega = np.array([omega_hat[2,1], -omega_hat[2,0], omega_hat[1,0]])
        self.q[0] = math.cos(theta/2)
        self.q[1:4] = omega*math.sin(theta/2)
        self.normalize()

    def inv(self):
        q_inv = Quaternion(self.scalar(), -self.vec())
        q_inv.normalize()
        return q_inv

    def __mul__(self, other):
        t0 = self.q[0]*other.q[0] - \
             self.q[1]*other.q[1] - \
             self.q[2]*other.q[2] - \
             self.q[3]*other.q[3]
        t1 = self.q[0]*other.q[1] + \
             self.q[1]*other.q[0] + \
             self.q[2]*other.q[3] - \
             self.q[3]*other.q[2]
        t2 = self.q[0]*other.q[2] - \
             self.q[1]*other.q[3] + \
             self.q[2]*other.q[0] + \
             self.q[3]*other.q[1]
        t3 = self.q[0]*other.q[3] + \
             self.q[1]*other.q[2] - \
             self.q[2]*other.q[1] + \
             self.q[3]*other.q[0]
        retval = Quaternion(t0, [t1, t2, t3])
        return retval

    def __str__(self):
        return str(self.scalar()) + ', ' + str(self.vec())
