"""
geom.py - various math geometry functions
and graft on our own loop
author : Benjamin Blundell
email : me@benjamin.computer

"""

import math

EPSILON = 4.37114e-05

class Quat() :
  def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
    self.x = x
    self.y = y
    self.z = z
    self.w = w
  
  def from_axis_angle(self, a, r):
    self.w = math.cos(r / 2)
    v = (a[0],a[1],a[2]) 
    v = norm(v)
    v = mults(v,math.sin(r/2))
    self.x = v[0]
    self.y = v[1]
    self.z = v[2]

  def get_conjugate(self):
    return Quat(-self.x, -self.y, -self.z)

  def get_qmult(self,q):
    w = -self.x * q.x - self.y * q.y - self.z * q.z + self.w * q.w 
    x = self.x * q.w + self.y * q.z - self.z * q.y + self.w * q.x 
    y = -self.x * q.z + self.y * q.w + self.z * q.x + self.w * q.y 
    z = self.x * q.y - self.y * q.x + self.z * q.w + self.w * q.z 
    
    return Quat(x,y,z,w)

  def length(self):
    return math.sqrt(self.x * self.x + 
        self.y * self.y +  
        self.z * self.z +
        self.w * self.w) 
  
  def normalize(self):
    l = 1.0 / self.length() 
    self.w *= l 
    self.x *= l 
    self.y *= l 
    self.z *= l

  def from_to(self, f, t):
    axis = cross(f,t)
    self.w = dot(f,t) 
    self.x = axis[0]
    self.y = axis[1]
    self.z = axis[2]
    self.normalize()
    self.w += 1.0 

    #if self.w <= EPSILON: 
    #  if f[2] * f[2] > f[0] * f[0]:
    #    self.w = 0.0
    #    self.x = 0 
    #    self.y = f[2] 
    #    self.z = -f[1] 
    #  else:
    #    self.w = 0.0
    #    self.x = f[1]
    #    self.y = -f[0] 
    #    self.z = 0.0 
    self.normalize()

  def get_matrix(self):
    xs = self.x + self.x
    ys = self.y + self.y
    zs = self.z + self.z
    wx = self.w * xs
    wy = self.w * ys
    wz = self.w * zs
    xx = self.x * xs
    xy = self.x * ys
    xz = self.x * zs
    yy = self.y * ys
    yz = self.y * zs
    zz = self.z * zs

    t = [ 1.0 - (yy+zz), 
        xy + wz, 
        xz - wy, 
        xy - wz, 
        1.0 - ( xx + zz ), 
        yz + wx, 
        xz + wy, 
        yz - wx, 
        1.0 - ( xx + yy )
    ]
    
    # Row major for numpy
    m = [ [t[0], t[3], t[6]] , [t[1], t[4], t[7]] , [t[2], t[5], t[8]] ]
    return m

def cross(u,v):
  x = (u[1]*v[2]) - (u[2]*v[1])
  y = (u[2]*v[0]) - (u[0]*v[2])
  z = (u[0]*v[1]) - (u[1]*v[0])
  return (x,y,z)

def sub(u,v):
  return (u[0] - v[0], u[1] - v[1], u[2] - v[2])

def norm(u):
  l = 1.0 / length(u)
  return (u[0] *l, u[1] * l, u[2] * l)

def add(u,v):
  return (u[0] + v[0], u[1] + v[1], u[2] + v[2])

def dot(u,v):
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def mults(u,s):
  return (u[0] * s, u[1] * s, u[2] * s)

def length(u) :
  return math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

def mv_mult(m,v):
  # Column major matrix format
  return (v[0] * m[0] + v[0] * m[1] + v[0] * m[2],
          v[1] * m[3] + v[1] * m[4] + v[1] * m[5],
          v[2] * m[6] + v[2] * m[7] + v[2] * m[8])

def transpose(m):
  return [m[0],m[3],m[6], m[1], m[4], m[7], m[2], m[5], m[8]] 

def rot_mat(v, a):
  s = math.sin(a)
  c = math.cos(a)
  v = norm(v)

  m = [[0,0,0], [0,0,0], [0,0,0]]

  # Row major for numpy
  m[0][0] = v[0] * v[0] * (1.0-c) + c 
  m[1][0] = v[0] * v[1] * (1.0-c) + v[2] * s
  m[2][0] = v[0] * v[2] * (1.0-c) - v[1] * s

  m[0][1] = v[0] * v[1] * (1.0-c) - v[2] * s 
  m[1][1] = v[1] * v[1] * (1.0-c) + c
  m[2][1] = v[1] * v[2] * (1.0-c) + v[0] * s

  m[0][2] = v[0] * v[2] * (1.0-c) + v[1] * s
  m[1][2] = v[1] * v[2] * (1.0-c) - v[0] * s
  m[2][2] = v[2] * v[2] * (1.0-c) + c

  return m


if __name__ == "__main__":
  qt = Quat()
  qt.from_axis_angle((0,1,0), -math.pi / 4)
  tt = norm(np.matrix(qt.get_matrix()).dot( ( 1,0,0 ) ).A1)
  print("TT",tt)


