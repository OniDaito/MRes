import tensorflow as tf
import math

# phi is x, psi is y

phi0 = 0
psi0 = 0

phi1 = -math.pi / 1.8 
psi1 = math.pi / 2

# Starting data
td = [[[ math.sin(phi0), math.cos(phi0), math.sin(psi0), math.cos(psi0)], 
  [ math.sin(phi1), math.cos(phi1), math.sin(psi1), math.cos(psi1) ]]]

(pre_phi,pre_psi)= tf.split(td,2,2)

(sin_phi,cos_phi)= tf.split(pre_phi,2,2)
(sin_psi,cos_psi)= tf.split(pre_psi,2,2)

phi = tf.atan2(sin_phi, cos_phi)
psi = tf.atan2(sin_psi, cos_psi)



def error_rama(x,y):
  # Take an estimate of the ramachandran plot

  a = tf.maximum(tf.sin(x*1.8-45.9+y*0.26)*1.2 + tf.cos(y*1.8)*0.3 - (0.5 * x) -1.4 + (y * 0.1), 0)
  b = tf.maximum(tf.sin(y-0.35 + x*0.85)*0.8 + tf.cos(x-1.0) -1.5, 0)
  
  return 1.0 - tf.minimum(1.0, a+b)


sess = tf.Session()
with sess.as_default():
  err = error_rama(phi, psi)

  print(err.eval())
  #print(psi.eval())

