import casadi as cs 

#def L(t,x,u):
#    return 
#
def E(t0,tf,x0,xf):
    return tf

def e(t0,tf,x0,xf):
    yf,zf=xf[0],xf[1]
    return cs.vertcat(x0-0,yf-1000,zf-1)

def f(t,x,u):
    g=9.8
    y,z,v=x[0],x[1],x[2]
    theta=u
    xd=cs.vertcat(v*cs.sin(theta),v*cs.cos(theta),g*cs.cos(theta))
    return xd

#def h(t,x,u):
#    return


