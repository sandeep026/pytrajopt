import numpy as np
import matplotlib.pyplot as plt
import casadi as cs

class opt_grid:

    def __init__(self,N):
        self.t0=0
        self.tf=1
        self.T=self.tf-self.t0
        self.N=N
        self.dt=self.T/self.N
        self.t=np.linspace(self.t0,self.tf,self.N+1)
        self.Nx=2
        self.Nu=1


class opt_problem:
    
    def __init__(self,Nx,Nu,N):
        self.opti=cs.Opti()
        self.X=self.opti.variable(Nx,N+1)
        self.U=self.opti.variable(Nu,N)

    def OPT(self,O,RD,RBC): 
        self.opti.minimize(O)
        self.opti.subject_to(RD==0)
        self.opti.subject_to(RBC==0)
        self.opti.solver("ipopt")
        self.sol=self.opti.solve()

    def PLOT(self,t,N):
            plt.style.use("pub_qual.mplstyle")
            plt.plot(t[0:N],(6-12*t[0:N]),t[0:N],self.sol.value(self.U))
            plt.xlabel("time [s]")
            plt.ylabel("$u$")
            plt.grid()
            plt.figure()
            plt.plot(t[0:N],np.abs((6-12*t[0:N])-self.sol.value(self.U))) 
            plt.xlabel("time [s]")
            plt.ylabel("|control input error|")
            plt.grid()
            plt.show()  
            print(np.linalg.norm(self.sol.value(self.U)-(6-12*t[0:N]),2))    
            print(np.linalg.norm(self.sol.value(self.X[0,:])-(3*t**2-2*t**3),2))

    def test(self,t,N):
            print(np.linalg.norm(self.sol.value(self.U)-(6-12*t[0:N]),2))    
            print(np.linalg.norm(self.sol.value(self.X[0,:])-(3*t**2-2*t**3),2))        



class opt_obj:

    def __init__(self,Nx,Nu,N,dt,X,U):
        self.obj=self.objective(Nx,Nu,N,dt,X,U)

    def L(self,x,u):
        L=u**2
        return L
    
    def objective(self,Nx,Nu,N,dt,X,U):

        x=cs.SX.sym("x",Nx,1)
        u=cs.SX.sym("u",Nu,1)
        F=self.L(x,u)
        F=cs.Function("F",[x,u],[F])
        F=F.map(N)
        F=F(X[:,1:N+1],U)
        obj=cs.dot(dt*cs.MX.ones(1,N),F)  # add a seperate method of numerical integration (riemann sum, midpoint/rectangle,trapezoidal, simpson,LGL)
        return obj


class opt_dynamics:


    def __init__(self,N,dt,X,U):
        self.res=self.res(N,dt,X,U)

       
    def xdot(self,x,u):
        xd1=x[1,:]
        xd2=u
        return cs.vertcat(xd1, xd2)    
    
    def res(self,N,dt,X,U):
        
        for i in range(N):
            temp=X[:,i+1]-X[:,i]-self.slope_approx(X[:,i],X[:,i],U[:,i],dt,"Her") 
            if i==0:
                r=temp
            else:    
                r=cs.vertcat(r,temp)

        return r        
    
    # constant u over the interval
    # one step integration on each interval
    def slope_approx(self,x1,x2,u,dt,opt):
        if opt=="EF":
            # euler forward
            k=dt*self.xdot(x1,u)
        elif opt=="EB":
            # euler backward
            k=dt*self.xdot(x2,u)
        elif opt=="Heun":
            # Heun's method
            k1=self.xdot(x1,u)
            k2=self.xdot(x1+k1*dt,u)
            k=0.5*(k1+k2)*dt
        elif opt=="Trap":
            # trapezoidal collocation
            k=0.5*dt*(self.xdot(x1,u)+self.xdot(x2,u))
        elif opt=="RK45":
            # RK45
            k1=self.xdot(x1,u)
            k2=self.xdot(x1+dt*k1/2,u)
            k3=self.xdot(x1+dt*k2/2,u)
            k4=self.xdot(x1+dt*k3,u)
            k=dt/6*(k1+2*k2+2*k3+k4)
        elif opt=="Her":
            xc=(x1+x2)/2+1/8*dt*(self.xdot(x1,u)-self.xdot(x2,u))
            uc=u
            k=dt/6*(self.xdot(x1,u)+4*self.xdot(xc,uc)+self.xdot(x2,u))
        return k    


class opt_BC:

    def __init__(self,N,X,U):
        self.x0=cs.MX(cs.vertcat(0,0))
        self.xf=cs.MX(cs.vertcat(1,0))
        #print(type(self.x0))
        self.res_BC=self.BC(self.x0,self.xf,N,X,U)

    def BC(self,x0,xf,N,X,U):
        return cs.vertcat(X[:,0]-self.x0,X[:,N]-self.xf)
        

        
        

OG=opt_grid(100)
OV=opt_problem(OG.Nx,OG.Nu,OG.N)
OO=opt_obj(OG.Nx,OG.Nu,OG.N,OG.dt,OV.X,OV.U)
OD=opt_dynamics(OG.N,OG.dt,OV.X,OV.U)
OBC=opt_BC(OG.N,OV.X,OV.U)
OV.OPT(OO.obj,OD.res,OBC.res_BC)
OV.test(OG.t,OG.N)








