"""
Newton's descent method
"""
############################
##### IMPORTED MODULES #####
############################
import numpy as np
import scipy.optimize as spo
import sys
import matplotlib.pyplot as plt

################################
##### FUNCTION DEFINITIONS #####
################################
#***** Fonctions test *****
#----- Quadratic function -----
def func_quad(xx):
    yy=xx.flatten()
    return np.asarray(2.0*yy[0]**2.0+yy[0]*yy[1]+yy[1]**2.0+yy[0]+yy[1])

def grad_quad(xx):
    yy=xx.flatten()
    return np.asarray([[4.0*yy[0]+yy[1]+1.0],[2.0*yy[1]+xx[0]+1.0]])

def hess_quad(xx):
    yy=xx.flatten()
    return np.asarray([[4.0,1.0],[1.0,2.0]])

#----- Rosenbrock function -----
def func_rosen(xx):
    yy=xx.flatten()
    return spo.rosen(yy)

def grad_rosen(xx):
    yy=xx.flatten()
    gra=spo.rosen_der(yy)
    return gra.reshape((gra.shape[0],1))

def hess_rosen(xx):
    yy=xx.flatten()
    return spo.rosen_hess(yy)

#***** Algorithms *****
#----- Step calculation algorithm -----
def compute_rho_Newton():
    """
    Calculating the descent step for Newton's method, which is equal to 1 in our case.

   
    """
    return(1)
    

    
def compute_rho_BFGS(xx,dire,func,tol,intervalle):
    """
    Golden-section search algorithm

    xx : current point, numpy array
    dire : current descent direction, numpy array
    func : function to minimize, python function
    tol : precision parameter, float
    

    Returns the step value for the BFGS method, float.
    """
    def q(rho):
        return func(xx+rho*dire)
    k=0
    a=0
    tho=(1.0+np.sqrt(5))/2.0
    c=intervalle
    while q(c)>=q(a):
        c-=(1.0-(1.0/tho))*(c-a)
    j=2
    b=a+j*(c-a)
    while q(b)<=q(c):
        j+=1
        b=a+j(c-a)
        
    while np.abs(b-a)>=tol and k<=20:
        ap=a+(b-a)*(1.0/tho)**2
        bp=a+(b-a)*(1.0/tho)
        
        if q(ap)<q(bp):
            b=bp
        elif q(ap)>q(bp):
            a=ap
        elif q(ap)==q(bp):
            a=ap
            b=bp
        k+=1
    return (a+b)/2.0

#----- Descent direction calculation algorithm -----
def compute_dire_Newton(xx,grad,hess):
    """
    Calculating Newton's descent direction.
    
    xx : current point, numpy array
    grad : gradient function, python function
    hess : Hessian function, python function

    Returns the direction of descent, numpy array.
    """
    return np.dot(np.linalg.inv(hess(xx)), -grad(xx))

def compute_dire_BFGS(xx,rho,grad,dire,Hk):
    """
    Calculating descent direction with BFGS

    xx : current point, numpy array
    rho : descent step, float
    grad : gradient function, fonction python
    dire : current direction, numpy array
    Hk : BFGS matrix, numpy array

    Returns the new descent direction, numpy array.
    Returns the new Hk matrix, numpy array.
    """
    X=xx-rho*dire
    dx=xx-X
    dg=grad(xx)-grad(X)
    A=np.vdot(dg,dx)
    B=np.dot((dg.T),Hk)
    C=np.dot(B,dg)
    
    
    D=np.dot(dx,dx.T)
    E=np.dot(dg,(dx.T))
    F=np.dot(Hk,E)
    
    U=(1+(C/A))*(D/A)-((F+F.T)/A)    
    
    S=Hk+U
    d=-np.dot(S,grad(xx))
    return d,S

#***** Plot function *****
def plot(xx_list,residu_list,function):
    """
    Ploting results.
    """
    #***** Plot results *****
    plt.figure()

    #***** Coordinates *****
    tmp_array=np.array(xx_list,ndmin=2)[:,:,0].T
    X1_min, X1_max=tmp_array[0].min(), tmp_array[0].max()
    X2_min, X2_max=tmp_array[1].min(), tmp_array[1].max()

    X1, X2 = np.meshgrid(np.linspace(X1_min-0.1*abs(X1_max-X1_min), \
                                     X1_max+0.1*abs(X1_max-X1_min), \
                                     101), \
                         np.linspace(X2_min-0.1*abs(X2_max-X2_min), \
                                     X2_max+0.1*abs(X2_max-X2_min), \
                                     101))

    #***** Function values at coordinates *****
    Z=np.zeros_like(X1)
    for i in xrange(Z.shape[0]):
        for j in xrange(Z.shape[1]):
            Z[i,j]=function(np.array([[X1[i,j]],[X2[i,j]]]))

    #***** Plot *****
    plt.contour(X1,X2,Z)

    xx_list=np.array(xx_list,ndmin=2)[:,:,0].T
    plt.plot(xx_list[0],xx_list[1],'k-x')

    #***** Plot history of convergence *****
    plt.figure()
    plt.plot(residu_list)
    plt.yscale('log')
    plt.grid()

    plt.show()

###############################
####### GRADIENT METHOD #######
###############################
"""
Tests
"""
from gradients import *
###### TEST 01: generic gradient #####
#def func_01(xx):
#    return np.asarray(2.0*xx[0]**2.0+xx[0]*xx[1]+xx[1]**2.0+xx[0]+xx[1])
#
#def grad_01(xx):
#    return np.asarray([4.0*xx[0]+xx[1]+1.0,2.0*xx[1]+xx[0]+1.0])
#
#test_01=gene_grad()
#
#test_01.param['function']=func_01
#test_01.param['gradient']=grad_01
#
#test_01.param['descent']['method']='conjugate'
#test_01.param['step']['method']='optimal'
#test_01.param['step']['optimal']['golden section']['tolerance']=1.0e-8
#test_01.param['step']['optimal']['golden section']['interval']=2.0
#
#test_01.param['guess']=np.asarray([[4.0],[-3.0]])
#test_01.param['tolerance']=1.0e-8
#test_01.param['itermax']=10000
#
#print "###################"
#print "##### TEST 01 #####"
#print "###################"
#test_01.run()
#test_01.plot()
#
###### TEST 02: generic gradient : Rosenborck function #####
#test_02=gene_grad()
#
#test_02.param['function']=spo.rosen
#test_02.param['gradient']=spo.rosen_der
#
#test_02.param['descent']['method']='conjugate'
#
#test_02.param['step']['method']='optimal'
#test_02.param['step']['optimal']['method']='golden section'
#test_02.param['step']['optimal']['golden section']['tolerance']=1.0e-8
#test_02.param['step']['optimal']['golden section']['interval']=10.0
#
#test_02.param['guess']=np.asarray([[-1.0],[-1.0]])
#test_02.param['tolerance']=1.0e-10
#test_02.param['itermax']=100000
#
#print "###################"
#print "##### TEST 02 #####"
#print "###################"
#test_02.run()
#test_02.plot()
#
###### TEST 03: generic gradient : Rosenborck function #####
#test_03=gene_grad()
#
#test_03.param['function']=spo.rosen
#test_03.param['gradient']=spo.rosen_der
#
#test_03.param['descent']['method']='conjugate'
#
#test_03.param['step']['method']='optimal'
#test_03.param['step']['optimal']['method']='golden section'
#test_03.param['step']['optimal']['golden section']['tolerance']=1.0e-8
#test_03.param['step']['optimal']['golden section']['interval']=10.0
#
#test_03.param['guess']=np.asarray([[-10.0],[-10.0]])
#test_03.param['tolerance']=1.0e-10
#test_03.param['itermax']=100000
#
#print "###################"
#print "##### TEST 03 #####"
#print "###################"
#test_03.run()
#test_03.plot()

#############################
###### NEWTON'S METHOD ######
#############################

##### Case 1 : Simple Newton, quadratic function #####
#***** Initialisation *****
xx=np.array([[4.0],[-3.0]])
list_xx=[xx]

dire=compute_dire_Newton(xx,grad_quad,hess_quad)

list_residu=[np.linalg.norm(grad_quad(xx),2)]

kmax=10000
tol=1.0e-10
k=0
#***** Loop *****
while np.linalg.norm(grad_quad(xx),2)>=tol and k<=kmax:
    #----- rho(k) -----
    rho=compute_rho_Newton()

    #----- x(k+1) -----
    xx=xx+rho*dire
    list_xx.append(xx)
    
    #----- d(k+1) -----
    dire=compute_dire_Newton(xx,grad_quad,hess_quad)

    #----- Remainder -----
    list_residu.append(np.linalg.norm(grad_quad(xx),2))

    #----- Increasing k -----
    k+=1
#
##***** Results *****
#
#print xx,k,np.linalg.norm(grad_quad(xx),2)
#plot (list_xx,list_residu,func_quad)

##### Case 2 : BFGS, quadratic function #####
#***** Initialization *****
xx=np.array([[4.0],[-3.0]])
list_xx=[xx]

Hk=np.identity(2)
dire=-Hk.dot(grad_quad(xx))

list_residu=[np.linalg.norm(grad_quad(xx),2)]

kmax=10
tol=1.0e-8
k=0
#***** Boucle *****
while np.linalg.norm(grad_quad(xx),2)>=tol and k<=kmax:
    #----- rho(k) -----
    rho=compute_rho_BFGS(xx,dire,func_quad,1.0e-7,10.0)

    #----- x(k+1) -----
    xx=xx+rho*dire
    list_xx.append(xx)

    #----- d(k+1) -----
    dire,Hk=compute_dire_BFGS(xx,rho,grad_quad,dire,Hk)

    #----- Remainder -----
    list_residu.append(np.linalg.norm(grad_quad(xx),2))

    #----- Increasing k -----
    k+=1

#***** Results *****
print(xx,k,np.linalg.norm(grad_quad(xx),2))
plot (list_xx,list_residu,func_quad)

