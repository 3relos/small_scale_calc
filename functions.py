'''
Small-scale dynamo functions

January 2026
Vasco Silver - University of Bonn

Module with all necessary equation for small scale calculations.
Mostly rely on theory and calculations from J. Schober et al., 2013, A&A 560.
Further surces stated at appropriate places.
'''

# Necessary modules
import numpy as np
import astropy.constants as ac
import astropy.units as au

# DEFINITION OF FUNCTIONS
def first_exceed_index(array, value):
    '''
    Function to get the index of an array where a given value is first time exceeded.

    :param array: 1D-array
    :param value: value, that should be exceeded

    :return: Array index
    '''
    mask = array > value
    return np.argmax(mask) if mask.any() else None

def M_num(u_rms,cs):
    '''
    Function for the Mach number.

    :param u_rms: Root mean square velocity
    :param cs: Speed of Sound

    :return: Value for the Mach number
    '''
    return u_rms/cs

def f_M(M): # Fedderrath, Chabier, Schober et al. 2011 PRL
    '''
    Function for the ratio of magnetic over turbulent energy at saturation for the solenoidal case.

    :param M: Mach number

    :return: Value of the ratio
    '''
    p = [0.020, 2.340, 23.33, 2.340, 1., 0., 0.]
    return (p[0]*(np.power(M,p[1])+p[2])/(np.power(M,p[3])+p[4])+p[5])*np.power(M,p[6])

def c_s(gamma,T,m):
    '''
    Function to get the sound of speed in a medium.
    
    :param gamma: adiabatic coefficient
    :param T: temperature
    :param m: particle mass

    :return: Value of sound speed
    '''
    k = ac.k_B.to(au.erg/au.K)
    return np.sqrt(gamma*k*T/m)

def V_Kepler(n,m,R,H):
    G = ac.G
    return np.sqrt(G*n*m*np.pi*R*H)

def V_acc(beta,cs,sphere=True,V_K=0):
    '''
    Function for the turbulent velocity from mass accretion.

    :param beta: fraction of kinetic energy in turbulent motion
    :param cs: Speed of Sound
    :param sphere: Boolean if galaxy is spherical (True) or disk-like (False)
    :param n: Particle density
    :param m: Particle mass
    :param R: Galaxy disk radius
    :param H: Galaxy disk hight

    :return: Value for the velocity
    '''
    if sphere:
        return 2*cs*np.sqrt(beta)
    else:
        return V_K*np.sqrt(beta)

def vel_rms():
    '''
    Function for the rest-fram root-mean squared velocity.

    :param beta: fraction of kinetic energy in turbulent motion
    :param cs: Speed of Sound

    :return: Value for the velocity
    '''
    return 2

def Reynold(L,V,nu):
    '''
    Function to get the Reynolds number.

    :param L: Forcing scale
    :param V: velocity
    :param nu: viscosity

    :return: Value of the Reynold number
    '''
    return V*L/nu

def magn_Reynold(L,V,eta):
    '''
    Function for the magnetic Reynolds number

    :param L: Forcing scale
    :param V: velocity
    :param eta: magnetic resistivity

    :return: Value of the magnetic Reynolds number
    '''
    return V*L/eta

def Prandtl_num(nu,eta):
    '''
    Function for the magnetic Reynolds number

    :param nu: viscosity
    :param eta: magnetic resistivity

    :return: Value of the magnetic Reynolds number
    '''
    return nu/eta

def visc_scale(L,theta,Re): # Schober, Schleicher, Fedderrath et al. 2012
    '''
    Function to get the viscous scale length.

    :param L: Forcing scale
    :param theta: slope of velocity spectrum
    :param Re: Reynolds number

    :return: Value of l_nu
    '''
    return L/np.power(Re,1./(1.+theta))

def Gamma(L,V,theta,Re):
    '''
    Function for the groth factor of the field strength on forcing scale.

    :param L: Forcing scale
    :param V: velocity
    :param theta: slope of velocity spectrum
    :param Re: Reynolds number

    :return: Value of the growth rate
    '''
    return (163.-304*theta)/60*V/L*np.power(Re,(1.-theta)/(1+theta))

def t_nu_f(t,B_nusat,B_nut):
    '''
    Function to get the time of saturation on viscous scale.

    :param t: Time
    :param B_nusat: Saturation field strength on viscous scale
    :param B_Lnl: Field strength on viscous scale over time (**1D-array**)

    :return: Value for t_nu
    '''
    ind_tnu = first_exceed_index(B_nut,B_nusat)
    t_nu = t[ind_tnu]
    return t_nu

def t_L_f(t,B_Lsat,B_Lnl):
    '''
    Function to get the time of saturation on forcing scale.

    :param t: Time
    :param B_Lsat: Saturation field strength on forcing scale
    :param B_Lnl: Field strength for non-linear growth over time (**1D-array**)

    :return: Value for t_L
    '''
    ind_tL = first_exceed_index(B_Lnl,B_Lsat)
    t_L = t[ind_tL]
    return t_L

def t_eddy_to(L,V):
    '''
    Function to get the eddy turnover time for non-linear growth.

    :param L: Forcing scale
    :param V: velocity

    :return: Value of eddy turnover time
    '''
    return L/V

def l_p_t(t,t_nu,l_nu,L,V,theta):
    '''
    Function to get the peak scale.

    :param t: Time
    :param t_nu: viscous time scale
    :param l_nu: viscous scale
    :param L: Forcing scale
    :param V: velocity
    :param theta: slope of velocity spectrum


    :return: 1D-array
    '''
    return l_nu + np.power(V/np.power(L,theta)*(t-t_nu),1./(1-theta))

def B_L_exp(t,B_nu_0,l_nu,L,Gam):
    '''
    Function to get the field on turbulent forcing scale for the exponential growth.

    :param t: Time
    :param B_nu_0: Initial field on viscous scale
    :param l_nu: viscous scale
    :param L: Forcing scale
    :param Gam: Growth function

    :return: 1D-array
    '''
    return np.exp(t*Gam)*np.power(l_nu/L,5./4)*B_nu_0

def B_L_nl(t,t_nu,l_nu,L,rho,V,theta,M):
    '''
    Function to get the field on turbulent forcing scale for the nonlinear growth.

    :param t: Time
    :param t_nu: viscous time scale
    :param l_nu: viscous scale
    :param L: Forcing scale
    :param rho: density
    :param V: velocity
    :param M: Mach number

    :return: 1D-Array
    '''
    l_p = l_p_t(t,t_nu,l_nu,L,V,theta)
    return np.sqrt(4*np.pi*rho)*V*np.power(l_p/L,theta+5/4)*np.power(f_M(M),1./2)

def B_L_sat(rho,V,M):
    '''
    Function to get the saturation field on turbulent forcing scale.

    :param rho: density
    :param V: velocity
    :param M: Mach number

    :return: Value for saturated field strength on forcing scale
    '''
    return np.sqrt(4*np.pi*rho)*V*np.sqrt(f_M(M))

def B_nu_t(t,B_nu_0,Gam):
    '''
    Function to get the field on viscous scale before saturation.

    :param t: Time
    :param B_nu_0: Initial field on viscous scale
    :param Gam: Growth function

    :return: 1D-array
    '''
    return B_nu_0*np.exp(t*Gam)

def B_nu_sat(l_nu,L,rho,V,theta,M):
    '''
    Function to get the saturation field on viscous scale.

    :param l_nu: viscous scale
    :param L: Forcing scale
    :param rho: density
    :param V: velocity
    :param M: Mach number

    :return: Value for saturated field strength on viscous scale
    '''
    return np.sqrt(4*np.pi*rho)*np.power(l_nu/L,theta)*V*np.sqrt(f_M(M))

def B_L_t(t,t_nu,t_L,B_nu_0,l_nu,L,rho,V,theta,M,Gam):
    '''
    Function to get the field on turbulent forcing scale.
    
    :param t: Time
    :param t_nu: viscous time scale
    :param t_L: forcing time scale
    :param B_nu_0: Initial field on viscous scale
    :param l_nu: viscous scale
    :param L: Forcing scale
    :param rho: density
    :param V: velocity
    :param theta: slope of velocity spectrum
    :param M: Mach number
    :param Gam: Growth function

    :return: 2D-array of field strengths, dim 1: different initial B, dim 2: time steps
    '''
    # initiate result array
    res = []*au.gram**(1/2)/(au.centimeter**(1/2)*au.second) # Total array
    # Get different B-fields for different growth phases
    res = np.append(res, B_L_exp(t[t<t_nu],B_nu_0,l_nu,L,Gam))
    mask = (t>=t_nu) & (t<t_L)
    res = np.append(res, B_L_nl(t[mask],t_nu,l_nu,L,rho,V,theta,M))
    res = np.append(res, [B_L_sat(rho,V,M)]*len(t[t>=t_L]))
    # OUTPUT
    return res