#! /usr/bin/env python

import math
import numpy as np
import astropy.units as u
#from cdecimal import Decimal
import decimal ###### acrescentei
#import asciidata, pyfits
import os, sys, math
from pylab import *
from numpy import *
#from jlu.gc.gcwork import starset
#from jlu.gc.gcwork import objects
#from jlu.gc.gcwork import util
#from jlu.gc.gcwork import young
#import pdb

def thieleinnes(a_au, e, incl, mstar , omega, bigOm, t_peri, t):
    pi_g = np.arccos(-1)
    
    #----------
    # Thiele-Innes Constants
    #----------
    cos_om = math.cos(math.radians(omega))
    sin_om = np.sin( math.radians(omega) )
    cos_bigOm = np.cos( math.radians(bigOm) )
    sin_bigOm = np.sin( math.radians(bigOm) )
    cos_i = np.cos( math.radians(incl) )
    sin_i = np.sin( math.radians(incl) )
    
    conA = a_au * (cos_om * cos_bigOm  - sin_om * sin_bigOm * cos_i)
    conB = a_au * (cos_om * sin_bigOm  + sin_om * cos_bigOm * cos_i)
    #conC = a_au * (sin_om * sin_i)
    conF = a_au * (-sin_om * cos_bigOm - cos_om * sin_bigOm * cos_i)
    conG = a_au * (-sin_om * sin_bigOm + cos_om * cos_bigOm * cos_i)
    #conH = a_au * (cos_om * sin_i)

    ecc = e

    mtot = mstar
    GG = 1.        #39.478/(4.*pi_g**2.)
    p_yr = np.sqrt((a_au**3.)/(mtot*GG))
    m = 2.*pi_g*((t-t_peri)/p_yr % 1.)
    #e1 = m + e*np.sin(m) + ((e**2.)*np.sin(2.*m)/2.)

    # Eccentric Anomaly
    def eccen_anomaly(m, ecc, thresh=1e-10):
        """
        m - a numpy array of mean anomalies
        ecc - the eccentricity of the orbit (single float value from 0-1)
        """
        # set default values

        #if (ecc < 0. or ecc >= 1.):
        #    print('Eccentricity must be 0<= ecc. < 1')
            
        #
        # Range reduction of m to -pi < m <= pi
        #
        mx = m.copy()

        ## ... m > pi
        zz = (where(mx > pi_g))[0]
        mx[zz] = mx[zz] % (2.0 * pi_g)
        zz = (where(mx > pi_g))[0]
        mx[zz] = mx[zz] - (2.0 * pi_g)

        # ... m < -pi
        zz = (where(mx <= -pi_g))[0]
        mx[zz] = mx[zz] % (2.0 * pi_g)
        zz = (where(mx <= -pi_g))[0]
        mx[zz] = mx[zz] + (2.0 * pi_g)

        #
        # Bail out for circular orbits...
        #
        if (ecc == 0.0):
            return mx

        aux   = (4.0 * ecc) + 0.50
        alpha = (1.0 - ecc) / aux

        beta = mx/(2.0*aux)
        aux = sqrt(beta**2 + alpha**3)
   
        z=beta+aux
        zz=(where(z <= 0.0))[0]
        z[zz]=beta[zz]-aux[zz]

        test=abs(z)**0.3333333333333333

        z =  test.copy()
        zz = (where(z < 0.0))[0]
        z[zz] = -z[zz]

        s0=z-alpha/z
        s1 = s0-(0.0780 * s0**5) / (1.0 + ecc)
        e0 = mx + ecc*((3.0 * s1) - (4.0 * s1**3))

        se0=sin(e0)
        ce0=cos(e0)

        f  = e0 - (ecc*se0) - mx
        f1 = 1.0 - (ecc*ce0)
        f2 = ecc*se0
        f3 = ecc*ce0
        f4 = -1.0 * f2
        u1 = -1.0 * f/f1
        u2 = -1.0 * f/(f1 + 0.50*f2*u1)
        u3 = -1.0 * f/(f1 + 0.50*f2*u2
                 + 0.166666666666670*f3*u2*u2)
        u4 = -1.0 * f/(f1 + 0.50*f2*u3
                 + 0.166666666666670*f3*u3*u3
                 + 0.0416666666666670*f4*u3**3)

        eccanom=e0+u4

        zz = (where(eccanom >= 2.00*pi_g))[0]
        eccanom[zz]=eccanom[zz]-2.00*pi_g
        zz = (where(eccanom < 0.0))[0]
        eccanom[zz]=eccanom[zz]+2.00*pi_g

        # Now get more precise solution using Newton Raphson method
        # for those times when the Kepler equation is not yet solved
        # to better than 1e-10
        # (modification J. Wilms)

        mmm = mx.copy()
        ndx = (where(mmm < 0.))[0]
        mmm[ndx] += (2.0 * pi_g)
        diff = eccanom - ecc*sin(eccanom) - mmm

        ndx = (where(abs(diff) > 1e-10))[0]
        for i in ndx:
            # E-e sinE-M
            fe = eccanom[i]-ecc*sin(eccanom[i])-mmm[i]
            # f' = 1-e*cosE
            fs = 1.0 - ecc*cos(eccanom[i])
            oldval=eccanom[i]
            eccanom[i]=oldval-fe/fs

            loopCount = 0
            while (abs(oldval-eccanom[i]) >= thresh):
                # E-e sinE-M
                fe = eccanom[i]-ecc*sin(eccanom[i])-mmm[i]
                # f' = 1-e*cosE
                fs = 1.0 - ecc*cos(eccanom[i])
                oldval=eccanom[i]
                eccanom[i]=oldval-fe/fs
                loopCount += 1
                
                if (loopCount > 10**6):
                    msg = 'eccen_anomaly: Could not converge for e = %d' % ecc
                    raise EccAnomalyError(msg)

            while (eccanom[i] >=  pi_g):
                eccanom[i] = eccanom[i] - (2.0 * pi_g)
                
            while (eccanom[i] < -pi_g ):
                eccanom[i] = eccanom[i] + (2.0 * pi_g)

        return eccanom

    Ebig = eccen_anomaly(m, ecc, thresh=1e-10)
    X_l = np.cos(Ebig) - e 
    Y_l = np.sqrt(1-e**2.)*np.sin(Ebig) 

    #### flipped! N is up, E is left, see Heintz 1990
    #### Normally it would be -> N is x pos axis, E is pos y axis
    y = (conA*X_l + conF*Y_l)         
    x = -(conB*X_l + conG*Y_l)

    return np.array([x,y])

class EccAnomalyError(Exception):
    def __init__(self, message):
        self.message = message
