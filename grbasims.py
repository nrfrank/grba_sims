'''
Author: Nathan Frank
Last Modified: 01/26/2016

Description: This module lays out a set of functions for running simulations
of Gamma-Ray Burst Afterglows. In addition to some helper functions for saving
arrays and images, the modules includes geometric functions for the calculation
of various angles on the surface and interior of a 3D sphere, i.e. the surface
of the blast wave, some root-finding methods, and definitions for calculating
the fundamental physical properties and surface brightness profile of the burst.
'''

''' Import Statement Block'''
#! /usr/bin/python

# import most of what we'll need.
import os, sys
from scipy import optimize
from scipy.integrate import quad  # , simps, romberg, quadrature
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from math import cos, acos, sin, sqrt, degrees, radians, copysign, pi
import numpy as np
import numpy.ma as ma
import time, datetime
import multiprocessing as mp
from ConfigParser import SafeConfigParser

# non-Windows specific import statements for matplotlib.
if 'win' not in sys.platform:
    print "Not running on windows."
    import matplotlib
    matplotlib.use('Agg')

# import matplotlib and various associated packages.
import matplotlib.pyplot as plt
import matplotlib.colors as c
from matplotlib.colors import LogNorm

# Define functions for saving arrays and matplotlib images.
def saveArray(array, path, ext='npy', verbose = True):
    '''
    Save an array, likely Numpy array, but possibly others.

    array: The object to be exported (saved). For example, a numpy array.
    
    path : The path (and filename, without the extension) to save the
        figure to.

    ext (default='npy'): The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

    verbose (default=True): Whether to print information about when and where the image
        has been saved.
        
    example:
        >>> saveArray('./path/to/save/to/array')
        Saving figure to './path/to/save/to/array.npy'
        
    '''
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
    
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving data array to '%s'..." % savepath)
    
    # Actually save the array
    np.save(savepath, array)
 
def save(path, ext='png', close=True, verbose=True):
    '''
    Save a figure from pyplot.

    path : The path (and filename, without the extension) to save the
        figure to.

    ext (default='png'): The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

    close (default=True): Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.

    verbose (default=True): Whether to print information about when and where the image
        has been saved.

    example:
        >>> save('./path/to/save/to/image')
        Saving figure to './path/to/save/to/image.png'
        
    '''
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath)

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

# Define generic "brent" root-finding method as well as bracketing methods
# for the lower and upper roots (y-(x) and y+(x)).
def brent(f, x1, x2, *args, **kwargs):
    '''
    Define generic "Brent" root-finding method. Requires root to lie within
    a bracketed interval defined by x1 and x2.
    
    f: function to find the root (zero crossing) of. Function may optionally
        accept arguments (args). See below.
        
    x1: Numeric lower bound of region with the root. Independent variable of f.
    
    x2: Numeric upper bound of region with the root. Independent variable of f.
    
    *args: Arguments to the function f, i.e. f is called as f(x, *args). Must be
        listed in the order they appear in the function definition.
    
    **kwargs: Optional keyword arguments to the brent method; see below. Allow
        users to adjust the precision, tolerance, and number of iterations.
        
    example:
        >>> f = lambda x: 3.0*x + 7.0
        >>> brent(f, -3, 0)
        -2.333333333333333
        >>> g = lambda x: 3*x**2 - 7*x - 13
        >>> brent(g, 0, 5)
        3.55297017726736
    '''
    tol = kwargs.get("tol", 1.0e-9)
    n = kwargs.get("n", 1000)
    EPS = kwargs.get("EPS", np.finfo(float).eps)
    
    a = x1; b = x2; c = x2
    fa = f(a, *args); fb = f(b, *args)
    
    if ((fa > 0.0 and fb > 0.0) or (fa < 0.0 and fb < 0.0)):
        sys.exit("Root must be bracketed in Brent method.")
    
    fc = fb
    
    for i in range(n):
        if ((fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0)):
            c = a
            fc = fa
            e = d = b-a
        if (abs(fc) < abs(fb)):
            a=b
            b=c
            c=a
            fa=fb
            fb=fc
            fc=fa
        
        tol1 = 2.0*EPS*abs(b) + 0.5*tol
        xm = 0.5*(c-b)
        
        if (abs(xm) <= tol1 or fb == 0.0):
            return(b)
        if (abs(e) >= tol1 and abs(fa) > abs(fb)):
            s = fb/fa
            if a == c:
                p = 2.0*xm*s
                q = 1.0 - s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2.0*xm*q*(q-r)-(b-a)*(r-1.0))
                q = (q-1.0)*(r-1.0)*(s-1.0)
            if p > 0.0:
                q = -q
            
            p = abs(p)
            min1 = 3.0*xm*q - abs(tol1*q)
            min2 = abs(e*q)
            
            if 2.0*p < (min1 if min1<min2 else min2):
                e = d
                d = p/q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a = b
        fa = fb
        if (abs(d) > tol1):
            b += d
        else:
            b += copysign(tol1, xm)
        fb = f(b, *args)
    sys.exit("Maximum number of iterations exceeded")

# Function to determine y- root intervals.
def ymBracket(f, xl, xh, *args, **kwargs):
    '''
    Simple bracketing method for lower y root (y-).
    
    f: function passed to bracketing method. This will always be a specially
        defined function based on the physical definition of y.
        
    xl: lower bound of the y- root; usually 0.0.
    
    xh: uppwer bound of the y- root; usually 0.7 as the peak value is around 0.65.

    *args: arguments to the function f.
    
    **kwargs: keyword agruments to the bracketing method.
    
    '''
    num = kwargs.get('num', 100)
    
    vals = np.linspace(xh, xl, num)
    fh = f(xh, *args)
    
    for i in range(len(vals)):
        xl = vals[i]
        fl = f(xl, *args)
        
        if copysign(1.0, fl) == -copysign(1.0, fh):
            # print "ym root lies in interval: ", xl, xh, fl, fh
            return((xl, xh))
        xh = xl
        fh = f(xh, *args)
    # print "Zero crossing was not found."
    return()

# Function to determine the y+ root intervals.
def ypBracket(f, xl, xh, *args, **kwargs):
    '''
    Simple bracketing method for upper y root (y+).
    
    f: function passed to bracketing method. This will always be a specially
        defined function based on the physical definition of y.
        
    xl: lower bound of the y+ root; usually 0.6.
    
    xh: uppwer bound of the y+; usually 1.0.

    *args: arguments to the function f.
    
    **kwargs: keyword agruments to the bracketing method.
    
    '''
    num = kwargs.get('num', 100)
    
    vals = np.linspace(xl, xh, num)
    fl = f(xl, *args)
    for i in range(len(vals)):
        xh = vals[i]
        fh = f(xh, *args)
        if copysign(1.0, fh) == -copysign(1.0, fl):
            # print "yp root lies in interval: ", xl, xh
            return((xl, xh))
        xl = xh
        fl = f(xl, *args)
    # print "Zero crossing was not found."
    return()

# Define geometric functions for various angles describing the afterglow.
def angle(x1, y1, x2, y2):
    '''
    Define the angle between two vectors with tails at (0,0) and heads at
    (x, y) locations on a grid. Uses a dot product to find the angle between
    vectors defined by these points. Always returns an angle in the interval [0, pi].
    
    (x1, y1): coordinates for first point on grid.
    
    (x2, y2): coordinates for second point on grid.
    
    example:
        >>> angle(-5, 0, 5, 0)  # should return pi
        3.1415926535897931
        >>> angle(0, 10, 10, 0) # should return pi/2
        1.5707963267948966
        >>> angle(0, 5, 5, 5)  # should return pi/4
        0.78539816339744839
    '''
    
    numer = (x1 * x2 + y1 * y2)
    denom = np.sqrt((np.power(x1, 2.0) + np.power(y1, 2.0)) * (np.power(x2, 2.0) + np.power(y2, 2.0)))
    
    return(np.arccos(numer/denom))

def theta(y, rPerp):
    '''
    Define the angle, theta, between the line of sight from observer to the
    GRB and a point at a perpendicular distance rPerp away from the L.O.S.
    Using the small angle approximation we define theta as:
        theta ~ rPerp/R ~ rPerp/(y*Rl).
    
    y [0-1]: The scaled variable in the radial direction. y := R/Rl.
    
    rPerp [0-1]: The perpendicular distance from the LOS in scaled units of Rl.
        Through testing it should not generally be greater than 0.2.
    
    example:
        >>> theta(0.5, 0.1)
        0.2
        >>> theta(0.1, 0.2)
        nan
    '''
    if y < rPerp:
        # y = rPerp
        return(float('nan'))
    else:
        # return(np.arcsin(rPerp/y))
        return(rPerp/y)

def thetaPrime(y, rPerp, thetaV, phi):
    '''
    Define the angle on the 3D spherical emitting surface. Employs the
    shperical law of cosines.
    
    y [0-1]: The scaled variable in the radial direction. y := R/Rl.
    
    rPerp [0-1]: The perpendicular distance from the LOS in scaled units of Rl.
        Through testing it should not generally be greater than 0.2.
    
    thetaV [0-pi/4]: The viewing angle, in radians, between the LOS to observer
        and the jet emission axis.
    
    phi [0-pi]: Interior angle of spherical triangle. phi = 0 corresponds to the
        direction toward the main axis from the LOS.
    '''

    value = np.cos(thetaV)*np.cos(theta(y, rPerp)) + np.sin(thetaV)*np.sin(theta(y, rPerp))*np.cos(phi)

    return(np.arccos(value))

if __name__ == '__main__':
    import sys
    view = sys.argv[1]
    
    # Define some constants.
    k = 0.0
    p = 2.2
    t0 = 1.0
    gamma0 = 50.0
    thetaV = radians(6.0)
    sig = radians(2.0)
    kap = 10.0
    dist = 0.2
    offset = 0.01
    n_grid = 100
    tolVal = 1.0e-9
    
    if view == 'front':
        xg = np.linspace(offset - dist, offset + dist, n_grid)
        yg = np.linspace(-dist, dist, n_grid)
        
        X, Y = np.meshgrid(xg, yg)
        array = np.zeros((n_grid, n_grid))
        
        for i in range(len(X)):
            for j in range(len(X[i])):
                phi = angle(offset, 0.0, offset - X[i][j], Y[i][j])
                rPerp = np.sqrt(np.power(X[i][j] - offset,2.0) + np.power(Y[i][j], 2.0))
                
                array[i][j] = degrees(thetaPrime(1.0, rPerp, thetaV, phi))

        
    elif view == 'top':
        xg = np.linspace(-dist, dist, n_grid)
        yg = np.linspace(0.0, 1.0, n_grid)
        
        X, Y = np.meshgrid(xg, yg)
        array = np.zeros((n_grid, n_grid))
        
        for i in range(len(X)):
            for j in range(len(X[i])):
                y = Y[i][j]
                if X[i][j] > 0.0:
                    phi = radians(180.0)
                    rPerp = X[i][j]
                else:
                    phi = radians(0.0)
                    rPerp = -X[i][j]
                
                array[i][j] = thetaPrime(y, rPerp, thetaV, phi)
    
    else:
        print "Invalid view chosen. Try again."
    
    array = ma.masked_invalid(array)
    
    plt.figure()
    plt.pcolormesh(X, Y, array, cmap = plt.cm.jet)
    plt.colorbar()
    plt.axis([min(xg), max(xg), min(yg), max(yg)])
    if view == 'top':
        plt.gca().invert_yaxis()
    plt.plot([min(xg),max(xg)],[0.0,0.0],color = 'black',linestyle = 'dashed')
    plt.plot([0.0,0.0],[min(yg),max(yg)],color = 'black',linestyle = 'dashed')
    plt.scatter(offset,0.0,color = 'black')
    plt.show()