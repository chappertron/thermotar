import pandas as pd
import thermotar as th
import numpy as np

import matplotlib.pyplot as plt

import warnings

from scipy import interpolate   

import thermotar as th


from thermotar.sub_modules.potential_chunk import Potential


# fit within specified range to specified orer polynomial   
def ranged_poly_fit(y,x,n=3,xl=None,xh=None,**kwargs):
    '''
        In the range of data, in x, fit to a polynomial of the given order.
        Essentially just short hand for this
        if xh and xl are not specified, it is just a regular polyfit

    '''

    if not xl: xl = x.min()
    if not xh: xh = x.max()
        
    select = (x >= xl) & (x <= xh)
    
    xs = x.loc[select]
    ys = y.loc[select]
    
    return np.polyfit(xs,ys,n,**kwargs)


def get_poly_min(fit,xh=None,xl=None):
    ''' 
        For a given set of polynomial coefficients, calculate the location of the minimum

        Looks only for global minimum for points in the range

        Finds the minima from the coefficients

        Ensure that only those taht are minima are added
        
        


        Maybe also validate that the value at the edge is not the same as the value of the minimum -  if minimum is at edge, suggests that there isn't a true inversion.
    '''

    poln = np.poly1d(fit) # create a polynomial from the coefficients
    crit_points = poln.deriv().r #roots of the derivative of the polynomial
    # filter crit points to be real
    crit_points_real = crit_points[crit_points.imag==0].real
    # filter further to ensure they are minima not maxima or points of inflection.

    if xh and xl:
        select = (crit_points_real <= xh) & (crit_points_real >= xl)
        crit_points_real = crit_points_real[select]
    # filter last so that 
    crit_points_real = crit_points_real[poln.deriv(2)(crit_points_real) > 0] # NB 2nd derivative is strictly greater than so that inflection points aren't found
    
    # y_crits

    y_crits = poln(crit_points_real) # evaluate the polynomial at the critical points

    y_min = y_crits.min() # find the critical points with the lowest value of y

    ### Old Implementation
    #y = np.polyval(fit
    #y_min = np.min(y)
    
    
    x_min = np.asscalar(crit_points_real[y_crits==y_min]) # go back to finding which on is the minimum
    
    return x_min,y_min



def basic_min(x,y):
    '''
        Find the minimum indexes of a dataframe, by using .min() and find the corresponding x value
    '''

    y_min = np.min(y)

    x_min = x[y == y_min]

    return x_min,y_min

def choose_temp_range(df ,ptp = 200, pot_name = 'phi_tot',temp_name ='temp' ):
    '''
        Take a chunk, find the absolute minimum potential and then return the range of ptp centred on this minimum, and an array of length grid points between the max and min
        The returned array is for use in interpolation with the poly fit later
    '''
    T = df[temp_name] # get the temperature data

    pot = df[pot_name] # get the potential data

    T_min, _pot_min = basic_min(T,pot) # find the temperature corresponding to the absoulte lowest value of the potential

    T_min = np.asscalar(T_min)

    Tl = T_min - ptp/2 # upper and lower limits of an interval ptp wide centred about T_min
    Th = T_min + ptp/2

    return Tl, Th
    

 

def find_min(y,x, n,  xl=None,xh=None,grid = 100000,err = False,validate = True):
    '''
        Find the minimum of one series with respect to another, using polynomial fittings

        interp_grid = grid to use for interpolation

        Interpolate with polynomials??? 

        y = data

        x = x data

        n = polynomial order to use

        TODO: Don't use a grid to find the minimum. Use a np.poly1d object to find the critical points, filter to be within the region ( and real) and the find the lowest of these!!!!


        Maybe also validate that the value at the edge is not the same as the value of the minimum -  if minimum is at edge, suggests that there isn't a true inversion.

    Optional inputs:

        xmin, xmax = range to fit over


    '''

    if not xh: xh = np.max(x)
    if not xl: xl = np.min(x)


    fit = ranged_poly_fit(y,x,n=n,xl=xl,xh=xh )

    #xs = np.linspace(xl,xh,grid)

    try:
        x_min,y_min = get_poly_min(fit,xl=xl,xh=xh) # to do, find more precise analytical minimum.
    except ValueError:
        x_min,y_min = (np.nan, np.nan)


    return x_min,y_min, fit


def find_phi_min(chunk,n,potential_name = 'phi_tot', temp_name = 'temp',temp_range = 300,temp_centre = None,show_plots = False,grid=100000,verbose = False,plot_markers = 10):
    temps = chunk.data[temp_name]
    phis = chunk.data[potential_name]

    if not temp_centre and (temp_range is not None):
        Tl,Th = choose_temp_range(chunk.data, ptp = temp_range,pot_name = potential_name, temp_name=temp_name)
    elif temp_range is not None:
        Tl,Th = (temp_centre - temp_range/2,temp_centre+temp_range/2)
    else: 
        Tl,Th = (temps.min(),temps.max())
    
    # don't over extend the range, otherwise and incorrect minima will be found!!!!
    if Th > temps.max(): Th = temps.max()
    if Tl < temps.min(): Tl = temps.min()


    if verbose: print(f'Fitting a {n}-order polynomial between T = {Tl:.3f},{Th:.3f} K.')

    T_min,phi_min,fit =  find_min(phis,temps,n,xl=Tl,xh=Th,grid=grid)

    if verbose: print(f'Minimum found at T = {T_min:.3f} ')

    if show_plots:    
        Ts = np.linspace(Tl,Th,grid)
        plt.plot(Ts,np.polyval(fit,Ts),c='b',label =f'{n} order fit ',ls = '--')
        plt.plot(temps,phis,'ro' ,markevery = plot_markers,label='data')
        plt.plot(T_min,phi_min,'ko')
        plt.xlabel(r'$T$/K')
        plt.ylabel(r'$\phi$/V')
        plt.legend()
        plt.show()

    return T_min,phi_min,fit

def find_x_intercept(y,x,offset=0, xmin=None,xmax=None,interp_grid = None, interp_modde = 'linear'):
    
    '''
        Find the x intercept of a set of data with a finite grid.

        Uses a scipy tool to find the closest match to zero(+offset), then the corresponding finite value of x

        can restrict to a range to prevent finding fake minima, for example noise in the data giving a minima that is not true??

        interp grid is there to interpolate if need be. If used will interpolate y data between xmin and xmax with the specified number of points

    '''
    # If not specified, set to maximum and minimum value of range
    if not xmin: xmin = np.min(x)
    if not xmax: xmax = np.min(x)
    
    if interp_grid:
        # if interpolation is desired, do it, else, don't 
        x_new = np.linspace(xmin,xmax,interp_grid)
        f = interpolate.interp1d(x,y)
        y_new = f(x_new)
    else:
        x_new = x
        y_new = y

    pass



def profile_calculating(chunk:Potential,w = 5,sigma = 3,win_type = None, trim_w = 5,bw = None,show_plots = False,recalc_post_trim = False,direct=False,correct = ['cos_theta']):
    ''' 
        Does a lot 

        bw: float, None 
            rebin the data with the specified bin width, if not None.

        w: int
            Number of points for rolling averages, set to 1 for none
            TODO: if one or None, bypass

        trim_w: float
            Distance to trim of each 'end' of the box. Distance units 
        
        win_type: str, None
            Type of window to use for the rolling average. Default is None, => rectangular window
        
        sigma: int/float TODO: Check, which
            Standard deviation/parameter to use for the rolling window 


        direct: bool
            If true, calculate STP directly from E/\grad T, else, calculate from the numerical derivative of phi with temp.
            default: False
        
        correct: list
            List of column names to apply corrections to, before processing. ensures properties that are zero at box edges are.
           
    '''

    if bw is not None:
        # rebin the data -> reduce many bins to 1
        chunk.rebin('coord',bw=bw,inplace=True)

    chunk.centre()

    for col in correct:
        chunk.correct(col,how = 'ave')

    # smooth
    chunk_smoothed = Potential(chunk.data.rolling(w,win_type=win_type).mean(std = sigma).copy())

    # should be done post trimming tbh
    #if recalc_post_smooth: chunk_smoothed.calculate_potentials()

    # Calculate STP and temp_grad

    chunk_smoothed.prop_grad('temp','coord')

    # Density Temperature gradient
    chunk_smoothed.data['drho_dT'] = np.gradient(chunk_smoothed.density_mass,chunk_smoothed.temp)
    # different ways to calculate 
    
    ## Os theta gradient ratio

    chunk_smoothed.data['cos_theta_grad_T'] = chunk_smoothed.cos_theta/chunk_smoothed.temp_grad
    

    if direct:
        chunk_smoothed.data['STP'] = chunk_smoothed.E_tot/chunk_smoothed.temp_grad
        chunk_smoothed.data['STP_P'] = chunk_smoothed.E_P_z/chunk_smoothed.temp_grad
        chunk_smoothed.data['STP_Q'] = chunk_smoothed.E_Q_zz/chunk_smoothed.temp_grad
    else:
        chunk_smoothed.data['STP'] = -1*np.gradient(chunk_smoothed.phi_tot, chunk_smoothed.temp)
        chunk_smoothed.data['STP_P'] = -1*np.gradient(chunk_smoothed.phi_P_z,chunk_smoothed.temp)
        chunk_smoothed.data['STP_Q'] = -1*np.gradient(chunk_smoothed.phi_Q_zz,chunk_smoothed.temp)

    ## calculate pressure. Assume z direction
    try:
        chunk_smoothed.data['Press'] = -1*(chunk_smoothed.loc_stress_3)*chunk_smoothed.density_number
    except AttributeError:
        warnings.warn('Pressure could not be computed. Local stress loc_stress_3 not found. TODO Add options for choosing the column')

    chunk_smoothed.raise_columns()

    # trim the fatt

    coord = chunk_smoothed.coord
    
    select = (coord.abs() < coord.max()-trim_w) & (coord.abs() > trim_w)

    chunk_trimmed =  Potential(chunk_smoothed.data.loc[select].copy())

    if recalc_post_trim:
        chunk_trimmed.calculate_potentials()
        chunk_trimmed.prop_grad('temp','coord')
        chunk_trimmed.data['STP'] = chunk_trimmed.E_tot/chunk_trimmed.temp_grad
        chunk_trimmed.data['STP_P'] = chunk_trimmed.E_P_z/chunk_trimmed.temp_grad
        chunk_trimmed.data['STP_Q'] = chunk_trimmed.E_Q_zz/chunk_trimmed.temp_grad
        chunk_trimmed.raise_columns()
    
    change_parity = True
    if change_parity:
        try:
            chunk_trimmed.parity('E_tot')
            chunk_trimmed.parity('E_Q_zz')
            chunk_trimmed.parity('E_P_z')
            chunk_trimmed.parity('temp_grad')
            chunk_trimmed.parity('cos_theta')
        except:
            pass

    chunk_folded = Potential(chunk_trimmed.fold_and_ave())


    return chunk_smoothed, chunk_trimmed, chunk_folded
