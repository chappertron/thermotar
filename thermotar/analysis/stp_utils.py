import pandas as pd
import thermotar as th
import numpy as np

import matplotlib.pyplot as plt

import warnings

from scipy import interpolate, optimize


import thermotar as th


from thermotar.sub_modules.potential_chunk import Potential


# fit within specified range to specified orer polynomial   
def ranged_poly_fit(y,x,n=3,xl=None,xh=None,**kwargs):
    '''
        In the range of data, in x, fit to a polynomial of the given order.
        Essentially just short hand for this
        if xh and xl are not specified, it is just a regular polyfit

    '''

    if xl is None: xl = x.min()
    if xh is None: xh = x.max()
        
    select = (x >= xl) & (x <= xh)
    
    xs = x.loc[select]
    ys = y.loc[select]
    
    return np.polyfit(xs,ys,n,**kwargs)

def quad_w_min(x,x0,y0,G):
    return G*(x-x0)**2+y0

def alternate_fit(y:pd.Series,x:pd.Series,sigma:pd.Series=None,xl=None,xh=None,constrain_curvature : bool = True,constrain_min : bool = False,func = quad_w_min ,**kwargs):
    '''
        Like ranged polyfit, but now fits to a quadratic function with the minimum location explicitly a parameter
        fit to quad_w_min


        y : array_like = ydata to be fitted to
        x : array_like = xdata to be fitted to
        sigma : array_Like = error associated with the ydata
        xl : float = lower limit of x data to fit
        xh : float = upper limit of x data to fit 
        constrain curvature : bool = True (default) - Forces the positive curvature of the quadratic quad_w_min
        constrain_min : bool = True (default) forces minimum to be a positive value!!!

    '''

    if xl is None : xl = x.min()
    if xh is None : xh = x.max()

    select = (x >= xl) & (x <= xh)
    
    xs = x.loc[select]
    ys = y.loc[select]
    
    xs_arr = xs.to_numpy()
    ys_arr = ys.to_numpy()
    
    ## DEBUG
    # print(f'{xl=}')
    # print(f'{xh=}')
    # print(f'{xs.min()=}')
    # print(f'{xs.max()=}')
    # print(f'{xs_arr.min()=}')
    # print(f'{xs_arr.max()=}')


    if sigma is not None : 
        sigmas = sigma.loc[select]
        absolute_sigma = True
    else: 
        sigmas =  sigma  # because you can't .loc a None type
        absolute_sigma=False


    if (func == quad_w_min) :
        
        if not constrain_curvature: 
            G_lower=-np.inf
        else:
            G_lower = 0
        if not constrain_min: 
            T_0_lower=-np.inf
        else:
            T_0_lower = 0

        bounds_lower = (T_0_lower,-np.inf,G_lower)    # if the quadratic, force it to be positive curvature
    else:
        bounds_lower = -np.inf



    p0 = (xs.mean(),ys.min(),1e-6)
    
    
    popt,pcov = optimize.curve_fit(func, xs_arr,ys_arr, sigma=sigmas,bounds=(bounds_lower,np.inf),absolute_sigma=absolute_sigma,p0=p0 ) # absolute sigma set with Kwargs. TODO Check if this should be set true by default. Should probably...


    return popt,pcov

def get_poly_min(fit,xh=None,xl=None):
    ''' 
        For a given set of polynomial coefficients, calculate the location of the minimum

        Looks only for global minimum for points in the range

        Finds the minima from the coefficients

        Ensure that only those that are minima are added
        
        


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
    

 

def find_min(y,x, n,  xl=None,xh=None,sigma=None,err = False,use_alt_quad=True,reject_extrap=True):
    '''
        Find the minimum of one series with respect to another, using polynomial fittings

        interp_grid = grid to use for interpolation

        Interpolate with polynomials??? 

        y = data

        x = x data

        n = polynomial order to use TODO : if second-order, use the alternative fitting function, with T_0 as a parameter



        Maybe also validate that the value at the edge is not the same as the value of the minimum -  if minimum is at edge, suggests that there isn't a true inversion.

    Optional inputs:

        xmin, xmax = range to fit over


    '''

    if not xh: xh = np.max(x)
    if not xl: xl = np.min(x)

    if n != 2 and not use_alt_quad: # use the special form for a quadratic G(x-x0)^2 +y_0
        fit = ranged_poly_fit(y,x,n=n,xl=xl,xh=xh )
        try:
            x_min,y_min = get_poly_min(fit,xl=xl,xh=xh)
        except ValueError:
            x_min,y_min = (np.nan, np.nan)
    else:
        try: 
            fit,pcov  = alternate_fit(y,x,xl=xl,xh=xh,sigma=sigma,constrain_curvature=True,func=quad_w_min)
        except RuntimeError:
            warnings.warn('Could not fit ')
            fit = np.repeat(np.nan,3)
            pcov = np.diag(np.repeat(np.nan,3))  ### identity with nan on diagonals
            # RuntimeError Occurs if there is no
        x_min=fit[0]
        y_min = fit[1]
    

    if (x_min < xl) | (x_min > xh) and reject_extrap:
        x_min = np.nan
        y_min = np.nan

    
    #xs = np.linspace(xl,xh,grid)


    if err and n==2 and sigma is not None:
        return x_min,y_min,fit, np.sqrt(np.diag(pcov))   ### return error bars, only if error of data points is accurately accounted for

    return x_min,y_min, fit


def find_phi_min(chunk,n,potential_name = 'phi_tot', temp_name = 'temp',sigma=None,temp_range = 300,temp_centre = None,
                 show_plots = False,grid=100000,verbose = False,plot_markers = 10, use_alt_quad=True,err=False):
    '''
        chunk: th.Chunk  = contains the data frame with the data to fit to, potential and temperature

        n: int = order of polynomial for fit

        grid : int = number of grid points to use for plotting   
    
    '''
    
    
    temps = chunk.data[temp_name]
    phis = chunk.data[potential_name]

    if temp_centre is None and (temp_range is not None):
        if verbose: print(f'temp_range around min {temp_range}' )
        Tl,Th = choose_temp_range(chunk.data, ptp = temp_range,pot_name = potential_name, temp_name=temp_name)
    elif temp_range is not None:
        if verbose: print(f'temp_range {temp_range} around temp_centre {temp_centre}' )
        Tl,Th = (temp_centre - temp_range/2,temp_centre+temp_range/2)
    else: 
        if verbose: print(f'Fitting to the full range' )
        Tl,Th = (temps.min(),temps.max())
    
    # don't over extend the range, otherwise and incorrect minima will be found!!!!
    print(f'{temps.max()=},{Th=}')
    print(f'{Th > temps.max()=}')
    print(f'{Tl < temps.min()=}')
    print(f'{temps.min()=},{Tl=}')
    if Th > temps.max(): 
        Th = temps.max()
        print('Changed to max possible temp')
    if Tl < temps.min(): 
        Tl = temps.min()
        print('Changed to minimum possible temp')

    if verbose: print(f'Fitting a {n}-order polynomial between T = {Tl:.3f},{Th:.3f} K.')

    if not err:
        T_min,phi_min,fit =  find_min(phis,temps,n,sigma=sigma,xl=Tl,xh=Th,use_alt_quad=use_alt_quad)
    else:
        T_min,phi_min,fit,err_params = find_min(phis,temps,n,sigma=sigma,xl=Tl,xh=Th,use_alt_quad=use_alt_quad,err=err)



    if verbose: print(f'Minimum found at T = {T_min:.3f} ')

    if show_plots:    
        Ts = np.linspace(Tl,Th,grid)
        if use_alt_quad and n == 2:
            plt.plot(Ts,quad_w_min(Ts,*fit),label=r'$G(T-T_0)^2+\phi_0$')
        else:
            plt.plot(Ts,np.polyval(fit,Ts),c='b',label =f'{n} order fit ',ls = '--')
        plt.plot(temps,phis,'r.' ,markevery = plot_markers,label='data')
        plt.plot(T_min,phi_min,'k.')
        plt.xlabel(r'$T$/K')
        plt.ylabel(r'$\phi$/V')
        plt.legend()
        plt.show()

    if err:
        return T_min,phi_min, fit,err_params
    else:
        return T_min,phi_min,fit

def find_x_intercept(y,x,offset=0, xmin=None,xmax=None,interp_grid = None, interp_modde = 'linear'):
    
    '''
        Find the x intercept of a set of data with a finite grid.

        Uses a scipy tool to find the closest match to zero(+offset), then the corresponding finite value of x

        can restrict to a range to prevent finding fake minima, for example noise in the data giving a minima that is not true??

        interp grid is there to interpolate if need be. If used will interpolate y data between xmin and xmax with the specified number of points

    '''
    # If not specified, set to maximum and minimum value of range
    if  xmin is None: xmin = np.min(x)
    if  xmax is None: xmax = np.min(x)
    
    if interp_grid:
        # if interpolation is desired, do it, else, don't 
        x_new = np.linspace(xmin,xmax,interp_grid)
        f = interpolate.interp1d(x,y)
        y_new = f(x_new)
    else:
        x_new = x
        y_new = y

    raise NotImplementedError("The method of finding from an interpolated intercept is not yet implemented")




def profile_calculating(chunk:Potential,w = 5,sigma = 3,win_type = None, trim_w = 5,bw = None,show_plots = False,recalc_post_trim = False,direct=False,correct = ['cos_theta']):
    ''' 
        Does a lot. Prepares the profiles and calculates properties that are ratios or derivatives of others.

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

    # smoothed and unfolded
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
            # Make properties that have signs that depend on the side of the simulation bopx have the correct signs 
            chunk_trimmed.parity('E_tot')
            chunk_trimmed.parity('E_Q_zz')
            chunk_trimmed.parity('E_P_z')
            chunk_trimmed.parity('temp_grad')
            chunk_trimmed.parity('cos_theta')
            chunk_trimmed.parity('P_z')

        except:
            pass

    chunk_folded = Potential(chunk_trimmed.fold_and_ave())


    return chunk_smoothed, chunk_trimmed, chunk_folded
