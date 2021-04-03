from numpy.lib.function_base import gradient
import thermotar as th

from thermotar.utils import df_utils

from scipy.integrate import cumtrapz, trapz


import numpy as np
import pandas as pd

class Potential(th.Chunk):

    # list of methods that can be inherited in replicator instances to apply to sub dfs.
    # these need to act on dfs.
    _inheritable = ['calculate_potentials']

    def calculate_potentials(self,Ps = ['P_x','P_y','P_z'],
               Qs = ['Q_xx','Q_yy','Q_zz','Q_xy','Q_xz','Q_yz'],charge = 'charge_dens',
              coords = 'coord',verbose = False, pot_corr = True,alt_corr = False, integrate_reverse = False):
        '''
            TODO : finish this for easier calculation of the profiles
            Calculate the electrostatic potential contributions for all components

            TODO: Correct the field contributions or calculate from potential.
            
            coords -> tuple of strings or string of names of coordinate coloums to intgrate against, assumes angstrom
        
            pot_coor : bool
                    Assume net charge neutral and thus that potential is zero at both box ends, and remove excess charge
                    Currently only applied to the total potential
        '''
        
        
        from scipy.constants import elementary_charge, epsilon_0
        
        epsilon_0_AA = epsilon_0 * 1e-10 # convert F m^-1 -> F AA^-1
        
        
        # all P, Q and rho are provided in units of C per Angstrom^-n, where n depends on the system
        
        # add the appropiate column names
        
        field_P_names = ['E_' + P for P in Ps]
        pot_P_names = ['phi_' + P for P in Ps]
        field_Q_names = ['E_'+ Q for Q in Qs]
        pot_Q_names = ['phi_'+ Q for Q in Qs]
        
        cols_to_add = ['E_tot'] + field_P_names + field_Q_names + ['phi_tot'] + pot_P_names + pot_Q_names
        if pot_corr:
            cols_to_add.append('phi_uncorrected')
            cols_to_add.append('phi_P_uncorrected')
        
        #print(df.head())
        
        self.data = df_utils.new_cols(self.data,cols_to_add)
        
        #print(df.head())
        
        # select temperature and rep
        if integrate_reverse:
            sub_df = self.data.sort_values(coords,ascending=False)
        else:
            sub_df = self.data  
        
        #print(sub_df.head())
        
        coordinates = sub_df[coords]
        #print(coordinates)
        
        #E_pol = 
        
        # all integrations specifiy initial = 0 to ensure the results are the same length
        
        # double integral of charge density
        sub_df['E_tot'] = cumtrapz(sub_df[charge],coordinates,initial=0)/epsilon_0_AA #V per angstrom
        # correct by subtracting the average.
        if pot_corr and not alt_corr:
            sub_df['E_tot'] -= sub_df['E_tot'].mean()
        if alt_corr:
            aveE = trapz(sub_df['E_tot'],coordinates)/ (coordinates.max()-coordinates.min())
            #print(aveE)

            sub_df['E_tot'] -= aveE

        
        sub_df['phi_tot'] = -1*cumtrapz(sub_df['E_tot'],coordinates,initial=0)
        # not needed because average field has already been removed
        # if pot_corr:
        #     sub_df['phi_uncorrected'] = sub_df['phi_tot']
        #     # get last value of this
        #     correction = -1 * sub_df['phi_tot'].iloc[-1]*sub_df[coords]/(sub_df[coords].max()-sub_df[coords].min())
        #     sub_df['phi_tot'] += correction
        
        
        
        
        # difference of P_i(z)-P_i(0)
        # assumes first coord is z = 0
        # this assumption could be removed
        sub_df[field_P_names] = (sub_df[Ps].iloc[0]-sub_df[Ps])/epsilon_0_AA
        
        
        sub_df[pot_P_names] = pd.DataFrame(cumtrapz(sub_df[Ps],coordinates,initial=0,axis=0))/epsilon_0_AA
        if verbose:
            print(sub_df['phi_P_x'])
            
            
        Q_grad = np.gradient(sub_df[Qs],sub_df[coords],axis=0)
        
        #print(Q_grad.shape)
        
        sub_df[field_Q_names] = pd.DataFrame(Q_grad-Q_grad[0])/epsilon_0_AA
        if verbose:
            print(Q_grad-Q_grad[0])
        sub_df[pot_Q_names] = -(sub_df[Qs]-sub_df[Qs].iloc[0])/epsilon_0_AA
        if pot_corr:
            for col in pot_Q_names:

                sub_df[col] -= (sub_df[col].iloc[-1])*sub_df[coords]/(sub_df[coords].max()-sub_df[coords].min()) # remove the last potential so it isn't slopey
            sub_df[field_Q_names] = -1*np.gradient(sub_df[pot_Q_names],sub_df[coords],axis=0) # derivative of the corrected potential field
        if pot_corr:
            # TODO add corrections for the other components
            sub_df['phi_P_uncorrected'] = sub_df['phi_P_z']
            # get last value of this
            for col in pot_P_names:
                correction = -1 * sub_df[col].iloc[-1]*sub_df[coords]/(sub_df[coords].max()-sub_df[coords].min())
                sub_df[col] += correction
            # recalucalte the potential gradient, now from the potential
            sub_df[field_P_names] = -1*np.gradient(sub_df[pot_P_names],sub_df[coords],axis=0)
        # raise the new cols, so they can be accessed with obj.colname notation
        for col in cols_to_add:
            setattr(self.__class__, col, df_utils.raise_col(self,col))
    
    
        if integrate_reverse:
            self.data = sub_df.sort_values(coords,ascending=True) #sort back again!

            
            
            
