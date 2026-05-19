import numpy as np
import scipy.integrate as integrate
from scipy.stats import rv_continuous
import matplotlib.pyplot as plt
import pandas as pd

#define required constants, values 
coords_files = {1:'pvqz_rpo1.res', 2:'pvqz_rpo2.res', 3:'pvqz_roh.res',
                4:'pvqz_aopo.res', 5:'pvqz_apoh.res', 6:'pvqz_tau.res'}
xlabels = [r'$r_{1}$ (Å)',r'$r_{2}$ (Å)',r'$r_{3}$ (Å)', r'$\alpha$ (°)', r'$\beta$ (°)', r'$\tau$ (°)']
coordinate_lower_bounds = [1.20, 1.25, 0.69, 72, 50, 0]
coordinate_upper_bounds = [2.10, 3.15, 1.90, 180, 180, 180]
#bounds up to roughly 1D 30,000cm^-1)
lower_bounds = {1:1.20, 2:1.25, 3:0.69, 4:72.0, 5:50.0, 6:0.000000 }
upper_bounds = {1:2.10, 2:3.15, 3:1.90, 4:180, 5:180.000000, 6:180.000000}

#old lower bounds (up to 1D 20,000cm^-1)
#lower_bounds = {1:1.248331, 2:1.314292, 3:0.727982, 4:78.069932, 5:57.477880, 6:0.000000 }
#upper_bounds = {1:1.888331, 2:2.354292, 3:1.527982, 4:168.597264, 5:180.000000, 6:180.000000}

#factors = {1:0.001, 2:0.0008, 3:0.001, 4:0.001, 5:0.001, 6:0.0003} #larger means more concentrated 
#factors = {1:0.0005, 2:0.0004, 3:0.0005, 4:0.0005, 5:0.0005, 6:0.0003}
factors = {1:0.0002, 2:0.0002, 3:0.0002, 4:0.0001, 5:0.0003, 6:0.0003}

def get_coeffs_powers(filename):
    """Obtain the linear coefficients of 1D PES curves
       obtains associated powers for given coefficients
       filename is the name of the res file containing fit data"""
    
    params = []
    powers = []
    with open(filename, 'r') as file:
        #skip unnecessary lines 
        for i in range(12):
            next(file)
        
        #extract parameters
        line = file.readline()
        while line[0].isalpha():
            
            #extract linear param from line
            linesplit = line.split()
            parameter = float(linesplit[-1])
            params.append(parameter)
            
            #extract power val from line
            line_numbers = linesplit[1:-2]
            for number in line_numbers:
                if int(number) != 0: 
                    powers.append(int(number))
            if line_numbers == ['0']*6:
                powers.append(0)
                
            line = file.readline()
    
    return np.array(params), np.array(powers)

def get_nonlin_eq(filename, coordinate):
    """obtains equilibrium parameters for given coordinate
       eg. (r_eq - stretch value at minima)
       input coordinate is an integer from 1 to 5"""
    
    with open(filename, 'r') as file:
        #skip unnecessary lines 
        for i in range(4):
            next(file)
            
        for i in range(coordinate):
            line = file.readline()
        parameter = float(line.split()[-1])
        file.close()
        
        return parameter

def get_stretch_factor(filename, coordinate):
    """obtains nonlinear params, a, for the stretch coordinates
       coordinate an integer from 1 to 3 """
    
    with open(filename, 'r') as file:
    #skip unnecessary lines 
        for i in range(9):
            next(file)

        for i in range(coordinate):
            line = file.readline()
        parameter = float(line.split()[-1])
    file.close() 

    return parameter


######################
######################



def morse_generator(r_eq, a):
    """creates a morse function type with the given
       nonlinear parameters"""
    def morse(x):
        return 1 - np.exp(-1*a*(x - r_eq))
    
    return morse

def polynomial_generator(x_eq):
    """Assumes x and x_eq are in degrees"""
    #convert degrees to radians
    def poly(x):
        return (x - x_eq)*np.pi/180
    return poly

def cosine_generator(x_eq):
    """Assumes x and x_eq are in degrees"""
    def cos_diff(x):
        #convert degrees to radians
        return np.cos(x*np.pi/180) - np.cos(x_eq*np.pi/180)
    
    return cos_diff

def cosine_arg(x, arg):
    return np.cos(arg*x*np.pi/180)

def power_series(params, powers, function):
    """return a function-type object which is the
       power series of given function"""
    def power_function(x):
        series = 0
        for param, power in zip(params, powers):
            series += param*(function(x)**power)
        return series
    
    return power_function

def fourier_series(params, args, function):
    """return a function-type object which is the
       power series of given function
       args are values which go into argument of function"""
    def fourier_function(x):
        series = 0
        for param, arg in zip(params, args):
            series += param*function(x, arg)
        return series
    
    return fourier_function

def exponential_dist(pes_series, factor, lower_bound, upper_bound):
    """Creates a distribution function from the fitted pes curves, e^(-factor*U) where 
       U is the PES fitted curve. Not normalized
    
       pes_series is a function made by the power or fourier series functions
       factor is scalar to multiply in the exponent argument
       lower_bound is lower bound of the distribution, 
       upper_bound is upper bound of the distribution"""
    
    #create exponential weighted distribution with pes curve
    def exp_minus(x):
        return np.exp(-1*factor*pes_series(x))
    
    #normalize distribution
    norm = integrate.quad(exp_minus, lower_bound, upper_bound)[0]
    def pes_dist(x):
        return np.exp(-1*factor*pes_series(x))/norm
    
    return pes_dist

def create_pes_dist(distribution, lower_bound, upper_bound):
    """creates scipy dsitribution class object from the 
       input distribution function"""
    
    class pes_dist(rv_continuous):
        def _pdf(self,x):
            return distribution(x)
        
    new_distribution = pes_dist(a=lower_bound, b=upper_bound)
    return new_distribution

#######################
#######################

def get_pes(coordinate):
    """returns a function which calculates the 1D PES of a given coordinate"""
    #get parameters for pes curve
    params, powers = get_coeffs_powers(coords_files[coordinate])
    if coordinate in [1,2,3]:
        #get nonlin params, form pes function
        r_eq = get_nonlin_eq(coords_files[coordinate], 
                         coordinate)
        a = get_stretch_factor(coords_files[coordinate], 
                           coordinate)
        morse_arg = morse_generator(r_eq, a)
        pes = power_series(params, 
                           powers, 
                           morse_arg)
    
    elif coordinate in [4,5]:
        #get nonlin params, form pes function
        r_eq = get_nonlin_eq(coords_files[coordinate], 
                             coordinate)
        if coordinate == 4: 
            angle_arg = cosine_generator(r_eq)
            pes = power_series(params, powers, angle_arg)
        else: 
            angle_arg = polynomial_generator(r_eq)
            pes = power_series(params, powers, angle_arg)
    else:
        #for coordinate 6 tau, form fourier series 
        pes = fourier_series(params, powers, cosine_arg)
    return pes

def get_distribution(coordinate):
    """Input coordinate (1 to 3), outputs distribution of given coordinate"""
    #get parameters for pes curve
    pes = get_pes(coordinate)
    #construct distribution
    density_function = exponential_dist(pes, 
                                        factors[coordinate], 
                                        lower_bounds[coordinate],
                                        upper_bounds[coordinate])
    distribution = create_pes_dist(density_function, 
                                   lower_bounds[coordinate],
                                   upper_bounds[coordinate])
    return distribution

def monte_carlo_HOPO(num_samples, savetxt = True, savename=None):
    """samples geometries from the distribution, returns txt file containing
       sampled geometries, to be used for ab initio calculation"""
    fig, ax = plt.subplots(2,3, figsize=(20,13))
    ax_flat = ax.flatten()
    plt.suptitle('Sampled Geometry Histograms', x=0.5, y=0.92, fontsize=20)
    sample_geometries = []
    points = [i for i in range(1,num_samples+1)]
    sample_geometries.append(points)
    
    #samples and plots distribution
    for i in range(1,7):
        distribution = get_distribution(i)
        samples = distribution.rvs(size=num_samples)
        sample_geometries.append(samples)
        print(f'Sampling for coordinate {i} done!')
        print()
        ax_flat[i-1].hist(samples, bins=30, alpha=0.6, color='b', label='Samples')
        ax_flat[i-1].set_xlabel(xlabels[i-1], size=20)
        ax_flat[i-1].tick_params(labelsize=15)
    ax_flat[0].set_ylabel('Count', fontsize = 20)
    ax_flat[3].set_ylabel('Count', fontsize = 20)
    
    #approximate energy distribution
    energy_vals = np.zeros(num_samples)
    for i in range(1,7):
        pes = get_pes(i)
        energy_vals += pes(sample_geometries[i]) #total E is sum of 1D energies. 
    plt.figure(figsize=(10,6))
    plt.title('Approximate Energy Distribution', fontsize=17)
    plt.hist(energy_vals, bins=30, alpha=0.5, color='purple', label='Approx. energies')
    plt.xlabel(r'Relative energy (cm$^{-1}$)', fontsize=17)
    plt.ylabel('Count', fontsize=17)
    plt.tick_params(labelsize = 15)
    
    #arrange for display
    sample_geometries = np.transpose(np.array(sample_geometries))
    
    #save geometries into txt file
    if savetxt:
        np.savetxt(savename, 
                   sample_geometries, 
                   fmt=('%d','%.6f','%.6f','%.6f','%.6f','%.6f','%.6f'), 
                   delimiter=' ')
    return sample_geometries

###########
#plot the geometry distribution
def plot_geometries(sampled_geometries):
    """plots geometries found by monte carlo sampler"""
    fig, ax = plt.subplots(2,2, figsize=(20,18))
    ax_flat = ax.flatten()
    
    #4 plots of different coordinates 
    for i in range(1,5):
        ax_flat[i-1].plot(sampled_geometries[:,i+1], 
                       sampled_geometries[:,i+2],
                       'o', 
                       markersize=1.5)
        ax_flat[i-1].set_xlabel(xlabels[i], size=20)
        ax_flat[i-1].set_ylabel(xlabels[i+1], size=20)
        ax_flat[i-1].tick_params(labelsize=15)
