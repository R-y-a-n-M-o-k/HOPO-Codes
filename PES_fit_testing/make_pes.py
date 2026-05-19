import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

def get_coeffs_orders(filename):
    """Obtain the linear coefficients of PES
       obtains associated orders, torsion argument for given coefficients
       filename is the name of the res file containing fit data"""
    
    params = []
    orders = []

    with open(filename, 'r') as file:
        #skip unnecessary lines 
        line = file.readline()
        while line[0] != 'f':
            line = file.readline()
    
        line_split = line.split()
        while line_split[0][0] == 'f':
            
            #get orders, convert to float 
            orders_line = line_split[1:-2]
            for j in range(len(orders_line)):
                orders_line[j] = float(orders_line[j])
            
            param = float(line_split[-1])
            
            #append into correct dipole component 
            params.append(param)
            orders.append(orders_line)
                
            line = file.readline()
            line_split = line.split()

    return (np.array(params), np.array(orders))

def get_non_linear_p(filename):
    """Obtain the non-linear coefficients of PES fit from 
       given residuals file, filename."""
    
    non_linear_params = []
    with open(filename, 'r') as file:
        #skip unnecessary lines 
        line = file.readline()
        while line[0] == '*':
            line = file.readline()
        
        line_split = line.split()
        #get nonlin_params 
        while len(line_split) == 3:
            non_lin_param = float(line_split[-1])
            non_linear_params.append(non_lin_param)

            line = file.readline()
            line_split = line.split()
    return np.array(non_linear_params)

def get_pes_v4(linear_params, nonlin_eq, orders):
    """computes the pes function for a given component direction,
       assumes fitting is done using the HPO2_v4 compiled fitting file.
       using input parameters (linear params, nonlin_eq) and powers
    """
    def pes(geometries):
        """Fitted pes function (R^6 -> R)
           - Assumes geometries is a n by 6 ndarray of input values, 
           - Assumes angles are given in degrees, (function converts to radians)
           Outputs value of potential energy for those given geometries.
        """
        pes_values = np.zeros(np.shape(geometries)[0])
        #set up variable arg to be modified to calculate pes values
        #start with input geometries
        arg = geometries.copy()
        nonlin_params = nonlin_eq.copy()
        
        #convert deg to rad, change angle poh
        arg[:, 3:] = arg[:, 3:] * np.pi/180
        nonlin_params[3:5] = nonlin_params[3:5]*np.pi/180
        
        #compute one sum-product term per loop
        for j in range(len(orders[:,0])):
            #initialise product parts
            term = np.zeros(np.shape(arg)[0])
            term.fill(linear_params[j])
            
            products = arg.copy()
            #compute morse_arguments
            products[:, :3] = (1 - np.exp(-1*nonlin_params[-3:]*
                                     (products[:, :3] - nonlin_eq[:3])
                                         ) 
                              )**orders[j,:3]
            
            #compute angle values
            products[:, 3] = (np.cos(products[:, 3]) 
                              - np.cos(nonlin_params[3])) **orders[j, 3]
            #this is flipped around
            products[:, 4] = (nonlin_params[4] - products[:, 4]) **orders[j, 4]
            #compute torsion
            products[:, 5] = np.cos(orders[j, 5]*products[:, 5])
            
            #compute term
            for k in range(6):
                term *= products[:,k]
                
            pes_values += term

        return pes_values  
    return pes

def get_pes_v6(linear_params, nonlin_eq, orders):
    """computes the pes function for a given component direction,
       assumes fitting is done using the HPO2_v4 compiled fitting file.
       using input parameters (linear params, nonlin_eq) and powers
    """
    def pes(geometries):
        """Fitted pes function (R^6 -> R)
           - Assumes geometries is a n by 6 ndarray of input values, 
           - Assumes angles are given in degrees, (function converts to radians)
           Outputs value of potential energy for those given geometries.
        """
        pes_values = np.zeros(np.shape(geometries)[0])
        #set up variable arg to be modified to calculate pes values
        #start with input geometries
        arg = geometries.copy()
        nonlin_params = nonlin_eq.copy()
        
        #convert deg to rad, change angle poh
        arg[:, 3:] = arg[:, 3:] * np.pi/180
        nonlin_params[3:5] = nonlin_params[3:5]*np.pi/180
        
        #compute one sum-product term per loop
        for j in range(len(orders[:,0])):
            #initialise product parts
            term = np.zeros(np.shape(arg)[0])
            term.fill(linear_params[j])
            
            products = arg.copy()
            #compute morse_arguments
            products[:, :3] = (1 - np.exp(-1*nonlin_params[-3:]*
                                     (products[:, :3] - nonlin_eq[:3])
                                         ) 
                              )**orders[j,:3]
            
            #compute angle values
            products[:, 3] = (np.cos(products[:, 3]) 
                              - np.cos(nonlin_params[3])) **orders[j, 3]
            
            #this is flipped around
            products[:, 4] = ( np.cos(np.pi - nonlin_params[4]) - np.cos(np.pi - products[:, 4]) )**orders[j,4]

            #compute torsion
            products[:, 5] = np.cos(orders[j, 5]*products[:, 5])
            
            #compute term
            for k in range(6):
                term *= products[:,k]
                
            pes_values += term

        return pes_values  
    return pes

#does not work yet 
def get_pes_v7_sin_sin(linear_params, nonlin_eq, orders):
    """computes the pes function for a given component direction,
       assumes fitting is done using the HPO2_v7 compiled fitting file.
       using input parameters (linear params, nonlin_eq) and powers
    """
    def pes(geometries):
        """Fitted pes function (R^6 -> R)
           - Assumes geometries is a n by 6 ndarray of input values, 
           - Assumes angles are given in degrees, (function converts to radians)
           Outputs value of potential energy for those given geometries.
        """
        pes_values = np.zeros(np.shape(geometries)[0])
        #set up variable arg to be modified to calculate pes values
        #start with input geometries
        arg = geometries.copy()
        nonlin_params = nonlin_eq.copy()
        
        #convert deg to rad, change angle poh
        arg[:, 3:] = arg[:, 3:] * np.pi/180
        nonlin_params[3:5] = nonlin_params[3:5]*np.pi/180
        
        #swap coord 1 and coord 2 (for v6 and 7 only)
        #arg[:, [0, 1]] = arg[:, [1, 0]]
        
        #compute one sum-product term per loop
        for j in range(len(orders[:,0])):
            
            #initialise product parts
            term = np.zeros(np.shape(arg)[0])
            term.fill(linear_params[j])
            
            products = arg.copy()
            #compute morse_arguments
            products[:, :3] = (1 - np.exp(-1*nonlin_params[-3:]*
                                     (products[:, :3] - nonlin_eq[:3])
                                         ) 
                              )**orders[j,:3]
            
            if orders[j, -1] != 0:
                #compute angle values, sin damp
                products[:, 3] = np.sin(products[:, 3])*(np.cos(products[:, 3]) 
                                  - np.cos(nonlin_params[3])) **orders[j, 3]
                #this is flipped around
                products[:, 4] = np.sin(products[:, 4])*(np.cos(np.pi - nonlin_params[4]) - np.cos(np.pi - products[:, 4]))**orders[j, 4]

            else:
                #compute angle values
                products[:, 3] = (np.cos(products[:, 3]) 
                                  - np.cos(nonlin_params[3])) **orders[j, 3]
                #this is flipped around
                products[:, 4] = (np.cos(np.pi - nonlin_params[4]) - np.cos(np.pi - products[:, 4])) **orders[j, 4]
                                  
            #compute torsion
            products[:, 5] = np.cos(orders[j, 5]*products[:, 5])
            
            #compute term
            for k in range(6):
                term *= products[:,k]
                
            pes_values += term

        return pes_values  
    return pes
        
        