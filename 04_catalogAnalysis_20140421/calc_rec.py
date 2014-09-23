"""
This script calculates Gutenberg-Richter recurrence parameters
(a and b values) from an earthquake catalogue. Recurrence parameters are
calculated using several methods:
    1. Least Squares
    2. Maximum Likelihood (Aki, 1965)
    3. Assuming b = 1

Results are plotted for both straightline fits and bounded
Gutenberg-Richter curves. Also plotted is the curve that would result
assuming a b-value of 1.

This assumes that a subset of a catlogue has already been created based on
depth and location - i.e. this is not a tool for exploring the data.

Usage: python recurrence_from_catalog.py <input_file>
<minimum magnitude = minimum magnitude in catalogue>
<maximum magnitude = maximum magnitude in catalogue + 0.1> <interval = 0.1>

Arguments:
Required:
Input file: This is the file that contains the earthquake catalogue. It
            is expected to be in sv format with a one-line header. It is expectd
            that the year of the earthquake will be in the 3rd column,
            and the magnitude in 6th column.
Optional:
minimum magnitude: The minumum magnitude for which the catalogue is complete.
            Defaults to the minimum magnitude within the catalogue
maximum magnitude: The maxumum magnitude expected for the source zone.
            Defaults to the maximum magnitude in the catalogue plu 0.1 magnitude
            units.
interval: The size of the bins used in generating a histogram of earthquake
            occurrence. Defaults to 0.1 magnitude units.    

Ref: Kramer, S.L. 1996. Geotechnical Earthquake Engineering, p123.

Creator: Jonathan Griffin, Australia-Indonesia Facility for Disaster Reduction
Created: 23 August 2010
"""

import sys,os
import numpy as np
from scipy import stats
import matplotlib
from matplotlib import pylab as py

def calc_recurrence(infile, min_mag = None, max_mag = None, interval = 0.05):

    """
    This function reads an earthquake catalogue file and calculates the
    Gutenberg-Richter recurrence parameters using both least squares and
    maximum likelihood (Aki 1965) approaches.

    Results are plotted for both straightline fits and bounded
    Gutenberg-Richter curves. Also plotted is the curve that would result
    assuming a b-value of 1.

    Funtion arguments:

        infile: file containing earthquake catalogue
                Expected input file format: csv
                One header line
                Year of earthquake in 3rd column, magnitude in 6th column.
        min_mag: minimum magnitude for which data will be used - i.e. catalogue
                completeness
        max_mag: maximum magnitude used in bounded G-R curve. If not specified,
                defined as the maximum magnitude in the catlogue + 0.1 magnitude
                units.
        interval: Width of magnitude bins for generating cumulative histogram
                of earthquake recurrence. Default avlue 0.1 magnitude units.  
    
    """

    
    magnitudes = []
    #years = []

    # If minimum magnitude is not specified, read all magnitudes
    if min_mag is not None:
        pass
    else:
        min_mag = -1.0
        
    ###########################################################################
    # Read data
    ###########################################################################
    
    for row in range(len(infile)):
        mag = infile[row]
        if (mag >= min_mag):# and (depth <= max_depth):
            magnitudes.append(mag)
        #years.append(float(row[2]))
    
    # If minimum magnitude is not specified default value to minimum in catalogue
    if min_mag == -1.0:
        min_mag = min(magnitudes)
    # If maximum magnitude is not specified default value to maximum in catalogue
    if max_mag is not None:
        pass
    else:
        max_mag = max(magnitudes) + 0.1

    num_eq = len(magnitudes)
    print 'Minimum magnitude:', min_mag
    print 'Total number of earthquakes:', num_eq
    #num_years = max(years)-min(years)
    #annual_num_eq = num_eq/num_years
    #print 'Annual number of earthquakes greater than Mw', min_mag,':', \
    #annual_num_eq
    print 'Maximum catalog magnitude:', max(magnitudes)
    print 'Mmax = ', max_mag
    max_mag_bin = max(magnitudes) + 0.15
    
    # Magnitude bins
    bins = np.arange(min_mag, max_mag_bin, interval)
    # Magnitude bins for plotting - we will re-arrange bins later
    plot_bins = np.arange(min_mag, max_mag, interval)

    ###########################################################################
    # Generate distribution
    ###########################################################################
    # Generate histogram
    hist = np.histogram(magnitudes,bins=bins)

    # Reverse array order
    hist = hist[0][::-1]
    bins = bins[::-1]

    # Calculate cumulative sum
    cum_hist = hist.cumsum()
    # Ensure bins have the same length has the cumulative histogram.
    # Remove the upper bound for the highest interval.
    bins = bins[1:]

    ## Get annual rate
    #cum_annual_rate = cum_hist/num_years

    #new_cum_annual_rate = []
    #for i in cum_annual_rate:
    #    new_cum_annual_rate.append(i+1e-20)

    # Take logarithm
    #log_cum_sum = np.log10(new_cum_annual_rate)
    
    ###########################################################################
    # Fit a and b parameters using a varity of methods
    ###########################################################################
    #print cum_hist
    #print bins
    '''
    # Fit a least squares curve
    b,a = np.polyfit(bins, log_cum_sum, 1)
    print 'Least Squares: b value', -1. * b, 'a value', a
    alpha = np.log(10) * a
    beta = -1.0 * np.log(10) * b

    # Maximum Likelihood Estimator fitting
    # b value
    b_mle = np.log10(np.exp(1)) / (np.mean(magnitudes) - min_mag)
    beta_mle = np.log(10) * b_mle
    print 'Maximum Likelihood: b value', b_mle

    
    ###########################################################################
    # Generate data to plot results
    ###########################################################################

    # Generate data to plot least squares linear curve
    # Calculate y-intercept for least squares solution
    yintercept = log_cum_sum[-1] - b * min_mag
    ls_fit = b * plot_bins + yintercept
    log_ls_fit = []
    for value in ls_fit:
        log_ls_fit.append(np.power(10,value))

    # Generate data to plot bounded Gutenberg-Richter for LS solution
    numer = np.exp(-1. * beta * (plot_bins - min_mag)) - \
            np.exp(-1. *beta * (max_mag - min_mag))
    denom = 1. - np.exp(-1. * beta * (max_mag - min_mag))
    ls_bounded = annual_num_eq * (numer / denom)
        
    # Generate data to plot maximum likelihood linear curve
    mle_fit = -1.0 * b_mle * plot_bins + 1.0 * b_mle * min_mag + np.log10(annual_num_eq)
    log_mle_fit = []
    for value in mle_fit:
        log_mle_fit.append(np.power(10,value))

    # Generate data to plot bounded Gutenberg-Richter for MLE solution
    numer = np.exp(-1. * beta_mle * (plot_bins - min_mag)) - \
            np.exp(-1. *beta_mle * (max_mag - min_mag))
    denom = 1. - np.exp(-1. * beta_mle * (max_mag - min_mag))
    mle_bounded = annual_num_eq * (numer / denom)

    # Compare b-value of 1
    fit_data = -1.0 * plot_bins + min_mag + np.log10(annual_num_eq)
    log_fit_data = []
    for value in fit_data:
        log_fit_data.append(np.power(10,value))

    ###########################################################################
    # Plot the results
    ###########################################################################

    # Plotting
    fig = py.scatter(bins, new_cum_annual_rate, label = 'Catalogue')
    ax = py.gca()
    ax.plot(plot_bins, log_ls_fit, c = 'r', label = 'Least Squares')
    ax.plot(plot_bins, ls_bounded, c = 'r', linestyle ='--', label = 'Least Squares Bounded')
    ax.plot(plot_bins, log_mle_fit, c = 'g', label = 'Maximum Likelihood')
    ax.plot(plot_bins, mle_bounded, c = 'g', linestyle ='--', label = 'Maximum Likelihood Bounded')
    ax.plot(plot_bins, log_fit_data, c = 'b', label = 'b = 1')
    
    #ax.plot(bins, ls_fit2, c = 'k')
    ax.set_yscale('log')
    ax.legend()
    ax.set_ylim([min(new_cum_annual_rate) * 0.1, max(new_cum_annual_rate) * 10.])
    ax.set_xlim([min_mag - 0.5, max_mag + 0.5])
    ax.set_ylabel('Annual probability')
    ax.set_xlabel('Magnitude')
    py.show()
    '''
    return cum_hist[::-1], bins[::-1]