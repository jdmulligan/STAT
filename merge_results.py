'''
Class to steer Bayesian analysis and produce plots.
'''

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy

import os
import sys
import pickle
import argparse

from src.design import Design
from src import emulator, mcmc, init

import run_analysis_base

################################################################
class MergeResults(run_analysis_base.RunAnalysisBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file, model, output_dir, **kwargs):

        # Initialize base class
        super(MergeResults, self).__init__(config_file, model, output_dir, **kwargs)
        
        self.output_dir_holdout = os.path.join(self.output_dir_base, '{}/holdout'.format(model))
        self.plot_dir = os.path.join(self.output_dir_base, model)
        
    #---------------------------------------------------------------
    # Run analysis
    #---------------------------------------------------------------
    def run_analysis(self):
  
        # Initialize data and model from files
        self.initialize()
        
        # Initialize pickled config settings
        init.Init(self.workdir).Initialize(self)
    
        # Store a list of the chi2 of the holdout residual
        self.avg_residuals = []

        n_design_points = len(next(os.walk(self.output_dir_holdout))[1])
        print('iterating through {} results'.format(n_design_points))
        for i in range(0, n_design_points):
        
            # Load pkl file of results
            result_path = os.path.join(self.output_dir_holdout, '{}/result.pkl'.format(i))
            with open(result_path, 'rb') as f:

                result_dict = pickle.load(f)

                # Holdout test
                true_raa = result_dict['true_raa']
                emulator_raa = result_dict['emulator_raa']
                
                # Closure test
                T_array = result_dict['T_array']
                qhat = result_dict['qhat']                          # Truth
                mean = result_dict['mean']                          # Extracted mean
                qhat_posteriors = result_dict['qhat_posteriors']    # Extracted posteriors

            # Plot summary of holdout tests
            #self.plot_avg_residuals()
            #self.plot_emulator_validation()

            # Plot summary of closure tests
            # ...

    #---------------------------------------------------------------
    # Plot emulator validation
    #---------------------------------------------------------------
    def plot_emulator_validation(self):

        # Construct a figure with two plots
        plt.figure(1, figsize=(10, 6))
        ax_scatter = plt.axes([0.1, 0.13, 0.6, 0.8]) # [left, bottom, width, height]
        ax_residual = plt.axes([0.81, 0.13, 0.15, 0.8])

        # Loop through emulators
        for i in range(self.SystemCount):
    
            system = self.AllData['systems'][i]
            if 'AuAu' in system:
                system_label = 'Au-Au 200 GeV'
            else:
                if '2760' in system:
                    system_label = 'Pb-Pb 2.76 TeV'
                elif '5020' in system:
                    system_label = 'Pb-Pb 5.02 TeV'
            
            color = sns.color_palette('colorblind')[i]

            # Get RAA points
            true_raa = np.array(self.true_raa[i])
            emulator_raa = np.array(self.emulator_raa[i])
            normalized_residual = np.divide(true_raa-emulator_raa, true_raa)

            # Draw scatter plot
            ax_scatter.scatter(true_raa, emulator_raa, s=1,
                               color=color, label=system_label)
            ax_scatter.set_xlabel(r'$R_{AA}^{true}$', fontsize=18)
            ax_scatter.set_ylabel(r'$R_{AA}^{emulator}$', fontsize=18)
            ax_scatter.legend(title=self.model, title_fontsize=22,
                              loc='upper left', fontsize=18, markerscale=10)
          
        # Draw normalization residuals
        max = 0.5
        bins = np.linspace(-max, max)
        ax_residual.hist(normalized_residual, color=color, histtype='step',
                         orientation='horizontal', linewidth=3, density=True, bins=bins)
        ax_residual.set_ylabel(r'$\left(R_{AA}^{true} - R_{AA}^{emulator}\right) / R_{AA}^{true}$',
                               fontsize=16)
          
        plt.savefig('{}/EmulatorValidation.pdf'.format(self.plot_dir))
        plt.close('all')

##################################################################
if __name__ == '__main__':
  
    # Define arguments
    parser = argparse.ArgumentParser(description='Jetscape STAT analysis')
    parser.add_argument('-o', '--outputdir', action='store',
                        type=str, metavar='outputdir',
                        default='./STATGallery')
    parser.add_argument('-c', '--configFile', action='store',
                        type=str, metavar='configFile',
                        default='analysis_config.yaml',
                        help='Path of config file')
    parser.add_argument('-m', '--model', action='store',
                        type=str, metavar='model',
                        default='LBT',
                        help='model')

    # Parse the arguments
    args = parser.parse_args()
    
    print('')
    print('Configuring MergeResults...')
    
    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)
    
    analysis = MergeResults(config_file = args.configFile, model=args.model,
                            output_dir=args.outputdir)
    analysis.run_model()
