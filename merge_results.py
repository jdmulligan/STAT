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
        
        # Emulator validation: Store lists of true RAA, emulator RAA at each holdout point
        SystemCount = len(self.AllData["systems"])
        true_raa_aggregated = [[] for i in range(SystemCount)]
        emulator_raa_aggregated = [[] for i in range(SystemCount)]

        # Store a list of the chi2 of the holdout residual
        self.avg_residuals = []

        # Store list of closure test result
        qhat_closure_list = []
    
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
                
                [[true_raa_aggregated[i].append(raa) for raa in true_raa[i]] for i in range(SystemCount)]
                [[emulator_raa_aggregated[i].append(raa) for raa in emulator_raa[i]] for i in range(SystemCount)]
                
                # Closure test
                T_array = result_dict['T_array']
                qhat = result_dict['qhat_truth']
                mean = result_dict['qhat_mean']
                qhat_closure = result_dict['qhat_closure']
                
                qhat_closure_list.append(qhat_closure)


        # Plot summary of holdout tests
        #self.plot_avg_residuals()
        self.plot_emulator_validation(true_raa_aggregated, emulator_raa_aggregated)

        # Plot summary of closure tests
        self.plot_closure_summary(T_array, qhat_closure_list)

    #---------------------------------------------------------------
    # Plot summary of closure tests
    #---------------------------------------------------------------
    def plot_closure_summary(self, T_array, qhat_closure_list):
    
        # qhat_closure_list is a list (per design point) of lists (of T values)
        # Generate a new list: average value per T
        qhat_closure_fraction = [1.*sum([qhat_list[i] for qhat_list in qhat_closure_list])/len(qhat_closure_list) for i,T in enumerate(T_array)]
        
        # Plot fraction of closure tests contained in 90% credible region
        plt.plot(T_array, qhat_closure_fraction, sns.xkcd_rgb['pale red'],
                 linewidth=2., label='Fraction of closure tests contained in 90% CR')
        plt.xlabel('T (GeV)')
        plt.ylabel('Fraction')
        
        ymin = 0.5
        ymax = 1.2
        axes = plt.gca()
        axes.set_ylim([ymin, ymax])
        
        # Draw legend
        first_legend = plt.legend(title=self.model, title_fontsize=15,
                                  loc='upper right', fontsize=12)
        ax = plt.gca().add_artist(first_legend)
        
        # Save
        plt.savefig('{}/Closure_Summary.pdf'.format(self.plot_dir, self.model), dpi = 192)
        plt.close('all')

    #---------------------------------------------------------------
    # Plot emulator validation
    #---------------------------------------------------------------
    def plot_emulator_validation(self, true_raa, emulator_raa):

        # Construct a figure with two plots
        plt.figure(1, figsize=(10, 6))
        ax_scatter = plt.axes([0.1, 0.13, 0.6, 0.8]) # [left, bottom, width, height]
        ax_residual = plt.axes([0.81, 0.13, 0.15, 0.8])

        # Loop through emulators
        SystemCount = len(self.AllData["systems"])
        for i in range(SystemCount):
    
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
            true_raa_i = np.array(true_raa[i])
            emulator_raa_i = np.array(emulator_raa[i])
            normalized_residual_i = np.divide(true_raa_i-emulator_raa_i, true_raa_i)

            # Draw scatter plot
            ax_scatter.scatter(true_raa_i, emulator_raa_i, s=1,
                               color=color, label=system_label)
            ax_scatter.set_xlabel(r'$R_{AA}^{true}$', fontsize=18)
            ax_scatter.set_ylabel(r'$R_{AA}^{emulator}$', fontsize=18)
            ax_scatter.legend(title=self.model, title_fontsize=22,
                              loc='upper left', fontsize=18, markerscale=10)
          
            # Draw normalization residuals
            max = 0.5
            bins = np.linspace(-max, max)
            ax_residual.hist(normalized_residual_i, color=color, histtype='step',
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
