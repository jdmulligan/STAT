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
import statistics

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
        T_qhat_closure_result_list = []
        T_qhat_closure_result_list2 = []
        T_qhat_closure_truth_list = []
        E_qhat_closure_result_list = []
        E_qhat_closure_result_list2 = []
        E_qhat_closure_truth_list = []
        theta_closure_list = []
        theta_closure_result_dict = {}
        theta_closure_result2_dict = {}
        for name in self.Names:
            theta_closure_result_dict[name] = []
            theta_closure_result2_dict[name] = []
    
        n_design_points = len(next(os.walk(self.output_dir_holdout))[1])
        print('iterating through {} results'.format(n_design_points))
        for i in range(0, n_design_points):
        
            # Load pkl file of results
            result_path = os.path.join(self.output_dir_holdout, '{}/result.pkl'.format(i))
            if not os.path.exists(result_path):
                print('Warning: {} does not exist'.format(result_path))
                continue
            with open(result_path, 'rb') as f:

                result_dict = pickle.load(f)

                # Holdout test
                true_raa = result_dict['true_raa']
                emulator_raa = result_dict['emulator_raa']
                
                [[true_raa_aggregated[i].append(raa) for raa in true_raa[i]] for i in range(SystemCount)]
                [[emulator_raa_aggregated[i].append(raa) for raa in emulator_raa[i]] for i in range(SystemCount)]
                
                # Closure test
                
                # qhat vs T
                T_array = result_dict['T_array']
                T_qhat_truth = result_dict['T_qhat_truth']
                T_qhat_mean = result_dict['T_qhat_mean']
                T_qhat_closure = result_dict['T_qhat_closure']
                T_qhat_closure2 = result_dict['T_qhat_closure2']
                T_qhat_closure_result_list.append(T_qhat_closure)
                T_qhat_closure_result_list2.append(T_qhat_closure2)
                T_qhat_closure_truth_list.append(T_qhat_truth)
                
                # qhat vs E
                E_array = result_dict['E_array']
                E_qhat_truth = result_dict['E_qhat_truth']
                E_qhat_mean = result_dict['E_qhat_mean']
                E_qhat_closure = result_dict['E_qhat_closure']
                E_qhat_closure2 = result_dict['E_qhat_closure2']
                E_qhat_closure_result_list.append(E_qhat_closure)
                E_qhat_closure_result_list2.append(E_qhat_closure2)
                E_qhat_closure_truth_list.append(E_qhat_truth)

                # ABCD closure
                #theta_closure_list.append(result_dict['theta'])
#               for name in self.Names:
#                   theta_closure_result_dict[name].append(result_dict['{}_closure'.format(name)])
#                   theta_closure_result2_dict[name].append(result_dict['{}_closure2'.format(name)])

#       # Print theta closure summary
#       for name in self.Names:
#           fraction = 1.*sum(theta_closure_result_dict[name])/len(theta_closure_result_dict[name])
#           print('{} closure: {}'.format(name, fraction))
#
#           fraction = 1.*sum(theta_closure_result2_dict[name])/len(theta_closure_result2_dict[name])
#           print('{} closure2: {}'.format(name, fraction))

        # Plot summary of holdout tests
        #self.plot_avg_residuals()
        self.plot_emulator_validation(true_raa_aggregated, emulator_raa_aggregated)

        # Plot summary of closure tests
        self.plot_closure_summary_qhat(T_array, T_qhat_closure_result_list,
                                       T_qhat_closure_truth_list, type='T', CR='90')
        self.plot_closure_summary_qhat(T_array, T_qhat_closure_result_list2,
                                       T_qhat_closure_truth_list, type='T', CR='60')
        self.plot_closure_summary_qhat(E_array, E_qhat_closure_result_list,
                                       E_qhat_closure_truth_list, type='E', CR='90')
        self.plot_closure_summary_qhat(E_array, E_qhat_closure_result_list2,
                                       E_qhat_closure_truth_list, type='E', CR='60')

    #---------------------------------------------------------------
    # Plot summary of closure tests
    #
    # qhat_closure_result_list is a list (per design point) of lists (of T values)
    # [ [True, True, ...], [True, False,  ...], ... ] where each sublist is a given design point
    
    # qhat_closure_truth_list is a list (per design point) of lists (of T values)
    # [ [qhat_T1, qhat_T2, ...], [qhat_T1, qhat_T2, ...], ... ] where each sublist is a given design point
    #---------------------------------------------------------------
    def plot_closure_summary_qhat(self, x_array, qhat_closure_result_list,
                                  qhat_closure_truth_list, type='T', CR='90'):
        
        # Construct 2D histogram of <qhat of design point> vs T,
        # where amplitude is fraction of successful closure tests
        
        # For each T and design point, compute <qhat of design point>,
        # T, and the fraction of successful closure tests
        x_list = []
        qhat_mean_list = []
        success_list = []
        for i,x in enumerate(x_array):
            for j,design in enumerate(qhat_closure_result_list):
            
                qhat_mean = statistics.mean(qhat_closure_truth_list[j])
                success = qhat_closure_result_list[j][i]
                
                x_list.append(x)
                qhat_mean_list.append(qhat_mean)
                success_list.append(success)
                
        # Now draw the mean success rate in 2D
        if type is 'T':
            xbins = np.linspace(0.15, 0.5, num=8)
        if type is 'E':
            xbins = np.linspace(5, 200, num=8)
        ybins =  [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]
        ybins_center =  [(ybins[i+1]+ybins[i])/2 for i in range(len(ybins)-1)]

        x = np.array(x_list)
        y = np.array(qhat_mean_list)
        z = np.array(success_list)
        
        H, xedges, yedges, binnumber= scipy.stats.binned_statistic_2d(x, y, z, statistic=np.mean,
                                                                      bins=[xbins, ybins])
        H = np.ma.masked_invalid(H) # masking where there was no data
        XX, YY = np.meshgrid(xedges, yedges)

        fig = plt.figure(figsize = (11,9))
        ax1=plt.subplot(111)
        plot1 = ax1.pcolormesh(XX, YY, H.T)
        fig.colorbar(plot1, ax=ax1)
        
        plt.xlabel('{} (GeV)'.format(type), size=14)
        if type is 'T':
            plt.ylabel(r'$\left< \hat{q}/T^3 \right>_{E=100\;\rm{GeV}}$', size=14)
        if type is 'E':
            plt.ylabel(r'$\left< \hat{q}/T^3 \right>_{T=300\;\rm{MeV}}$', size=14)
        plt.title('Fraction of closure tests contained in {}% CR'.format(CR), size=14)
            
        for i in range(len(xbins)-1):
            for j in range(len(ybins)-1):
                zval = H[i][j]
                ax1.text(xbins[i]+0.025, ybins_center[j], '{:0.2f}'.format(zval), size=8, ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
       
        # Save
        plt.savefig('{}/Closure_Summary2D_{}_{}.pdf'.format(self.plot_dir, type, CR), dpi = 192)
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
