'''
Class to steer Bayesian analysis and produce plots.
'''

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy

import os
import sys
import pickle

import src.reader as Reader
from src.design import Design
from src import emulator, mcmc

################################################################
class RunAnalysis():

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, **kwargs):
    super(RunAnalysis, self).__init__(**kwargs)
    
    # Set output dir
    self.output_dir = '20200701_test_LBT'
    self.debug_level = 0
    
    # Specify model: [MATTER, LBT, MATTER+LBT1, MATTER+LBT2]
    self.model = 'LBT'
    
    # Emulator parameters
    self.retrain_emulator = True
    self.n_pc = 3
    self.n_restarts = 10
    
    # MCMC parameters
    self.rerun_mcmc = False
    self.n_walkers = 500
    self.n_burn_steps = 1000
    self.n_steps = 1000
    
    # Holdout test options
    self.do_holdout_tests = True
    self.do_closure_tests = False
    self.n_max_holdout_tests = 1
    
    if self.n_max_holdout_tests < 0:
      self.n_max_holdout_tests = sys.maxsize
    
    # Set model parameter ranges
    # For Matter or LBT: (A, B, C, D)
    # For Matter+LBT 1,2: (A, C, B, D, Q), i.e. transformed versions of {A+C, A/(A+C), B, D, Q} from .dat
    if self.model == 'MATTER':
      self.ranges = [(0.01, 2), (0.01, 20), (0.01, 2), (0.01, 20)]
    elif self.model == 'LBT':
      self.ranges = [(0.01, 2), (0.01, 20), (0.01, 2), (0.01, 20)]
    elif self.model == 'MATTER+LBT1':
      self.ranges = [(0, 1.5), (0, 1.0), (0, 20), (0, 20), (1, 4)]
    elif self.model == 'MATTER+LBT2':
      self.ranges = [(0, 1.5), (0, 1.0), (0, 20), (1, 4)]
    self.Ranges = np.array(self.ranges).T
      
    if self.model == 'MATTER' or self.model == 'LBT':
      self.Names = [r"$A$", r"$B$", r"$C$", r"$D$"]
      self.Names_untransformed = self.Names
    else:
      self.Names = [r"$A$", r"$C$", r"$B$", r"$D$", r"$Q$"]
      self.Names_untransformed = [r"$A+C$", r"$A/(A+C)$", r"$B$", r"$D$", r"$Q$"]
      
    print(self)
  
  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def run_analysis(self):
  
    # Initialize data and model from files, and Run the analysis with all training points
    self.init()
    output_dir = os.path.join(self.output_dir, 'main')
    self.run_single_analysis(output_dir = output_dir)
    
    # Hold out one point from the emulator training, and re-train
    if self.do_holdout_tests:
    
      # Store a list of the chi2 of the holdout residual
      self.avg_residuals = []
      
      # For each emulator:
      # Store lists of true RAA, emulator RAA at each holdout point
      # (over all pt, centralities)
      self.SystemCount = len(self.AllData["systems"])
      self.true_raa = [[] for i in range(self.SystemCount)]
      self.emulator_raa = [[] for i in range(self.SystemCount)]
    
      n_design_points = len(self.AllData['design'])
      for i in range(0, n_design_points):
      
        if i  > self.n_max_holdout_tests - 1:
          break
      
        # Re-initialize data structures, with the updated holdout information
        print('Running holdout test {} / {}'.format(i, n_design_points))
        self.init(exclude_index = i)
        print('    {}'.format(self.AllData['holdout_design']))
        if len(self.AllData['design']) != n_design_points - 1:
          sys.exit('Only {} design points remain, but there should be {}!'.format(
                    len(self.AllData['design']), n_design_points - 1))

        self.retrain_emulator = True
        output_dir = os.path.join(self.output_dir, 'holdout/{}'.format(i))
        
        # If closure test is enabled, perform the MCMC sampling
        if self.do_closure_tests:
          self.rerun_mcmc = True
          self.run_single_analysis(output_dir = output_dir, holdout_test = True, closure_test = True)
        else:
          self.run_single_analysis(output_dir = output_dir, holdout_test = True)
          
        plt.close('all')
       
      # Plot summary of holdout tests
      self.plot_dir = os.path.join(self.output_dir, 'holdout')
      self.plot_avg_residuals()
      self.plot_emulator_validation()
      
      # Plot summary of closure tests
      # ...

  #---------------------------------------------------------------
  # Plot emulator validation
  #---------------------------------------------------------------
  def plot_emulator_validation(self):
  
    figure, axes = plt.subplots(figsize=(8*self.SystemCount, 8),
                                ncols=self.SystemCount, nrows=2,
                                gridspec_kw={'height_ratios': [2,1]})
    for i in range(self.SystemCount):

      # Get RAA points
      true_raa = np.array(self.true_raa[i])
      emulator_raa = np.array(self.emulator_raa[i])
      normalized_residual = np.divide(np.abs(true_raa-emulator_raa), true_raa)
      
      # Draw scatter plot
      axes[0][i].scatter(true_raa, emulator_raa)
      axes[0][i].set_xlabel(r'$R_{AA}^{true}$')
      axes[0][i].set_ylabel(r'$R_{AA}^{emulator}$')
      
      # Draw normalization residuals
      axes[1][i].hist(normalized_residual)
      axes[1][i].set_xlabel(r'$\left(R_{AA}^{true} - R_{AA}^{emulator}\right) / R_{AA}^{true}$')
      
    system = self.AllData["systems"][i]
    plt.savefig('{}/EmulatorValidation.pdf'.format(self.plot_dir))
    plt.close('all')

  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def run_single_analysis(self, output_dir = '.', holdout_test = False, closure_test = False):
  
    # Create output dir
    self.plot_dir = os.path.join(output_dir, 'plots')
    if not os.path.exists(self.plot_dir):
      os.makedirs(self.plot_dir)
      
    # Re-train emulator, if requested
    if self.retrain_emulator:
      # Clean cache for emulator
      for system in self.AllData["systems"]:
          if os.path.exists('cache/emulator/' + system + ".pkl"):
              os.remove('cache/emulator/' + system + ".pkl")
    
      # Re-train emulator
      os.system('python -m src.emulator --retrain --npc {} --nrestarts {}'.format(self.n_pc, self.n_restarts))
    
    # Load trained emulator
    self.EmulatorAuAu200 = emulator.Emulator.from_cache('AuAu200')
    self.EmulatorPbPb2760 = emulator.Emulator.from_cache('PbPb2760')
    self.EmulatorPbPb5020 = emulator.Emulator.from_cache('PbPb5020')
    
    # Construct plots characterizing the emulator
    self.plot_design(holdout_test = holdout_test)
    self.plot_RAA(self.AllData["design"], 'Design')
    #self.plot_PC_residuals()
    
    if holdout_test:
      self.plot_emulator_RAA_residuals(holdout_test = True)
      if not closure_test:
        return
    else:
      self.plot_emulator_RAA_residuals()
    
    # Run MCMC
    if self.rerun_mcmc:
      if os.path.exists('cache/mcmc_chain.hdf'):
        os.remove("cache/mcmc_chain.hdf")
      os.system('python -m src.mcmc --nwalkers {} --nburnsteps {} {}'.format(self.n_walkers, self.n_burn_steps, self.n_steps))
    
    # Load MCMC chain
    self.chain = mcmc.Chain()
    self.MCMCSamples = self.chain.load()
    
    # Plot dependence of MC sampling on number of steps
    self.plot_MCMC_samples()

    # Transform coordinates
    if self.model in ['MATTER+LBT1', 'MATTER+LBT2']:
      self.TransformedSamples = np.copy(self.MCMCSamples)
      self.TransformedSamples[:,0] = self.MCMCSamples[:,0] * self.MCMCSamples[:,1]
      self.TransformedSamples[:,1] = self.MCMCSamples[:,0] - self.MCMCSamples[:,0] * self.MCMCSamples[:,1]
    else:
      self.TransformedSamples = np.copy(self.MCMCSamples)
    
    # Plot posterior distributions of parameters
    self.plot_correlation(suffix = '', holdout_test = holdout_test)
    if self.model in  ['MATTER+LBT1', 'MATTER+LBT2']:
      self.plot_correlation(suffix = '_Transformed', holdout_test = holdout_test)
    
    # Plot RAA for samples of the posterior parameter space
    sample_points = self.MCMCSamples[ np.random.choice(range(len(self.MCMCSamples)), 100), :]
    self.plot_RAA(sample_points, 'Posterior')
    
    plt.close('all')
    
    # Plot qhat/T^3 for the holdout point
    if closure_test:
      self.plot_closure_test()
      
    plt.close('all')
    
  #---------------------------------------------------------------
  # Plot qhat/T^3 for the holdout point
  #---------------------------------------------------------------
  def plot_closure_test(self, E=100.):
    
    T_array = np.linspace(0.16, 0.5)
      
    # Plot truth value
    qhat = [self.qhat(T=T, E=E, parameters=self.AllData['holdout_design']) for T in T_array]
    plt.plot(T_array, qhat, sns.xkcd_rgb['pale red'],
             linewidth=2., label='Truth')
    plt.xlabel('T (GeV)')
    plt.ylabel(r'$\hat{q}/T^3$')
    
    ymin = 0
    ymax = 10
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])
    
    # Plot 90% confidence interval of qhat solution
    # --> Construct distribution of qhat by sampling each ABCD point
    qhat_posteriors = [[self.qhat(T=T, E=E, parameters=parameters)
                        for parameters in self.TransformedSamples]
                        for T in T_array]
    
    # Get list of mean qhat values for each T
    mean = [np.mean(qhat_values) for qhat_values in qhat_posteriors]
    plt.plot(T_array, mean, sns.xkcd_rgb['denim blue'],
             linewidth=2., linestyle='--', label='Extracted')
             
    # Get confidence interval for each T
    # Note: use stdev rather than stderr,
    #std_err_list = [scipy.stats.sem(qhat_values) for qhat_values in qhat_posteriors]
    std_err_list = [np.std(qhat_values) for qhat_values in qhat_posteriors]
    confidence = 0.9
    n = len(qhat_posteriors[0])
    q = scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    h = [q*std_err for std_err in std_err_list]
          
    err_low = np.array(mean) - np.array(h)
    err_up =  np.array(mean) + np.array(h)
    plt.fill_between(T_array, err_low, err_up, color=sns.xkcd_rgb['light blue'],
                     label='{}% Confidence Interval'.format(int(confidence*100)))
             
    # Draw legend
    first_legend = plt.legend(title=self.model, title_fontsize=15,
                             loc='upper right', fontsize=12)
    ax = plt.gca().add_artist(first_legend)
   
    # Draw text info
    
    plt.savefig('{}/Closure.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')
    
    # Plot distribution of posterior qhat values for a given T
    plt.hist(qhat_posteriors[0], bins=50,
            histtype='step', color='green')

    plt.savefig('{}/ClosureDist.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')
    
  #---------------------------------------------------------------
  # Plot design points
  #---------------------------------------------------------------
  def plot_design(self, holdout_test = False):
      
    # Tranform {A+C, A/(A+C), B, D, Q}  to {A,B,C,D,Q}
    design_points = self.AllData['design']
    if self.model in ['MATTER+LBT1', 'MATTER+LBT2']:
      transformed_design_points = np.copy(design_points)
      transformed_design_points[:,0] = design_points[:,0] * design_points[:,1]
      transformed_design_points[:,1] = design_points[:,0] - design_points[:,0] * design_points[:,1]
    else:
      transformed_design_points = np.copy(design_points)
    
    NDimension = len(self.AllData["labels"])
    figure, axes = plt.subplots(figsize = (3 * NDimension, 3 * NDimension), ncols = NDimension, nrows = NDimension)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i==j:
                ax.hist(transformed_design_points[:,i], bins=50,
                        range=self.Ranges[:,i], histtype='step', color='green')
                ax.set_xlabel(self.Names[i])
                ax.set_xlim(*self.Ranges[:,j])
            if i>j:
                ax.hist2d(transformed_design_points[:, j], transformed_design_points[:, i],
                          bins=50, range=[self.Ranges[:,j], self.Ranges[:,i]],
                          cmap='Greens')
                ax.set_xlabel(self.Names[j])
                ax.set_ylabel(self.Names[i])
                ax.set_xlim(*self.Ranges[:,j])
                ax.set_ylim(*self.Ranges[:,i])
                
                if holdout_test:
                  ax.plot(self.AllData['holdout_design'][j], self.AllData['holdout_design'][i], 'ro')

            if i<j:
                ax.axis('off')
    plt.tight_layout(True)
    plt.savefig('{}/DesignPoints.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')
    
  #---------------------------------------------------------------
  # Plot RAA of the model at a set of points in the parameter space
  #---------------------------------------------------------------
  def plot_RAA(self, points, name):

    TempPrediction = {"AuAu200": self.EmulatorAuAu200.predict(points),
                     "PbPb2760": self.EmulatorPbPb2760.predict(points),
                     "PbPb5020": self.EmulatorPbPb5020.predict(points)}

    SystemCount = len(self.AllData["systems"])

    figure, axes = plt.subplots(figsize = (15, 5 * SystemCount), ncols = 2, nrows = SystemCount)

    for s1 in range(0, SystemCount):  # Collision system
        for s2 in range(0, 2): # Centrality range
            axes[s1][s2].set_xlabel(r"$p_{T}$")
            axes[s1][s2].set_ylabel(r"$R_{AA}$")
            
            # Plot data points
            S1 = self.AllData["systems"][s1]
            O  = self.AllData["observables"][0][0]
            S2 = self.AllData["observables"][0][1][s2]

            DX = self.AllData["data"][S1][O][S2]['x']
            DY = self.AllData["data"][S1][O][S2]['y']
            DE = np.sqrt(self.AllData["data"][S1][O][S2]['yerr']['stat'][:,0]**2 + self.AllData["data"][S1][O][S2]['yerr']['sys'][:,0]**2)
                    
            # Plot emulator predictions at design points
            for i, y in enumerate(TempPrediction[S1][O][S2]):
                axes[s1][s2].plot(DX, y, 'b-', alpha=0.1, label="Posterior" if i==0 else '')
            axes[s1][s2].errorbar(DX, DY, yerr = DE, fmt='ro', label="Measurements")

    plt.tight_layout(True)
    figure.savefig('{}/RAA_{}.pdf'.format(self.plot_dir, name), dpi = 192)
    plt.close('all')

  #---------------------------------------------------------------
  # Plot residuals of RAA between the emulator and the true model values, at the design points
  #---------------------------------------------------------------
  def plot_emulator_RAA_residuals(self, holdout_test = False):

    # Get training points
    if holdout_test:
      Examples = [self.AllData['holdout_design']]
    else:
      Examples = self.AllData['design']

    # Get emulator predictions at training points
    TempPrediction = {"AuAu200": self.EmulatorAuAu200.predict(Examples),
                     "PbPb2760": self.EmulatorPbPb2760.predict(Examples),
                     "PbPb5020": self.EmulatorPbPb5020.predict(Examples)}

    SystemCount = len(self.AllData["systems"])
    figure, axes = plt.subplots(figsize = (15, 5 * SystemCount), ncols = 2, nrows = SystemCount)

    # Loop through system and centrality range
    sum_chi2 = 0.
    n = 0
    for s1 in range(0, SystemCount):  # Collision system
        for s2 in range(0, 2): # Centrality range
            axes[s1][s2].set_xlabel(r"$p_{T}$")
            axes[s1][s2].set_ylabel(r"$(R_{AA}^{emulator} - R_{AA}^{model}) / R_{AA}^{model}$")
    
            # Get keys for given system, centrality
            S1 = self.AllData["systems"][s1]
            O  = self.AllData["observables"][0][0]
            S2 = self.AllData["observables"][0][1][s2]

            # Get MC values at training points
            if holdout_test:
              model_x = self.AllData['holdout_model'][S1][O][S2]['x'] # pt-bin values
              model_y = self.AllData['holdout_model'][S1][O][S2]['Y'] # 1d array of model Y-values at holdout point
            else:
              model_x = self.AllData['model'][S1][O][S2]['x'] # pt-bin values
              model_y = self.AllData['model'][S1][O][S2]['Y'] # 2d array of model Y-values at each training point
    
            # Get emulator predictions at training points
            emulator_y = TempPrediction[S1][O][S2] # 2d array of emulator Y-values at each training point
    
            # Plot difference between model and emulator
            for i, y in enumerate(TempPrediction[S1][O][S2]):
    
              if holdout_test:
                model_y_1d = model_y
                [self.true_raa[s1].append(raa) for raa in TempPrediction[S1][O][S2][i]]
                [self.emulator_raa[s1].append(raa) for raa in model_y_1d]
              else:
                model_y_1d = model_y[i]
    
              deltaRAA = (TempPrediction[S1][O][S2][i] - model_y_1d) / model_y_1d
              for x in np.square(deltaRAA):
                sum_chi2 += x
                n += 1
    
              axes[s1][s2].plot(model_x, deltaRAA, 'b-', alpha=0.1, label="Posterior" if i==0 else '')

    plt.tight_layout(True)
    figure.savefig('{}/RAA_Residuals_Design.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')
    
    if holdout_test:
      self.avg_residuals.append(sum_chi2/n)

  #---------------------------------------------------------------
  # Plot residuals of each PC
  #---------------------------------------------------------------
  def plot_PC_residuals(self):
  
    for system in self.AllData["systems"]:
        # Get emulators for a given system (one per PC) from cache
        gps = emulator.emulators[system].gps
        nrows = len(gps)
        ncols = gps[0].X_train_.shape[1]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4., nrows*4.) )
        ymax = np.ceil(max(np.fabs(g.y_train_).max() for g in gps))
        ylim = (-ymax, ymax)

        design = Design(system)
        test_points = [r*design.min + (1 - r)*design.max for r in [.2, .5, .8]]

        # Loop through emulators (one per PC)
        for ny, (gp, row) in enumerate(zip(gps, axes)):
            
            # Get list of training y-values
            y = gp.y_train_

            # Loop through training parameters {A+C,A/(A+C),B,D,Q}
            for nx, (x, label, xlim, ax) in enumerate(zip(gp.X_train_.T, design.labels, design.range, row)):
                
                # Plot training points
                ax.plot(x, y, 'o', ms=3., color='.75', zorder=10)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel(label)
                ax.set_ylabel('PC {}'.format(ny))
                
                # Plot emulator prediction (and stdev) for three different
                x = np.linspace(xlim[0], xlim[1], 100)
                X = np.empty((x.size, ncols))
                for k, test_point in enumerate(test_points):
                    X[:] = test_point
                    X[:, nx] = x
                    mean, std = gp.predict(X, return_std=True)

                    color = plt.cm.tab10(k)
                    ax.plot(x, mean, lw=.2, color=color, zorder=30)
                    ax.fill_between(x, mean - std, mean + std, lw=0, color=color, alpha=.3, zorder=20)
                    
        plt.tight_layout(True)
        plt.savefig('{}/EmulatorPCs_{}.pdf'.format(self.plot_dir, system), dpi = 192)
        plt.close('all')

  #---------------------------------------------------------------
  # Check that burn-in is sufficient
  #---------------------------------------------------------------
  def plot_MCMC_samples(self):

    with self.chain.dataset() as d:
      W = d.shape[0]     # number of walkers
      S = d.shape[1]     # number of steps
      N = d.shape[2]     # number of paramters
      T = int(S / 200)   # "thinning"
      A = 20 / W
      figure, axes = plt.subplots(figsize = (15, 2 * N), ncols = 1, nrows = N)
      for i, ax in enumerate(axes):
        for j in range(0, W):
          ax.plot(range(0, S, T), d[j, ::T, i], alpha = A)
      plt.tight_layout(True)
      plt.savefig('{}/MCMCSamples.pdf'.format(self.plot_dir), dpi = 192)
      plt.close('all')

  #---------------------------------------------------------------
  # Plot posterior parameter distributions, either in  transformed
  # or non-transformed coordinates
  #---------------------------------------------------------------
  def plot_correlation(self, suffix = '', holdout_test = False):

    if 'Transformed' in suffix:
      Names = self.Names
      samples = self.TransformedSamples
      color = 'blue'
      colormap = 'Blues'
    else:
      Names = self.Names_untransformed
      samples = self.MCMCSamples
      color = 'green'
      colormap = 'Greens'
      
    NDimension = len(self.AllData["labels"])
    figure, axes = plt.subplots(figsize = (3 * NDimension, 3 * NDimension), ncols = NDimension, nrows = NDimension)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i==j:
                ax.hist(samples[:,i], bins=50,
                        range=self.Ranges[:,i], histtype='step', color=color)
                ax.set_xlabel(Names[i])
                ax.set_xlim(*self.Ranges[:,j])
            if i>j:
                ax.hist2d(samples[:, j], samples[:, i],
                          bins=50, range=[self.Ranges[:,j], self.Ranges[:,i]],
                          cmap=colormap)
                ax.set_xlabel(Names[j])
                ax.set_ylabel(Names[i])
                ax.set_xlim(*self.Ranges[:,j])
                ax.set_ylim(*self.Ranges[:,i])
                
                if holdout_test:
                  ax.plot(self.AllData['holdout_design'][j], self.AllData['holdout_design'][i], 'ro')

            if i<j:
                ax.axis('off')
    plt.tight_layout(True)
    plt.savefig('{}/Posterior_Correlations{}.pdf'.format(self.plot_dir, suffix), dpi = 192)
    plt.close('all')

  #---------------------------------------------------------------
  def plot_avg_residuals(self):

    design_points = self.AllData['design']
    if self.model in ['MATTER+LBT1', 'MATTER+LBT2']:
      transformed_design_points = np.copy(design_points)
      transformed_design_points[:,0] = design_points[:,0] * design_points[:,1]
      transformed_design_points[:,1] = design_points[:,0] - design_points[:,0] * design_points[:,1]
    else:
      transformed_design_points = np.copy(design_points)
    
    if len(self.avg_residuals) < len(self.AllData['design']):
      transformed_design_points = transformed_design_points[0:self.n_max_holdout_tests]
    
    NDimension = len(self.AllData["labels"])
    figure, axes = plt.subplots(figsize = (3 * NDimension, 3 * NDimension), ncols = NDimension, nrows = NDimension)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i==j:
                ax.hist(transformed_design_points[:,i], bins=50, weights=self.avg_residuals,
                        range=self.Ranges[:,i], histtype='step', color='blue')
                ax.set_xlabel(self.Names[i])
                ax.set_xlim(*self.Ranges[:,j])
            if i>j:
                ax.hist2d(transformed_design_points[:, j], transformed_design_points[:, i], weights=self.avg_residuals,
                          bins=50, range=[self.Ranges[:,j], self.Ranges[:,i]],
                          cmap='Blues')
                ax.set_xlabel(self.Names[j])
                ax.set_ylabel(self.Names[i])
                ax.set_xlim(*self.Ranges[:,j])
                ax.set_ylim(*self.Ranges[:,i])
            if i<j:
                ax.axis('off')
    plt.tight_layout(True)
    plt.savefig('{}/Average_Residuals.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')

  #---------------------------------------------------------------
  # Return value of qhat/T^3
  def qhat(self, T=0, E=0, parameters=None):
  
    Lambda = 0.2
    C_R = 4./3.
    coeff = 42 * C_R * scipy.special.zeta(3) / np.pi * np.square(4*np.pi/9)
  
    if self.model == 'ML1':
      A = parameters[0]
      B = parameters[1]
      C = parameters[2]
      D = parameters[3]
      Q0 = parameters[4]
      term1 = A * (np.log(E/Lambda) - np.log(B)) / np.square(np.log(E/Lambda))  * np.heaviside(E-Q0, 0.)
      term2 = C * (np.log(E/T) - np.log(D)) / np.square(np.log(E*T/(Lambda*Lambda)))
    elif self.model == 'ML2':
      A = parameters[0]
      C = parameters[1]
      D = parameters[2]
      Q0 = parameters[3]
      term1 = A * (np.log(E/Lambda) - np.log(Q0/Lambda)) / np.square(np.log(E/Lambda)) * np.heaviside(E-Q0, 0.)
      term2 = C * (np.log(E/T) - np.log(D)) / np.square(np.log(E*T/(Lambda*Lambda)))
    else:
      A = parameters[0]
      B = parameters[1]
      C = parameters[2]
      D = parameters[3]
      term1 = A * (np.log(E/Lambda) - np.log(B)) / np.square(np.log(E/Lambda))
      term2 = C * (np.log(E/T) - np.log(D)) / np.square(np.log(E*T/(Lambda*Lambda)))
        
    return coeff * (term1 + term2)

  #---------------------------------------------------------------
  # Initialize data
  #---------------------------------------------------------------
  def init(self, exclude_index = -1):
  
    self.init_files()
    self.init_model(exclude_index)

  #---------------------------------------------------------------
  # Initialize data to dictionary
  #---------------------------------------------------------------
  def init_model(self, exclude_index = -1):
    
    # Initialize empty dictionary
    self.AllData = {}

    # Basic information
    self.AllData["systems"] = ["AuAu200", "PbPb2760", "PbPb5020"]
    self.AllData["keys"] = self.RawDesign["Parameter"]
    self.AllData["labels"] = self.RawDesign["Parameter"]
    self.AllData["ranges"] = self.ranges
    self.AllData["observables"] = [('R_AA', ['C0', 'C1'])]

    # If a holdout point is passed, exclude it from the design and prediction
    if exclude_index >= 0:
      self.exclude_holdout(exclude_index)

    # Data points
    self.Data = {"AuAu200": {"R_AA": {"C0": self.RawData1["Data"], "C1": self.RawData2["Data"]}},
        "PbPb2760": {"R_AA": {"C0": self.RawData3["Data"], "C1": self.RawData4["Data"]}},
        "PbPb5020": {"R_AA": {"C0": self.RawData5["Data"], "C1": self.RawData6["Data"]}}}

    # Model predictions
    self.Prediction = {"AuAu200": {"R_AA": {"C0": {"Y": self.RawPrediction1["Prediction"], "x": self.RawData1["Data"]['x']},
                                       "C1": {"Y": self.RawPrediction2["Prediction"], "x": self.RawData2["Data"]['x']}}},
                 "PbPb2760": {"R_AA": {"C0": {"Y": self.RawPrediction3["Prediction"], "x": self.RawData3["Data"]['x']},
                                       "C1": {"Y": self.RawPrediction4["Prediction"], "x": self.RawData4["Data"]['x']}}},
                 "PbPb5020": {"R_AA": {"C0": {"Y": self.RawPrediction5["Prediction"], "x": self.RawData5["Data"]['x']},
                                       "C1": {"Y": self.RawPrediction6["Prediction"], "x": self.RawData6["Data"]['x']}}}}

    # Covariance matrices - the indices are [system][measurement1][measurement2], each one is a block of matrix
    self.Covariance = Reader.InitializeCovariance(self.Data)
    self.Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")] = Reader.EstimateCovariance(self.RawData1, self.RawData1, SysLength = {"default": 0.2})
    self.Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C1")] = Reader.EstimateCovariance(self.RawData2, self.RawData2, SysLength = {"default": 0.2})
    self.Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C0")] = Reader.EstimateCovariance(self.RawData3, self.RawData3, SysLength = {"default": 0.2})
    self.Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C1")] = Reader.EstimateCovariance(self.RawData4, self.RawData4, SysLength = {"default": 0.2})
    self.Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C0")] = Reader.EstimateCovariance(self.RawData5, self.RawData5, SysLength = {"default": 0.2})
    self.Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C1")] = Reader.EstimateCovariance(self.RawData6, self.RawData6, SysLength = {"default": 0.2})

    # This is how we can add off-diagonal matrices
    # Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C1")] = Reader.EstimateCovariance(RawData5, RawData6, SysLength = {"default": 100}, SysStrength = {"default": 0.1})
    # Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C0")] = Reader.EstimateCovariance(RawData6, RawData5, SysLength = {"default": 100}, SysStrength = {"default": 0.1})

    # This is how we can supply external pre-generated matrices
    # Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")] = RawCov1["Matrix"]
    #Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")] = RawCov11E["Matrix"]
    #Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C1")] = RawCov22E["Matrix"]
    #Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C0")] = RawCov33E["Matrix"]
    #Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C1")] = RawCov44E["Matrix"]
    #Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C0")] = RawCov55E["Matrix"]
    #Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C1")] = RawCov66E["Matrix"]

    # Assign data to the dictionary
    self.AllData["design"] = self.RawDesign["Design"]
    self.AllData["model"] = self.Prediction
    self.AllData["data"] = self.Data
    self.AllData["cov"] = self.Covariance
    
    # Save to the desired pickle file
    with open('input/default.p', 'wb') as handle:
      pickle.dump(self.AllData, handle, protocol = pickle.HIGHEST_PROTOCOL)

    if self.debug_level > 0:
      print(self.AllData["design"].shape)
      print(self.AllData["model"]["AuAu200"]["R_AA"]["C0"]["Y"].shape)
      print(self.AllData["model"]["AuAu200"]["R_AA"]["C1"]["Y"].shape)
      print(self.AllData["model"]["PbPb2760"]["R_AA"]["C0"]["Y"].shape)
      print(self.AllData["model"]["PbPb2760"]["R_AA"]["C1"]["Y"].shape)
      print(self.AllData["model"]["PbPb5020"]["R_AA"]["C0"]["Y"].shape)
      print(self.AllData["model"]["PbPb5020"]["R_AA"]["C1"]["Y"].shape)
    
      print("AuAu200, C0")
      print(self.Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")])
      print("AuAu200, C1")
      print(self.Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C1")])
      print("PbPb2760, C0")
      print(self.Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C0")])
      print("PbPb2760, C1")
      print(self.Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C1")])
      
      SystemCount = len(self.AllData["systems"])
      figure, axes = plt.subplots(figsize = (15, 5 * SystemCount), ncols = 2, nrows = SystemCount)
      axes[0][0].imshow((self.Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")]))
      axes[0][1].imshow((self.Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C1")]))
      axes[1][0].imshow((self.Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C0")]))
      axes[1][1].imshow((self.Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C1")]))
      axes[2][0].imshow((self.Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C0")]))
      axes[2][1].imshow((self.Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C1")]))
      figure.tight_layout()

  #---------------------------------------------------------------
  # Exclude a holdout point from the design and prediction, and store it
  #---------------------------------------------------------------
  def exclude_holdout(self, exclude_index = None):

    # Store the holdout point design and prediction
    HoldoutDesign = self.RawDesign['Design'][exclude_index]
    HoldoutPrediction1 = self.RawPrediction1['Prediction'][exclude_index]
    HoldoutPrediction2 = self.RawPrediction2['Prediction'][exclude_index]
    HoldoutPrediction3 = self.RawPrediction3['Prediction'][exclude_index]
    HoldoutPrediction4 = self.RawPrediction4['Prediction'][exclude_index]
    HoldoutPrediction5 = self.RawPrediction5['Prediction'][exclude_index]
    HoldoutPrediction6 = self.RawPrediction6['Prediction'][exclude_index]
    
    # Model predictions
    HoldoutPrediction = {"AuAu200": {"R_AA": {"C0": {"Y": HoldoutPrediction1, "x": self.RawData1["Data"]['x']},
                                       "C1": {"Y": HoldoutPrediction2, "x": self.RawData2["Data"]['x']}}},
                 "PbPb2760": {"R_AA": {"C0": {"Y": HoldoutPrediction3, "x": self.RawData3["Data"]['x']},
                                       "C1": {"Y": HoldoutPrediction4, "x": self.RawData4["Data"]['x']}}},
                 "PbPb5020": {"R_AA": {"C0": {"Y": HoldoutPrediction5, "x": self.RawData5["Data"]['x']},
                                       "C1": {"Y": HoldoutPrediction6, "x": self.RawData6["Data"]['x']}}}}
     
    # Store the holdout point in the dictionary
    self.AllData['holdout_design'] = HoldoutDesign
    self.AllData['holdout_model'] = HoldoutPrediction
    
    # Remove the holdout point from the design
    self.RawDesign['Design'] = np.delete(self.RawDesign['Design'], exclude_index, axis = 0)

    # Remove the holdout point from the prediction
    self.RawPrediction1['Prediction'] = np.delete(self.RawPrediction1['Prediction'], exclude_index, axis=0)
    self.RawPrediction2['Prediction'] = np.delete(self.RawPrediction2['Prediction'], exclude_index, axis=0)
    self.RawPrediction3['Prediction'] = np.delete(self.RawPrediction3['Prediction'], exclude_index, axis=0)
    self.RawPrediction4['Prediction'] = np.delete(self.RawPrediction4['Prediction'], exclude_index, axis=0)
    self.RawPrediction5['Prediction'] = np.delete(self.RawPrediction5['Prediction'], exclude_index, axis=0)
    self.RawPrediction6['Prediction'] = np.delete(self.RawPrediction6['Prediction'], exclude_index, axis=0)
    
    # If closure test is enabled, set the data to be equal to the held-out point
    if self.do_closure_tests:
      self.RawData1["Data"]["y"] = HoldoutPrediction1
      self.RawData2["Data"]["y"] = HoldoutPrediction2
      self.RawData3["Data"]["y"] = HoldoutPrediction3
      self.RawData4["Data"]["y"] = HoldoutPrediction4
      self.RawData5["Data"]["y"] = HoldoutPrediction5
      self.RawData6["Data"]["y"] = HoldoutPrediction6

  #---------------------------------------------------------------
  # Initialize data
  #---------------------------------------------------------------
  def init_files(self):
  
    # Read data files
    if self.model == 'MATTER':
      self.RawData1   = Reader.ReadData('input/MATTER/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2   = Reader.ReadData('input/MATTER/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3   = Reader.ReadData('input/MATTER/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4   = Reader.ReadData('input/MATTER/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5   = Reader.ReadData('input/MATTER/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6   = Reader.ReadData('input/MATTER/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'LBT':
      self.RawData1   = Reader.ReadData('input/LBTJake/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2   = Reader.ReadData('input/LBTJake/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3   = Reader.ReadData('input/LBTJake/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4   = Reader.ReadData('input/LBTJake/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5   = Reader.ReadData('input/LBTJake/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6   = Reader.ReadData('input/LBTJake/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT1':
      self.RawData1 = Reader.ReadData('input/MATTERLBT1/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2 = Reader.ReadData('input/MATTERLBT1/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3 = Reader.ReadData('input/MATTERLBT1/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4 = Reader.ReadData('input/MATTERLBT1/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5 = Reader.ReadData('input/MATTERLBT1/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6 = Reader.ReadData('input/MATTERLBT1/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT2':
      self.RawData1 = Reader.ReadData('input/MATTERLBT2/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2 = Reader.ReadData('input/MATTERLBT2/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3 = Reader.ReadData('input/MATTERLBT2/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4 = Reader.ReadData('input/MATTERLBT2/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5 = Reader.ReadData('input/MATTERLBT2/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6 = Reader.ReadData('input/MATTERLBT2/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    else:
      sys.exit('Unknown model {}! Options are: M, L, ML1, ML2'.format(self.model))

    # Read covariance
    self.RawCov11L = Reader.ReadCovariance('input/LBT/Covariance_PHENIX_AuAu200_RAACharged_0to10_2013_PHENIX_AuAu200_RAACharged_0to10_2013_Jake.dat')
    self.RawCov22L = Reader.ReadCovariance('input/LBT/Covariance_PHENIX_AuAu200_RAACharged_40to50_2013_PHENIX_AuAu200_RAACharged_40to50_2013_Jake.dat')
    self.RawCov33L = Reader.ReadCovariance('input/LBT/Covariance_ATLAS_PbPb2760_RAACharged_0to5_2015_ATLAS_PbPb2760_RAACharged_0to5_2015_Jake.dat')
    self.RawCov44L = Reader.ReadCovariance('input/LBT/Covariance_ATLAS_PbPb2760_RAACharged_30to40_2015_ATLAS_PbPb2760_RAACharged_30to40_2015_Jake.dat')
    self.RawCov55L = Reader.ReadCovariance('input/LBT/Covariance_CMS_PbPb5020_RAACharged_0to10_2017_CMS_PbPb5020_RAACharged_0to10_2017_Jake.dat')
    self.RawCov66L = Reader.ReadCovariance('input/LBT/Covariance_CMS_PbPb5020_RAACharged_30to50_2017_CMS_PbPb5020_RAACharged_30to50_2017_Jake.dat')

    self.RawCov11E = Reader.ReadCovariance('input/Example/Covariance_PHENIX_AuAu200_RAACharged_0to10_2013_PHENIX_AuAu200_RAACharged_0to10_2013_SmallL.dat')
    self.RawCov22E = Reader.ReadCovariance('input/Example/Covariance_PHENIX_AuAu200_RAACharged_40to50_2013_PHENIX_AuAu200_RAACharged_40to50_2013_SmallL.dat')
    self.RawCov33E = Reader.ReadCovariance('input/Example/Covariance_ATLAS_PbPb2760_RAACharged_0to5_2015_ATLAS_PbPb2760_RAACharged_0to5_2015_SmallL.dat')
    self.RawCov44E = Reader.ReadCovariance('input/Example/Covariance_ATLAS_PbPb2760_RAACharged_30to40_2015_ATLAS_PbPb2760_RAACharged_30to40_2015_SmallL.dat')
    self.RawCov55E = Reader.ReadCovariance('input/Example/Covariance_CMS_PbPb5020_RAACharged_0to10_2017_CMS_PbPb5020_RAACharged_0to10_2017_SmallL.dat')
    self.RawCov66E = Reader.ReadCovariance('input/Example/Covariance_CMS_PbPb5020_RAACharged_30to50_2017_CMS_PbPb5020_RAACharged_30to50_2017_SmallL.dat')

    # Read design points
    if self.model == 'MATTER':
      self.RawDesign = Reader.ReadDesign('input/MATTER/Design.dat')
    elif self.model == 'LBT':
      self.RawDesign = Reader.ReadDesign('input/LBTJake/Design.dat')
    elif self.model == 'MATTER+LBT1':
      self.RawDesign = Reader.ReadDesign('input/MATTERLBT1/Design.dat')
    elif self.model == 'MATTER+LBT2':
      self.RawDesign= Reader.ReadDesign('input/MATTERLBT2/Design.dat')

    # Read model prediction
    if self.model == 'MATTER':
      self.RawPrediction1   = Reader.ReadPrediction('input/MATTER/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2   = Reader.ReadPrediction('input/MATTER/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3   = Reader.ReadPrediction('input/MATTER/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4   = Reader.ReadPrediction('input/MATTER/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5   = Reader.ReadPrediction('input/MATTER/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6   = Reader.ReadPrediction('input/MATTER/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'LBT':
      self.RawPrediction1   = Reader.ReadPrediction('input/LBTJake/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2   = Reader.ReadPrediction('input/LBTJake/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3   = Reader.ReadPrediction('input/LBTJake/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4   = Reader.ReadPrediction('input/LBTJake/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5   = Reader.ReadPrediction('input/LBTJake/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6   = Reader.ReadPrediction('input/LBTJake/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT1':
      self.RawPrediction1 = Reader.ReadPrediction('input/MATTERLBT1/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2 = Reader.ReadPrediction('input/MATTERLBT1/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3 = Reader.ReadPrediction('input/MATTERLBT1/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4 = Reader.ReadPrediction('input/MATTERLBT1/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5 = Reader.ReadPrediction('input/MATTERLBT1/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6 = Reader.ReadPrediction('input/MATTERLBT1/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT2':
      self.RawPrediction1 = Reader.ReadPrediction('input/MATTERLBT2/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2 = Reader.ReadPrediction('input/MATTERLBT2/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3 = Reader.ReadPrediction('input/MATTERLBT2/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4 = Reader.ReadPrediction('input/MATTERLBT2/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5 = Reader.ReadPrediction('input/MATTERLBT2/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6 = Reader.ReadPrediction('input/MATTERLBT2/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')

  #---------------------------------------------------------------
  # Return formatted string of class members
  #---------------------------------------------------------------
  def __str__(self):
    s = []
    variables = self.__dict__.keys()
    for v in variables:
      s.append('{} = {}'.format(v, self.__dict__[v]))
    return "[i] {} with \n .  {}".format(self.__class__.__name__, '\n .  '.join(s))

#----------------------------------------------------------------------
if __name__ == '__main__':

  analysis = RunAnalysis()
  analysis.run_analysis()
