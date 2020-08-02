'''
Base class to steer Bayesian analysis and produce plots.
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
import yaml

import reader

################################################################
class RunAnalysisBase():

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, config_file, model, output_dir, exclude_index, **kwargs):
    super(RunAnalysisBase, self).__init__(**kwargs)
    
    self.model = model
    self.output_dir_base = output_dir
    self.exclude_index = exclude_index
    
    # Initialize yaml settings
    self.initialize_config(config_file)
    
  #---------------------------------------------------------------
  # Initialize config
  #---------------------------------------------------------------
  def initialize_config(self, config_file):
  
    # Read config file
    with open(config_file, 'r') as stream:
      config = yaml.safe_load(stream)
      
    self.debug_level = config['debug_level']
    
    # Emulator parameters
    self.retrain_emulator = config['retrain_emulator']
    self.n_pc = config['n_pc']
    self.n_restarts = config['n_restarts']
    
    # MCMC parameters
    self.rerun_mcmc = config['rerun_mcmc']
    self.n_walkers = config['n_walkers']
    self.n_burn_steps = config['n_burn_steps']
    self.n_steps = config['n_steps']
      
  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def run_model(self):
  
    # Initialize a few settings
    self.output_dir = os.path.join(self.output_dir_base, self.model)
    self.init_model_type()
    print(self)
    
    # Run user-defined function
    self.run_analysis()

  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def init_model_type(self):
    
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
 
  #---------------------------------------------------------------
  # Run user-defined function
  #---------------------------------------------------------------
  def run_analysis(self):
  
    raise NotImplementedError('You must implement run_analysis()!')

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
    self.Covariance = reader.InitializeCovariance(self.Data)
    self.Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")] = reader.EstimateCovariance(self.RawData1, self.RawData1, SysLength = {"default": 0.2})
    self.Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C1")] = reader.EstimateCovariance(self.RawData2, self.RawData2, SysLength = {"default": 0.2})
    self.Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C0")] = reader.EstimateCovariance(self.RawData3, self.RawData3, SysLength = {"default": 0.2})
    self.Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C1")] = reader.EstimateCovariance(self.RawData4, self.RawData4, SysLength = {"default": 0.2})
    self.Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C0")] = reader.EstimateCovariance(self.RawData5, self.RawData5, SysLength = {"default": 0.2})
    self.Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C1")] = reader.EstimateCovariance(self.RawData6, self.RawData6, SysLength = {"default": 0.2})

    # This is how we can add off-diagonal matrices
    # Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C1")] = reader.EstimateCovariance(RawData5, RawData6, SysLength = {"default": 100}, SysStrength = {"default": 0.1})
    # Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C0")] = reader.EstimateCovariance(RawData6, RawData5, SysLength = {"default": 100}, SysStrength = {"default": 0.1})

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
    print('Wrote input/default.p')

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
    
    # For closure test: set the data to be equal to the held-out point
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
      self.RawData1   = reader.ReadData('input/MATTER/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2   = reader.ReadData('input/MATTER/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3   = reader.ReadData('input/MATTER/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4   = reader.ReadData('input/MATTER/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5   = reader.ReadData('input/MATTER/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6   = reader.ReadData('input/MATTER/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'LBT':
      self.RawData1   = reader.ReadData('input/LBTJake/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2   = reader.ReadData('input/LBTJake/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3   = reader.ReadData('input/LBTJake/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4   = reader.ReadData('input/LBTJake/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5   = reader.ReadData('input/LBTJake/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6   = reader.ReadData('input/LBTJake/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT1':
      self.RawData1 = reader.ReadData('input/MATTERLBT1/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2 = reader.ReadData('input/MATTERLBT1/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3 = reader.ReadData('input/MATTERLBT1/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4 = reader.ReadData('input/MATTERLBT1/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5 = reader.ReadData('input/MATTERLBT1/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6 = reader.ReadData('input/MATTERLBT1/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT2':
      self.RawData1 = reader.ReadData('input/MATTERLBT2/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2 = reader.ReadData('input/MATTERLBT2/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3 = reader.ReadData('input/MATTERLBT2/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4 = reader.ReadData('input/MATTERLBT2/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5 = reader.ReadData('input/MATTERLBT2/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6 = reader.ReadData('input/MATTERLBT2/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    else:
      sys.exit('Unknown model {}! Options are: M, L, ML1, ML2'.format(self.model))

    # Read covariance
    self.RawCov11L = reader.ReadCovariance('input/LBT/Covariance_PHENIX_AuAu200_RAACharged_0to10_2013_PHENIX_AuAu200_RAACharged_0to10_2013_Jake.dat')
    self.RawCov22L = reader.ReadCovariance('input/LBT/Covariance_PHENIX_AuAu200_RAACharged_40to50_2013_PHENIX_AuAu200_RAACharged_40to50_2013_Jake.dat')
    self.RawCov33L = reader.ReadCovariance('input/LBT/Covariance_ATLAS_PbPb2760_RAACharged_0to5_2015_ATLAS_PbPb2760_RAACharged_0to5_2015_Jake.dat')
    self.RawCov44L = reader.ReadCovariance('input/LBT/Covariance_ATLAS_PbPb2760_RAACharged_30to40_2015_ATLAS_PbPb2760_RAACharged_30to40_2015_Jake.dat')
    self.RawCov55L = reader.ReadCovariance('input/LBT/Covariance_CMS_PbPb5020_RAACharged_0to10_2017_CMS_PbPb5020_RAACharged_0to10_2017_Jake.dat')
    self.RawCov66L = reader.ReadCovariance('input/LBT/Covariance_CMS_PbPb5020_RAACharged_30to50_2017_CMS_PbPb5020_RAACharged_30to50_2017_Jake.dat')

    self.RawCov11E = reader.ReadCovariance('input/Example/Covariance_PHENIX_AuAu200_RAACharged_0to10_2013_PHENIX_AuAu200_RAACharged_0to10_2013_SmallL.dat')
    self.RawCov22E = reader.ReadCovariance('input/Example/Covariance_PHENIX_AuAu200_RAACharged_40to50_2013_PHENIX_AuAu200_RAACharged_40to50_2013_SmallL.dat')
    self.RawCov33E = reader.ReadCovariance('input/Example/Covariance_ATLAS_PbPb2760_RAACharged_0to5_2015_ATLAS_PbPb2760_RAACharged_0to5_2015_SmallL.dat')
    self.RawCov44E = reader.ReadCovariance('input/Example/Covariance_ATLAS_PbPb2760_RAACharged_30to40_2015_ATLAS_PbPb2760_RAACharged_30to40_2015_SmallL.dat')
    self.RawCov55E = reader.ReadCovariance('input/Example/Covariance_CMS_PbPb5020_RAACharged_0to10_2017_CMS_PbPb5020_RAACharged_0to10_2017_SmallL.dat')
    self.RawCov66E = reader.ReadCovariance('input/Example/Covariance_CMS_PbPb5020_RAACharged_30to50_2017_CMS_PbPb5020_RAACharged_30to50_2017_SmallL.dat')

    # Read design points
    if self.model == 'MATTER':
      self.RawDesign = reader.ReadDesign('input/MATTER/Design.dat')
    elif self.model == 'LBT':
      self.RawDesign = reader.ReadDesign('input/LBTJake/Design.dat')
    elif self.model == 'MATTER+LBT1':
      self.RawDesign = reader.ReadDesign('input/MATTERLBT1/Design.dat')
    elif self.model == 'MATTER+LBT2':
      self.RawDesign= reader.ReadDesign('input/MATTERLBT2/Design.dat')

    # Read model prediction
    if self.model == 'MATTER':
      self.RawPrediction1   = reader.ReadPrediction('input/MATTER/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2   = reader.ReadPrediction('input/MATTER/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3   = reader.ReadPrediction('input/MATTER/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4   = reader.ReadPrediction('input/MATTER/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5   = reader.ReadPrediction('input/MATTER/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6   = reader.ReadPrediction('input/MATTER/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'LBT':
      self.RawPrediction1   = reader.ReadPrediction('input/LBTJake/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2   = reader.ReadPrediction('input/LBTJake/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3   = reader.ReadPrediction('input/LBTJake/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4   = reader.ReadPrediction('input/LBTJake/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5   = reader.ReadPrediction('input/LBTJake/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6   = reader.ReadPrediction('input/LBTJake/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT1':
      self.RawPrediction1 = reader.ReadPrediction('input/MATTERLBT1/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2 = reader.ReadPrediction('input/MATTERLBT1/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3 = reader.ReadPrediction('input/MATTERLBT1/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4 = reader.ReadPrediction('input/MATTERLBT1/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5 = reader.ReadPrediction('input/MATTERLBT1/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6 = reader.ReadPrediction('input/MATTERLBT1/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT2':
      self.RawPrediction1 = reader.ReadPrediction('input/MATTERLBT2/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2 = reader.ReadPrediction('input/MATTERLBT2/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3 = reader.ReadPrediction('input/MATTERLBT2/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4 = reader.ReadPrediction('input/MATTERLBT2/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5 = reader.ReadPrediction('input/MATTERLBT2/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6 = reader.ReadPrediction('input/MATTERLBT2/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    
  #---------------------------------------------------------------
  # Return formatted string of class members
  #---------------------------------------------------------------
  def __str__(self):
    s = []
    variables = self.__dict__.keys()
    for v in variables:
      s.append('{} = {}'.format(v, self.__dict__[v]))
    return "[i] {} with \n .  {}".format(self.__class__.__name__, '\n .  '.join(s))
