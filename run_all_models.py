'''
Class to steer Bayesian analysis for all models
'''

import os
import argparse
import yaml

import run_analysis_base

################################################################
class RunAllModels():

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, config_file, output_dir, exclude_index, **kwargs):
  
    # Initialize
    self.config_file = config_file
    self.output_dir = output_dir
    self.exclude_index = exclude_index
    
    if exclude_index < 0:
        self.subdir = 'main'
    else:
        self.subdir =  'holdout/{}'.format(exclude_index)
    
    # Read config file
    with open(config_file, 'r') as stream:
      config = yaml.safe_load(stream)
      
    self.models = config['models']
    self.alpha = config['alpha']

  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def run_all_models(self):
  
    for i, model in enumerate(self.models):
    
      # Set path to default.p
      pkl_path = os.path.join(model,'{}/default.p'.format(self.subdir))
    
      # Clear previous default.p (work around janky __init__ design)
      if os.path.exists(pkl_path):
        os.system('rm {}'.format(pkl_path))
        print('removed {}'.format(pkl_path))
      else:
        print('{} does not exist'.format(pkl_path))
        
      # Write a new default.p (must be done before I can call the analysis script...)
      init = run_analysis_base.RunAnalysisBase(self.config_file, model, self.output_dir,
                                               self.alpha[i], self.exclude_index)
      init.init_model_type()
      init.initialize()
        
      # Run analysis
      os.system('python run_analysis.py -c {} -m {} -o {} -a {} -i {}'.format(self.config_file,
                                                                              model,
                                                                              self.output_dir,
                                                                              self.alpha[i],
                                                                              self.exclude_index))

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
    parser.add_argument('-i', '--excludeIndex', action='store',
                        type=int, metavar='excludeIndex',
                        default=-1,
                        help='Index of design point to exclude from emulator')

    # Parse the arguments
    args = parser.parse_args()
    
    print('')
    print('Configuring RunAllModels...')
    print('configFile: \'{0}\''.format(args.configFile))
    print('outputdir: \'{0}\''.format(args.outputdir))
    print('exclude_index: \'{0}\''.format(args.excludeIndex))
    
    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
      print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
      sys.exit(0)
    
    analysis = RunAllModels(config_file = args.configFile, output_dir=args.outputdir,
                            exclude_index=args.excludeIndex)
    analysis.run_all_models()
