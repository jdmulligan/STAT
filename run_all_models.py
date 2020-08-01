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
    
    # Read config file
    with open(config_file, 'r') as stream:
      config = yaml.safe_load(stream)
      
    self.models = config['models']

  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def run_all_models(self):
  
    for model in self.models:
    
      # Clear previous config file (work around janky __init__ design)
      if os.path.exists('input/default.p'):
        os.system('rm input/default.p')
        print('removed default.p')
      else:
        print('default.p does not exist')
        
      # Write a new default.p (must be done before I can call the analysis script...)
      init = run_analysis_base.RunAnalysisBase(self.config_file, model,
                                               self.output_dir, self.exclude_index)
      init.init_model_type()
      init.init()
        
      # Run analysis
      os.system('python run_analysis.py -c {} -m {} -o {} -i {}'.format(self.config_file, model,
                                                                        self.output_dir, self.exclude_index))

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
