'''
Class to steer Bayesian analysis for all models
'''

import os
import argparse
import yaml

################################################################
class MergeAllModels():

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, config_file, output_dir, **kwargs):
  
    # Initialize
    self.config_file = config_file
    self.output_dir = output_dir
    
    # Read config file
    with open(config_file, 'r') as stream:
      config = yaml.safe_load(stream)
      
    self.models = config['models']

  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def merge_all_models(self):
  
    for model in self.models:

      # Run analysis
      os.system('python merge_results.py -c {} -m {} -o {}'.format(self.config_file, model,
                                                                   self.output_dir))

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

    # Parse the arguments
    args = parser.parse_args()
    
    print('')
    print('Configuring RunAllModels...')
    print('configFile: \'{0}\''.format(args.configFile))
    print('outputdir: \'{0}\''.format(args.outputdir))
    
    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
      print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
      sys.exit(0)
    
    analysis = MergeAllModels(config_file = args.configFile, output_dir=args.outputdir)
    analysis.merge_all_models()
