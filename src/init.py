""" Class to initialize common objects. """

import pickle
from pathlib import Path

################################################################
class Init():

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, workdir = '.', **kwargs):

    print('Init class created.')
    
    self.workdir = Path(workdir)
    self.cachedir = self.workdir / 'cache'
    print('workdir: {}'.format(self.workdir))
    print('cachedir: {}'.format(self.cachedir))

  #---------------------------------------------------------------
  # Initialize settings as class members of obj
  #---------------------------------------------------------------
  def Initialize(self, obj):

    obj.workdir = self.workdir
    obj.cachedir = self.cachedir
    obj.cachedir.mkdir(parents=True, exist_ok=True)

    obj.AllData = pickle.load((obj.workdir / 'input/default.p').open('rb'))

    #: Sets the collision systems for the entire project,
    #: where each system is a string of the form
    #: ``'<projectile 1><projectile 2><beam energy in GeV>'``,
    #: such as ``'PbPb2760'``, ``'AuAu200'``, ``'pPb5020'``.
    #: Even if the project uses only a single system,
    #: this should still be a list of one system string.
    obj.systems = obj.AllData["systems"]

    #: Design attribute. This is a list of
    #: strings describing the inputs.
    #: The default is for the example data.
    obj.keys = obj.AllData["keys"]

    #: Design attribute. This is a list of input
    #: labels in LaTeX for plotting.
    #: The default is for the example data.
    obj.labels = obj.AllData["labels"]

    #: Design attribute. This is list of tuples of
    #: (min,max) for each design input.
    #: The default is for the example data.
    obj.ranges = obj.AllData["ranges"]

    #: Design array to use - should be a numpy array.
    #: Keep at None generate a Latin Hypercube with above (specified) range.
    #: Design array for example is commented under default.
    obj.design_array = obj.AllData["design"]

    #: Dictionary of the model output.
    #: Form MUST be data_list[system][observable][subobservable][{'Y': ,'x': }].
    #:     'Y' is an (n x p) numpy array of the output.
    #:
    #:     'x' is a (1 x p) numpy array of numeric index of columns of Y (if exists). In the example data, x is p_T.
    #: This MUST be changed from None - no built-in default exists. Uncomment the line below default for example.
    obj.data_list = obj.AllData["model"]

    #: Dictionary for the model validation output
    #: Must be the same for as the model output dictionary
    #data_list_val = pickle.load((cachedir / 'model/validation/data_dict_val.p').open('rb'))
    obj.data_list_val = None

    #: Dictionary of the experimental data.
    #: Form MUST be exp_data_list[system][observable][subobservable][{'y':,'x':,'yerr':{'stat':,'sys'}}].
    #:      'y' is a (1 x p) numpy array of experimental data.
    #:
    #:      'x' is a (1 x p) numpy array of numeric index of columns of Y (if exists). In the example data, x is p_T.
    #:
    #:      'yerr' is a dictionary with keys 'stat' and 'sys'.
    #:
    #:      'stat' is a (1 x p) array of statistical errors.
    #:
    #:      'sys' is a (1 x p) array of systematic errors.
    #: This MUST be changed from None - no built-in default exists. Uncomment the line below default for example.
    obj.exp_data_list = obj.AllData["data"]

    #: Experimental covariance matrix.
    #: Set exp_cov = None to have the script estimate the covariance matrix.
    #: Example commented below default.
    obj.exp_cov = obj.AllData["cov"]


    #: Observables to emulate as a list of 2-tuples
    #: ``(obs, [list of subobs])``.
    obj.observables = obj.AllData["observables"]

  #---------------------------------------------------------------
  # Initialize settings as class members of obj
  #---------------------------------------------------------------
  def systems(self):

    AllData = pickle.load((self.workdir / 'input/default.p').open('rb'))

    #: Sets the collision systems for the entire project,
    #: where each system is a string of the form
    #: ``'<projectile 1><projectile 2><beam energy in GeV>'``,
    #: such as ``'PbPb2760'``, ``'AuAu200'``, ``'pPb5020'``.
    #: Even if the project uses only a single system,
    #: this should still be a list of one system string.
    return AllData["systems"]

  #---------------------------------------------------------------
  # Return formatted string of class members
  #---------------------------------------------------------------
  def __str__(self):
    s = []
    variables = self.__dict__.keys()
    for v in variables:
      s.append('{} = {}'.format(v, self.__dict__[v]))
    return "[i] {} with \n .  {}".format(self.__class__.__name__, '\n .  '.join(s))
