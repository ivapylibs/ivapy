#==================================== paths ====================================
## @addtogroup ivapy_testing
# @{
# @package paths
# @brief   Simple path generation for testing signals.
#

import numpy as np
from ivapy.Configuration import AlgConfig
import scipy.interpolate as curve


class CfgStepLines(AlgConfig):
  """!
  @brief    Configuration specifier for piecewise lines instance.

  General set of options for generating the points along piecewise linear path
  specified from waypoints/breakpoints.

  | Property | Meaning |
  | :------- | :------ |
  } numPoints | Number of points along a given line segment. |


  Just doing numpoints so it is quick and dirty to create.  Later have
  more options.  Like total number of points, interpolation by length
  or arc length or possibly some scalar (fictitious time).
  """

  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgStepLines.get_default_settings()

    super().__init__(init_dict, key_list, new_allowed)

    # self.merge_from_lists(XX)

  #------------------------- get_default_settings ------------------------
  #
  # @brief    Recover the default settings in a dictionary.
  #
  @staticmethod
  def get_default_settings():
    '''!
    @brief  Defines most basic, default settings for RealSense D435.

    @param[out] default_dict  Dictionary populated with minimal set of
                              default settings.
    '''

    default_dict = dict(numPoints = 10, isPeriodic = False)
    return default_dict



class StepLines:
  """!
  @brief    Given a set of points representing piecewise line segments, provides
            an interface for generating them as a path of discrete points/steps.

  This is perhaps one of the simplest implementations.  Each pair of waypoints defines a
  segment, each of which is broken up into a set number of smaller segments along the
  line connecting the waypoints.  There is no notion of time, only a discrete step
  number.  If the path is defined to be periodic, then the next points after the last
  point will be the first point of the path. No concern is given to how nice that jump
  is relative to the path.
  """

  def __init__(self, params, waypoints):
     """!
     @brief     Construct instance based on parameters and waypoints array.

     @param[in] params      Parameter settings.
     @param[in] waypoints   Array of line segment end-points.
     """
     
     if (params is None):
       params = CfgStepLines()

     self.settings  = params        # @< Implementation settings.
     self.points    = waypoints     # @< Set of points seeding the path.
     self.isReady   = False         # @< Is the instance ready to output steps?

     self.path  = None              # @< The synthesized path steps.
     self.t     = 0                 # @< Line parameter indicating location along path.
     self.tMax  = 0

     self.build()
     pass

  #================================ build ================================
  #
  def build(self):
    """!
    @brief  Build the path from the specification, presuming it has been given.
    """
    if self.points is None:
      return

    npts = np.shape(self.points)[1]
    tIn  = np.array(range(0,npts))
    tOut = np.array(range(0,(npts-1)*self.settings.numPoints)) / self.settings.numPoints

    pFun = curve.interp1d(tIn, self.points)

    self.path = pFun(tOut)
    self.t    = 0
    self.tMax = (npts-1)*self.settings.numPoints-1
    self.isReady = True


  def next(self):
    if not self.isReady:
      return None

    pt     = self.path[:,self.t]
    self.t = self.t+1
    if (self.t > self.tMax):
      if(self.settings.isPeriodic):
        self.t = 0
      else:
        self.t = self.tMax

    return pt


## @}
#
#==================================== paths ====================================
