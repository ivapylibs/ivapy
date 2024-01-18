#==================================== paths ====================================
## @addtogroup ivapy_testing
# @{
# @package paths
# @brief   Simple path generation for testing signals.
#

import numpy as np
from ivapy.Configuration import AlgConfig


class CfgPiecewiseLines(AlgConfig):
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
      init_dict = CfgPiecewiseLines.get_default_settings()

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



class PiecewiseLines:
  """!
  @brief    Given a set of points representing piecewise line segments, provides
            an interface for generating them as a path.
  """

  def __init__(self, params = CfgPiecewiseLines(), waypoints = None):
     """!
     @brief     Construct instance based on parameters and waypoints array.

     @param[in] params      Parameter settings.
     @param[in] waypoints   Array of line segment end-points.
     """
     self.settings  = params
     self.points    = waypoints
     self.isReady   = False

     self.path  = None
     self.t     = 0

     self.build()
     pass

  def build(self):
    """!
    @brief  Build the path from the specification.
    """
    thePath = np.array([])
    pass


  def next(self):
    pass


## @}
#
#==================================== paths ====================================
