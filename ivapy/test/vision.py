#==================================== vision ===================================
## @addtogroup ivapy_testing
# @{
# @package vision
# @brief   Simple image generation for testing vision algorithms.

import numpy as np


def squareInImage(imSize, rLoc, rad, col):
  """!
  @brief    Draw a square in the image as given location.

  @param[in]    imSize  The image dimensions.
  @param[in]    rLoc    The image coordinates of square center.
  @param[in]    rad     Radius (integer) of the square (+/- from center).
  @param[in]    colspec The color specification of the image (1D = gray, 3D = rgb).
  """

  img = np.matlib.repmat(np.zeros(np.shape(col)), imSize)
  ii = np.range(rLoc[2]-rad:(rLoc[2]+rad+1))
  jj = np.range(rLoc[1]-rad:(rLoc[1]+rad+1))

  img[ii,jj,:] = col



## @}
# 
#==================================== vision ===================================
