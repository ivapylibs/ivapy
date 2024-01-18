#==================================== vision ===================================
## @addtogroup ivapy_testing
# @{
# @package vision
# @brief   Simple image generation for testing vision algorithms.

import numpy as np
import numpy.matlib as npml


def squareInImage(imSize, rLoc, rad, col):
  """!
  @brief    Draw a square in the image as given location.

  @param[in]    imSize  The image dimensions.
  @param[in]    rLoc    The image coordinates of square center.
  @param[in]    rad     Radius (integer) of the square (+/- from center).
  @param[in]    colspec The color specification of the image (1D = gray, 3D = rgb).
  """

  img = npml.repmat(np.zeros(np.shape(col)), imSize[0], imSize[1])
  ii = np.array(range((rLoc[1,0]-rad),(rLoc[1,0]+rad+1)))
  jj = np.array(range((rLoc[0,0]-rad),(rLoc[0,0]+rad+1)))

  if (len(np.shape(img)) == 2):
    img[ii,jj] = col
  else:
    img[ii,jj,:] = col

  return img


## @}
# 
#==================================== vision ===================================
