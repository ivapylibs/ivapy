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

  #print('DDDDDDDDD')
  #print(rLoc)
  #print(rLoc - rad)
  #print(all(rLoc - rad > 0))
  #print('UUUUUUUUU')
  #print(rLoc + rad)
  #print(imSize)
  #print(rLoc + rad < imSize)
  #print(all(rLoc + rad < imSize))
  #print('---------')

  loLimOK = all(rLoc - rad > 0)
  upLimOK = all(rLoc + rad < imSize[-1])

  if (loLimOK and upLimOK):
    rLoc = rLoc.astype('int')

    if (len(np.shape(img)) == 2):
      img[(rLoc[1]-rad):(rLoc[1]+rad+1),(rLoc[0]-rad):(rLoc[0]+rad+1)] = col
    else:
      img[ii,jj,:] = col

  return img


## @}
# 
#==================================== vision ===================================
