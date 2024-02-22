#================================== display_cv =================================
## @addtogroup Display_CV
# @{
#
# @package  display_cv
# @brief    Display helper routines using OpenCV
#
# @author   Yiye Chen                   yychen2019@gatech.edu
# @author   Patricio A. Vela            pvela@gatech.edu
#
# @date 2021/10/07       [created, Surveillance repository]
# @date 2021/10/19       [moved to the camera repository]
# @date 2023/12/14       [moved to ivapy repository]
#

#================================= display_cv ==================================

from typing import Callable, List
from dataclasses import dataclass

import numpy as np
import cv2

import improcessor.basic as improcessor

@dataclass
class plotfig:
  num   : any
  axes  : any


#
#-------------------------------------------------------------------------
#======================== OpenCV Display Interface =======================
#-------------------------------------------------------------------------
#



#=============================== close ==============================
#
# @brief    Close the displayed window.
#
def close(window_name):
  """!
  @brief    Close a given window by name.

  @param[in]    window_name     Named window to close.
  """
  cv2.destroyWindow(window_name)

#================================== wait =================================
#
def wait(dur=0):
  """!
  @brief    Wait specified duration (ms) for keypress. Usually forces display refresh.

  Much like Matlab, this wait also serves as a trigger to udpate visuals for OpenCV.
  Otherwise, processing of main loop will continue and pre-empty graphics refreshes.

  @param[in]    dur     [0 ms] Optional duration in ms of waiting for keypress.
  """
  opKey = cv2.waitKey(dur)
  return opKey


#===================== OpenCV Single Image Interfaces ====================
#-------------------------------------------------------------------------


#================================= depth =================================
#
def depth(depIm, depth_clip=0.08, ratio=None, window_name="OpenCV Display"):
    """!
    @brief  Display depth image using OpenCV window.

    The depth frame will be scaled to have the range 0-255.
    There will be no color bar for the depth

    Args:
        depIm (np.ndarray, (H, W)): The depth map
        depth_clip (float in [0, 1]): The depth value clipping percentage. The top and bottom extreme depth value to remove. Default to 0.0 (no clipping)
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    """

    # Apply depth clipping if requested.
    # Convert depth to cv2 image format: scale to 255, convert to 3-channel
    if depth_clip > 0:
        assert depth_clip < 0.5
        improc = improcessor.basic(improcessor.basic.clipTails,(depth_clip,))
        depth_show = improc.apply(depth)
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_show, alpha=255./depth_show.max(), beta=0), cv2.COLORMAP_JET)
    else:
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depIm, alpha=255./depth.max(), beta=0), cv2.COLORMAP_JET)

    # Rescale if requested.
    if ratio is not None: 
        H, W = depIm.shape[:2]
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
        depth_color = cv2.resize(depth_color, (W_vis, H_vis))

    # Display colorized depth image.
    cv2.imshow(window_name, depth_color[:,:,::-1])

#================================== rgb ==================================
#
def rgb(rgb, ratio=None, window_name="Image"):
    """!
    @brief  Display rgb image using the OpenCV

    The rgb frame will be resized to a visualization size prior to visualization.
    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    """
    if ratio is not None:                                   # Resize if requested.
        H, W = rgb.shape[:2]
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
        rgb_vis = cv2.resize(rgb, (W_vis, H_vis))
        cv2.imshow(window_name, rgb_vis[:,:,::-1])
    else:
        cv2.imshow(window_name, rgb[:,:,::-1])

#================================== bgr ==================================
#
def bgr(bgr, ratio=None, window_name="Image"):
    '''!
    @brief  Display bgr image using the OpenCV

    The bgr frame will be resized to a visualization size prior to visualization.
    Args:
        bgr (np.ndarray, (H, W, 3)): The bgr image
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    '''
    if ratio is not None:                                   # Resize if requested.
        H, W = bgr.shape[:2]
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
        bgr_vis = cv2.resize(bgr, (W_vis, H_vis))
        cv2.imshow(window_name, bgr_vis)
    else:
        cv2.imshow(window_name, bgr)

#================================== gray =================================
#
def gray(gim, ratio=None, window_name="Image"):

    '''!
    @brief  Display grayscale image using the OpenCV

    The rgb frame will be resized to a visualization size prior to visualization.
    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    '''
    if ratio is not None:                                   # Resize if requested.
        H, W = gim.shape[:2]
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
        gim_vis = cv2.resize(gim, (W_vis, H_vis))
        cv2.imshow(window_name, gim_vis)
    else:
        cv2.imshow(window_name, gim)

#================================= binary ================================
#
def binary(bIm, ratio=None, window_name="Binary"):
    '''!
    @brief  Display binary image using OpenCV display routines.

    The binary frame will be resized as indicated prior to visualization.

    @param[in]  bIm     Binary image as a numpy H x W ndarray.
    @param[in]  ratio   Resize ratio of image to display (float, optional).
                        Default is None = no resizing.
    @param[in]  window_name     Window display name (Default = "Binary").    
    '''

    cIm = cv2.cvtColor(bIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

    if ratio is not None:                                   # Resize if requested.
        H, W = bIm.shape[:2]
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
        cIm = cv2.resize(cIm, (W_vis, H_vis))
        cv2.imshow(window_name, cIm)
    else:
        cv2.imshow(window_name, cIm)

#=============================== trackpoint ==============================
#
def trackpoint(rgb, p, ratio=None, window_name="Image"):
    '''!
    @brief  Display rgb image with trackpoint overlay. 

    The rgb frame will be resized to a visualization size prior to visualization.

    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    '''
    H, W = rgb.shape[:2]
    if ratio is None:                                   # Resize if requested.
        rgb_vis = cv2.resize(rgb, (W, H))
    else:
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
        rgb_vis = cv2.resize(rgb, (W_vis, H_vis))
        p = ratio * p

    p_vis = np.fix(p)

    cv2.drawMarker(rgb_vis, (int(p_vis[0,0]),int(p_vis[1,0])), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
    cv2.imshow(window_name, rgb_vis[:,:,::-1])

#============================== trackpoints ==============================
#
def trackpoints(rgb, p, ratio=None, window_name="Image"):
    '''!
    @brief  Display rgb image with multiple trackpoints overlaid.

    The rgb frame will be resized to a visualization size prior to visualization.
    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    '''
    if ratio is None:                                   # Resize if requested.
        rgb_vis = rgb
    else:
        H, W = rgb.shape[:2]
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
        rgb_vis = cv2.resize(rgb, (W_vis, H_vis))
        p = ratio * p

    p_vis = np.fix(p)

    nTPs = np.shape(p_vis)[1]
    for ii in range(nTPs):
      cv2.drawMarker(rgb_vis, (int(p_vis[0,ii]),int(p_vis[1,ii])), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
    
    cv2.imshow(window_name, rgb_vis[:,:,::-1])


#=========================== trackpoint_binary ===========================
#
def trackpoint_binary(bIm, p, ratio=None, window_name="Image"):
    '''!
    @brief  Display binary image with trackpoint overlay.

    The rgb frame will be resized to a visualization size prior to visualization.
    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    '''
    cIm = cv2.cvtColor(bIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
    trackpoint(cIm, p, ratio, window_name)

#============================ trackpoint_gray ============================
#
def trackpoint_gray(gIm, p, ratio=None, window_name="Image"):

    '''!
    @brief  Display grayscale image with trackpoint overlay.

    The rgb frame will be resized to a visualization size prior to visualization.
    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    '''
    cIm = cv2.cvtColor(gIm, cv2.COLOR_GRAY2BGR)
    trackpoint(cIm, p, ratio, window_name)


#============================== polyline_rgb =============================
#
def polyline_rgb(I, polyPts, isClosed = False, lineColor = (255,255,255) , 
                             lineThickness = 2, window_name = "Poly+Image"):
  """!
  @brief    Annotate image with sequential line segments from mouse input (open or closed path).

  @param[in]    I
  @param[in]    polyPts
  @param[in]    isClosed
  @param[in]    lineColor
  @param[in]    lineThickness
  @param[in]    window_name
  """
  if (polyPts is not None):
    Imark = I.copy()
    cv2.polylines(Imark, [np.transpose(polyPts)], isClosed, lineColor, lineThickness)
    rgb(Imark, window_name = window_name)
  else:
    rgb(I, window_name = window_name)


#================== OpenCV Side-by-Side Image Interfaces =================
#-------------------------------------------------------------------------

#================================= images ================================
#
#
def images(images:list, ratio=None, window_name="OpenCV Display"):
    """!
    @brief  Display a set of images horizontally concatenated (side-by-side).

    Args:
        images (list): A list of the OpenCV images (bgr) of the same size
        window_name (str, Optional): The window name for display. Defaults to \"OpenCV display\"
    """
    # Resized image dimensions. 
    H, W = images[0].shape[:2]
    if ratio is not None:
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
    else:
        H_vis = H
        W_vis = W

    # Resize if requested. Stack images horizontally. Display w/imshow.
    images_vis = [cv2.resize(img, (W_vis, H_vis)) for img in images]
    image_display = np.hstack(tuple(images_vis))
    cv2.imshow(window_name, image_display)


#=============================== rgb_depth ===============================
#
def rgb_depth(rgb, depth, depth_clip=0.08, ratio=None, window_name="OpenCV Display"):
    """!
    @brief  Display rgb and depth images side-by-side using OpenCV.

    Depth frame will be scaled to range [0,255].
    The rgb and the depth images will be resized to visualization size, then concatenated horizontally.
    The concatenated image will be displayed in a window.
    There will be no color bar for the depth

    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        depth (np.ndarray, (H, W)): The depth map
        depth_clip (float in [0, 1]): The depth value clipping percentage. The top and bottom extreme depth value to remove. Default to 0.0 (no clipping)
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    """
    # depth clip
    if depth_clip > 0:
        assert depth_clip < 0.5
        improc = improcessor.basic(improcessor.basic.clipTails,(depth_clip,))
        depth_show = improc.apply(depth)
    else:
        depth_show = depth

    # convert the depth to cv2 image format: scale to 255, convert to 3-channel
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_show, alpha=255./depth_show.max(), beta=0), cv2.COLORMAP_JET)

    # diplay
    images((rgb[:,:,::-1], depth_colormap), ratio=ratio, window_name=window_name)

#============================ rgb_binary ============================
#
def rgb_binary(cIm, bIm, ratio=None, window_name="Color+Binary"):
    """!
    @brief  Display an RGB and binar image side-by-side using OpenCV API.

    The images will be resized as indicated by the ratio and concatenated
    horizontally. The concatenated image is what gets displayed in the window.

    @param[in]  cIm             Color image (RGB).
    @param[in]  bIm             Binary image (0/1 or logical)
    @param[in]  ratio           Resizing ratio.
    @param[in]  window_name     Window name for display.
    """

    cBm = cv2.cvtColor(bIm.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

    images((cIm[:,:,::-1], cBm), ratio=ratio, window_name=window_name)


#============================= To Be Removed =============================
#- BELOW THIS LINE -------------------------------------------------------

# @todo Should delete this function.
def display_rgb_dep(rgb, depth, depth_clip=0.08, ratio=None, window_name="OpenCV Display"):

    """Display the rgb and depth image using the OpenCV

    The depth frame will be scaled to have the range 0-255.
    The rgb and the depth frame will be resized to a visualization size and then concatenate together horizontally 
    Then the concatenated image will be displayed in a same window.

    There will be no color bar for the depth

    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        depth (np.ndarray, (H, W)): The depth map
        depth_clip (float in [0, 1]): The depth value clipping percentage. The top and bottom extreme depth value to remove. Default to 0.0 (no clipping)
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    """
    # depth clip
    if depth_clip > 0:
        assert depth_clip < 0.5
        improc = improcessor.basic(improcessor.basic.clipTails,(depth_clip,))
        depth_show = improc.apply(depth)
    else:
        depth_show = depth

    # convert the depth to cv2 image format: scale to 255, convert to 3-channel
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_show, alpha=255./depth_show.max(), beta=0), cv2.COLORMAP_JET)

    # diplay
    images((rgb[:,:,::-1], depth_colormap), ratio=ratio, window_name=window_name)

# @todo Should delete this function.
def display_dep(depth, depth_clip=0.08, ratio=None, window_name="OpenCV Display"):

    '''!
    @brief  Display depth image using OpenCV window.

    The depth frame will be scaled to have the range 0-255.
    There will be no color bar for the depth

    Args:
        depth (np.ndarray, (H, W)): The depth map
        depth_clip (float in [0, 1]): The depth value clipping percentage. The top and bottom extreme depth value to remove. Default to 0.0 (no clipping)
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    '''
    # Apply depth clipping if requested.
    # Convert depth to cv2 image format: scale to 255, convert to 3-channel
    if depth_clip > 0:
        assert depth_clip < 0.5
        improc = improcessor.basic(improcessor.basic.clipTails,(depth_clip,))
        depth_show = improc.apply(depth)
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_show, alpha=255./depth_show.max(), beta=0), cv2.COLORMAP_JET)
    else:
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255./depth.max(), beta=0), cv2.COLORMAP_JET)

    # Rescale if requested.
    if ratio is not None: 
        H, W = depth.shape[:2]
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
        depth_color = cv2.resize(depth_color, (W_vis, H_vis))

    # Display colorized depth image.
    cv2.imshow(window_name, depth_color[:,:,::-1])

#-- ABOVE THIS LINE ------------------------------------------------------
#============================= To Be Removed =============================

#==================== OpenCV Single Window Mouse Input ===================
#-------------------------------------------------------------------------

#============================== polyline_rgb =============================
#
def polyline_rgb(I, polyPts, isClosed = False, lineColor = (255,255,255) , 
                             lineThickness = 2, window_name = "Poly+Image"):

  if (polyPts is not None):
    Imark = I.copy()
    cv2.polylines(Imark, [np.transpose(polyPts)], isClosed, lineColor, lineThickness)
    rgb(Imark, window_name = window_name)
  else:
    rgb(I, window_name = window_name)

#============================== markers_rgb ==============================
#
def markers_rgb(I, thePts, ptColor = (255,255,255) , 
                           ptMarkType = cv2.MARKER_CROSS,
                           ptSize = 20, ptThickness = 1, ptLineType = 8, 
                           window_name = "Marked Image"):

  if thePts is None:
    rgb(I, window_name = window_name)
  else:
    psize = np.shape(thePts)

    Imark = I.copy()
    for ii in range(0,psize[1]):
      cv2.drawMarker(Imark, thePts[:,ii], ptColor, ptMarkType , ptSize, \
                                                   ptThickness, ptLineType)

    rgb(Imark, window_name = window_name)


#=============================== getline_rgb =============================
#
def getline_rgb(I, isClosed = False, window_name = "Image"):
  """!
  @brief    Quick and dirty equivalent implementation to Matlab's getline.

  @param[in]    I               Image to display.
  @param[in]    window_name     Name of existing window to use.
  @param[in]    isClosed        Return as closed polygon.
  """

  # Obtained original code from:
  #
  # https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
  #
  # and modified to provide desired functionality.

  # Function to process mouse clicks. 
  #
  def click_event(event, x, y, flags, params):
  
    nonlocal pts

    # Check for left mouse click: adds point.
    if event == cv2.EVENT_LBUTTONDOWN:
      if pts is None:
        pts = np.array( [[x],[y]] )
      else:
        pts = np.append( pts, [[x],[y]] , axis=1)
  
      polyline_rgb(I, pts, window_name = window_name, isClosed = isClosed)
  
    # Checking for right mouse clicks: removes last point.
    if event==cv2.EVENT_RBUTTONDOWN:

      if pts is not None:
        pts = pts[:, :-1]

      polyline_rgb(I, pts, window_name = window_name, isClosed = isClosed)
  
  
  # driver function
  pts   = None
  rgb(I, window_name = window_name)

  # setting mouse handler for the image
  # and calling the click_event() function
  cv2.setMouseCallback(window_name, click_event)

  # wait for a key to be pressed to exit
  cv2.waitKey(0)

  # close the window
  close(window_name)

  # return list of points.
  if (pts is not None) and isClosed:
    pts = np.append( pts, np.transpose([pts[:,0]]), axis=1 )
    # Not sure why requires transpose / returns a row vector from column request.

  return pts


#============================ getlinemask_rgb ============================
#
def getlinemask_rgb(I, window_name = "Image"):
  """!
  @brief    Quick and dirty polyline input to mask output. Closed path.

  @param[in]    I               Image to display.
  @param[in]    window_name     Name of existing window to use.
  """

  thePolyLine = getline_rgb(I, True, window_name)

  imsize = np.shape(I)
  theMask = np.full(imsize[0:2], 0, dtype=np.uint8)
  if thePolyLine is not None:
    cv2.fillPoly(theMask, [np.transpose(thePolyLine)], 255)

  return theMask.astype(dtype=bool)


#-------------------------------------------------------------------------
#========================== Image Stream Viewing =========================
#-------------------------------------------------------------------------

#========================= rgbd_wait_for_confirm =========================
#
#
def rgbd_wait_for_confirm(rgb_dep_getter:Callable, color_type="rgb", window_name = "display",
        instruction = "Press \'c\' key to select the frames. Press \'q\' to return None", 
        capture_click = False,
        ratio = None):
    """!
    @brief  Get imagery from callable, present to user, and process on user input.

    An interface function for letting the user select the desired frame from the given sensor
    source. The function will display the color and the depth information received from the
    source, and then wait for the user to confirm via keyboard. 
    
    NOTE: can't distinguish different keys for now. So only support "press any key to confirm"

    @param[in]  color_dep_getter    (Callable): RGBD source. \
            Expect to get the sensor information in the np.ndarray format via: \
                        rgb, depth = color_dep_getter() \
            When there is no more info, expected to return None
    @param[in]  color_type          (str): Color type. RGB or BGR. 
    @param[in]  window_name         (str, optional): Window name 
    @param[in]  instruction         (str, optional): Instruction text printed to user for
                                    selection. Defaults to None.
    @param[in]  ratio               (float, optional): Image rescaling before display.  \
                                    Default is None, no resizing.

    @param[out] rgb                 [np.ndarray]: RGB image confirmed/chosen by user
    @param[out] depth               [np.ndarray]: Depth image confirmed/chosen by the user
    """
    # get the next stream of data
    rgb, dep = rgb_dep_getter()

    # Display instruction
    print(instruction)

    # get started
    while((rgb is not None) and (dep is not None)):

        # visualization 
        display_rgb_dep(rgb, dep, window_name=window_name, ratio=ratio)

        # wait for confirm
        opKey = cv2.waitKey(1)
        if opKey == ord('c'):
            break
        elif opKey == ord('q'):
            return None, None
        
        # if not confirm, then go to the next stream of data
        rgb, dep = rgb_dep_getter()

    cv2.destroyWindow(window_name)

    return rgb, dep

#=============================== getpts_rgb ==============================
#
def getpts_rgb(I, window_name = "Image"):
  """!
  @brief    Quick and dirty equivalent implementation to Matlab's getpts.

  @param[in]    I               Image to display.
  @param[in]    window_name     Name of existing window to use.
  @param[in]    isClosed        Return as closed polygon.
  """

  # See getline_rgb for code inspiration.
  #
  def click_event(event, x, y, flags, params):
  
    nonlocal pts

    # Check for left mouse click: adds point.
    if event == cv2.EVENT_LBUTTONDOWN:
      if pts is None:
        pts = np.array( [[x],[y]] )
      else:
        pts = np.append( pts, [[x],[y]] , axis=1)
  
      markers_rgb(I, pts, window_name = window_name)
  
    # Checking for right mouse clicks: removes last point.
    if event==cv2.EVENT_RBUTTONDOWN:

      if pts is not None:
        pts = pts[:, :-1]

      markers_rgb(I, pts, window_name = window_name)
  
  
  # driver function
  pts   = None
  rgb(I, window_name = window_name)

  # setting mouse handler for the image
  # and calling the click_event() function
  cv2.setMouseCallback(window_name, click_event)

  # wait for a key to be pressed to exit
  cv2.waitKey(0)

  # close the window
  close(window_name)

  return pts

## @}
#
#================================= display ==================================
  
# @quitf
#
# NOTE: MOST LIKELY DOES NOT WORK. CODE HAS BEEN UPDATED. NEED TO REVISE OR DELETE.
if __name__ == "__main__":
    imgSource = lambda : (
        (np.random.rand(100,100,3) * 255).astype(np.uint8), \
        np.random.rand(100,100)
    )
    color, dep = wait_for_confirm(imgSource, color_type="rgb", instruction="This is just a dummy example. Press the \'c\' key to confirm", \
        ratio=2)
    # visualize
    rgb_depth(color, dep, window_name="The selected sensor info from the  wait_for_confirm example. Use the OpenCv for display")
    wait(1)
    #plt.show()
    
#
#================================= display ==================================
