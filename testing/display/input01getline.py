#!/usr/bin/python
#================================== getline01basic =================================
"""!
@brief  Simple demonstration of how to use getline_rgb function.

"""
#================================== getline01basic =================================
#
# @file     getline01basic.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/12/14          [created]
#
#
# NOTE: 90 column width, 2 space indent, wrap margin at 6.
#
#================================== getline01basic =================================


#==[0] Prep environment.
#
import numpy as np
import ivapy.display_cv as disp


#==[1] Create fake data.
#
I = np.zeros( (300, 300, 3) )
I[100:200,100:200,0] = 250
I[150:250,150:250,2] = 250


#==[2] Get user input and display final list of points in the polyline.
#
pts = disp.getline_rgb(I,isClosed = False)
print(pts)


#
#================================== getline01basic =================================

