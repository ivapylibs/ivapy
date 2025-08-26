#=================================== helper ==================================
'''!
@brief  Implements a variety of helper functions.
'''
#=================================== helper ==================================
#
# @file     helper.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/07/XX
#
# NOTE: 90 columns, 2 space indent, wrap at 4.
#
#=================================== helper ==================================


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))

#=================================== helper ==================================
