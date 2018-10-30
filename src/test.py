#!/usr/bin/python

"""!
Speech Transmission Index (STI) test script

Copyright (C) 2011 Jon Polom <jmpolom@wayne.edu>
Licensed under the GNU General Public License

@author Jonathan Polom <jmpolom@wayne.edu>
@version 0.5
"""

from sti import classSTI

__author__ = "Jonathan Polom <jmpolom@wayne.edu>"
__version__ = "0.5"

def testSTI():
    """!
    @brief main function to test @b classSTI object
    """
    
    csti = classSTI()
    
    # read audio
    refAudio, refRate, readstatus = csti.readwav('speech/eval1.wav')
    degrAudio, degrRate, readstatus = csti.readwav('speech/eval1_echo100.wav')

    # calculate the STI. Visually verify console output.
    vsti = csti.stiFromAudio(refAudio, degrAudio, refRate, name='eval1.wav')
    
    print("Test Result:")
    
    # test result
    if abs(vsti - 0.63) < 0.002:
        print("OK")
    else:
        print("FAILED")
        
    return vsti

if __name__ == '__main__':
    """!
    @brief main script to run
    @param[out] v_sti [array-like or float] The calculated speech transmission index (STI) value(s)
    """
    
    v_sti = testSTI()
