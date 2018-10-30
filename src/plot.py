#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
@author achmadi
"""

from sti import classSTI
import matplotlib.pyplot as plt

def plotSTI():
    """!
    @brief main function to plot wav (degraded vs references) 
    """
    
    csti = classSTI()
    
    # read audio
    refAudio, refRate, readstt = csti.readwav('speech/eval1.wav')
    degrAudio, degrRate, readstt = csti.readwav('speech/eval1_echo100.wav')
    
    plt.figure(0)
    plt.subplot(211)
    plt.title('Reference Wave...')
    plt.plot(refAudio)
    plt.subplot(212)
    plt.title('Degraded Wave...')
    plt.plot(degrAudio)
    
    plt.show()

if __name__ == '__main__':
    """!
    @brief main script to run
    """
    
    plotSTI()
    
