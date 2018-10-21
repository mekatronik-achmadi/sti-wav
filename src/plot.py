#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 20:48:08 2018

@author: achmadi
"""

from sti import readwav
import matplotlib.pyplot as plt

def plotSTI():
    # read audio
    refAudio, refRate = readwav('speech/eval1.wav')
    degrAudio, degrRate = readwav('speech/eval1_echo100.wav')
    
    plt.figure(0)
    plt.subplot(211)
    plt.title('Reference Wave...')
    plt.plot(refAudio)
    plt.subplot(212)
    plt.title('Degraded Wave...')
    plt.plot(degrAudio)
    
    plt.show()
   
if __name__ == '__main__':
    plotSTI()
    