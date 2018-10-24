#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:50:08 2018

@author: achmadi
"""

import os
import fnmatch

from sti import classSTI

def testSTI():
    
    csti = classSTI()

    path = "UntukSTI_54"
    fresult = "result_%s.csv" % path
    pattern = "*_Ref.wav"
    strsplit = "_"    
  
    with open(fresult, 'w') as txtfile:
        txtfile.write("NO, FILE_EVAL, FILE_REF, STI \n")
    
    lst_fref = fnmatch.filter(os.listdir(path), pattern)     
        
    for num, fname in enumerate(lst_fref):
        fsplit = fname.split(strsplit)
        
        fwav = "%s_%s.wav" % (fsplit[0],fsplit[1])
        fwavref = "%s_%s_%s" % (fsplit[0],fsplit[1],fsplit[2])
        
        pfwav = "%s/%s" % (path,fwav)
        pfwavref = "%s/%s" % (path,fwavref)
        
        print("processing %s vs %s \n\n" % (fwav,fwavref))
        
        degrAudio, degrRate, readstt = csti.readwav(pfwav)
        refAudio, refRate, readrefstt = csti.readwav(pfwavref)
        
        if readstt==0 and readrefstt==0:
            v_sti = csti.stiFromAudio(refAudio, degrAudio, refRate, name=fwav, fftCohRes=0.976)
        
            with open(fresult, 'a') as txtfile:
                txtfile.write("%i, %s, %s, %.2f \n" % (num,fwav,fwavref,v_sti))
                
        else:
            with open(fresult, 'a') as txtfile:
                txtfile.write("%i, %s, %s, error-wav \n" % (num,fwav,fwavref))
        
if __name__ == '__main__':
    testSTI()
        
   