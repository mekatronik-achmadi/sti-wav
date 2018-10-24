#!/usr/bin/python

"""
Speech Transmission Index (STI) from speech waveforms (real speech)

Copyright (C) 2011 Jon Polom <jmpolom@wayne.edu>
Licensed under the GNU General Public License
"""

import matplotlib.mlab as matmlab
import numpy as np
import numpy.ma as npma
import scipy.io as scyio
import scipy.signal as scysig
import sys
import warnings as warns

#__author__ = "Jonathan Polom <jmpolom@wayne.edu>"
#__date__ = date(2011, 04, 22)
#__version__ = "0.5"

def thirdOctaves(minFreq, maxFreq):
    """
    Calculates a list of frequencies spaced 1/3 octave apart in hertz
    between minFreq and maxFreq

    Input
    -----
    * minFreq : float or int

        Must be non-zero and non-negative

    * maxFreq : float or int

        Must be non-zero and non-negative

    Output
    ------
    * freqs : ndarray
    """

    if minFreq <= 0 or maxFreq <= 0:
        raise ValueError("minFreq and maxFreq must be non-zero and non-negative")
    else:
        maxFreq = float(maxFreq)

        f = float(minFreq)
        freqs = np.array([f])

        while f < maxFreq:
            f = f * 10**0.1
            freqs = np.append(freqs, f)

        return freqs

def fftWindowSize(freqRes, hz):
    """
    Calculate power of 2 window length for FFT to achieve specified frequency
    resolution. Useful for power spectra and coherence calculations.

    Input
    -----
    * freqRes : float

        Desired frequency resolution in hertz

    * hz : int

        Sample rate, in hertz, of signal undergoing FFT

    Output
    ------
    * window : int
    """

    freqRes = float(freqRes)         # make sure frequency res is a float
    pwr = 1                          # initial power of 2 to try
    res = hz / float(2**pwr) # calculate frequency resolution

    while res > freqRes:
        pwr += 1
        res = hz / float(2**pwr)

    return 2**pwr

def downsampleBands(audio, hz, downsampleFactor):
    """
    Downsample audio by integer factor

    Input
    -----
    * audio : array-like

        Array of original audio samples

    * hz : float or int

        Original audio sample rate in hertz

    * downsampleFactor : int

        Factor to downsample audio by, if desired

    Output
    ------
    * dsAudio : ndarray

        Downsampled audio array

    * hz : int

        Downsampled audio sample rate in hertz
    """

    ### Achmadi here
    dsAudio = np.array([])

    # calculate downsampled audio rate in hertz
    downsampleFactor = int(downsampleFactor)        # factor must be integer
    hz = int(hz / downsampleFactor)

    for band in audio:
        ds = scysig.decimate(band, downsampleFactor, ftype='fir')

        try:
            dsAudio = np.append(dsAudio, ds)
        except:
            dsAudio = ds

    return dsAudio, hz

def octaveBandFilter(audio, hz,
                     octaveBands=[125, 250, 500, 1000, 2000, 4000, 8000],
                     butterOrd=6, hammingTime=16.6):

    """
    Octave band filter raw audio. The audio is filtered through butterworth
    filters of order 6 (by default), squared to obtain the envelope and finally
    low-pass filtered using a 'hammingTime' length Hamming filter at 25 Hz.

    Input
    -----
    * audio : array-like

        Array of raw audio samples

    * hz : float or int

        Audio sample rate in hertz

    * octaveBands : array-like

        list or array of octave band center frequencies

    * butterOrd : int

        butterworth filter order

    * hammingTime : float or int

        Hamming window length, in milliseconds relative to audio sample rate

    Output
    ------
    * octaveBandAudio : ndarray

        Octave band filtered audio

    * hz : float or int

        Filtered audio sample rate
    """

    ### Achmadi here
    octaveBandAudio = np.array([])

    print(("Butterworth filter order: %.2f") % butterOrd)
    print(("Hamming filter length:   %.2f ms") % hammingTime)
    print(("Audio sample rate:       %.2f") % hz)

    # calculate the nyquist frequency
    nyquist = hz * 0.5

    # length of Hamming window for FIR low-pass at 25 Hz
    hammingLength = (hammingTime / 1000.0) * hz

    # process each octave band
    for f in octaveBands:
        bands = str(octaveBands[:octaveBands.index(f) + 1]).strip('[]')
        statusStr = "Octave band filtering audio at: " + bands
        unitStr = "Hz ".rjust(80 - len(statusStr))
        sys.stdout.write(statusStr)
        sys.stdout.write(unitStr)
        ### Achmadi here
        # python3 on Linux need both CR and LF
        # ori: stdout.write('\r')
        sys.stdout.write('\r\n')
        sys.stdout.flush()

        # filter the output at the octave band f
        f1 = f / np.sqrt(2)
        f2 = f * np.sqrt(2)

        # for some odd reason the band-pass butterworth doesn't work right
        # when the filter order is high (above 3). likely a SciPy issue?
        # also, butter likes to complain about possibly useless results when
        # calculating filter coefficients for high order (above 4) low-pass
        # filters with relatively low knee frequencies (relative to nyquist F).
        # perhaps I just don't know how digital butterworth filters work and
        # their limitations but I think this is odd.
        # the issue described here will be sent to their mailing list
        if f < max(octaveBands):
            with warns.catch_warnings():      # suppress the spurious warnings given
                warns.simplefilter('ignore')  # under certain conditions
                b1,a1 = scysig.butter(butterOrd, f1/nyquist, btype='high')
                b2,a2 = scysig.butter(butterOrd, f2/nyquist, btype='low')

            filtOut = scysig.lfilter(b1, a1, audio)   # high-pass raw audio at f1
            filtOut = scysig.lfilter(b2, a2, filtOut) # low-pass after high-pass at f1
        else:
            with warns.catch_warnings():
                warns.simplefilter('ignore')
                b1,a1 = scysig.butter(butterOrd, f/nyquist, btype='high')
            filtOut = scysig.lfilter(b1, a1, audio)

        filtOut = np.array(filtOut)**2
        
        ### Achmadi here
        # input need to be non-ngeative integer object
        # ori: b = scysig.firwin(hammingLength, 25.0, window='hamming', nyq=nyquist)
        
        b = scysig.firwin(int(hammingLength), 25.0, window='hamming', nyq=int(nyquist))
        filtOut = scysig.lfilter(b, 1, filtOut)
        filtOut = filtOut * -1.0

        # stack-up octave band filtered audio
        try:
            octaveBandAudio = np.vstack((octaveBandAudio, filtOut))
        except:
            octaveBandAudio = filtOut

    return octaveBandAudio

def octaveBandSpectra(filteredAudioBands, hz, fftResSpec=0.06):
    """
    Calculate octave band power spectras

    Input
    -----
    * filteredAudioBands : array-like

        Octave band filtered audio

    * hz : float or int

        Audio sample rate in hertz. Must be the same for clean and dirty audio

    * fftRes : float or int

        Desired FFT frequency resolution

    Output
    ------
    * spectras : ndarray

        Power spectra values

    * fftfreqs : ndarray

        Frequencies for FFT points
    """
    ### Achmadi here
    spectras = np.array([])
    fftfreqs = np.array([])

    # FFT window size for PSD calculation: 32768 for ~0.06 Hz res at 2 kHz
    psdWindow = fftWindowSize(fftResSpec, hz)

    print("Calculating octave band power spectras")
    print(("(FFT length: %.2f samples)") % psdWindow)

    for band in filteredAudioBands:
        spectra, freqs = matmlab.psd(band, NFFT=psdWindow, Fs=hz)
        spectra = np.reshape(spectra, len(freqs))  # change to row vector
        spectra = spectra / max(spectra)        # scale to [0,1]

        # stack-up octave band spectras
        try:
            spectras = np.vstack((spectras, spectra))
            fftfreqs = np.vstack((fftfreqs, freqs))
        except:
            spectras = spectra
            fftfreqs = freqs

    return spectras, fftfreqs

def octaveBandCoherence(degrAudioBands, refAudioBands,
                        hz, fftResCoh=0.122):
    """
    Calculate coherence between clean and degraded octave band audio

    Input
    -----
    * degrAudioBands : array-like

        Degraded octave band audio

    * refAudioBands : array-like

        Reference (clean) octave band audio

    * hz : float or int

        Audio sample rate. Must be common between clean and dirty audio

    * fftRes : float or int

        Desired FFT frequency resolution

    Output
    ------
    * coherences : ndarray

        Coherence values

    * fftfreqs : ndarray

        Frequencies for FFT points
    """
    ### Achmadi here
    coherences = np.array([])
    fftfreqs = np.array([])

    # FFT window size for PSD calculation: 32768 for ~0.06 Hz res at 2 kHz
    # Beware that 'cohere' isn't as forgiving as 'psd' with FFT lengths
    # larger than half the length of the signal
    psdWindow = fftWindowSize(fftResCoh, hz)

    print("Calculating degraded and reference audio coherence")
    print(("(FFT length: %.2f samples)") % psdWindow)

    for i,band in enumerate(degrAudioBands):
        with warns.catch_warnings():      # catch and ignore spurious warnings
            warns.simplefilter('ignore')  # due to some irrelevant divide by 0's
            coherence, freqs = matmlab.cohere(band, refAudioBands[i],
                                      NFFT=psdWindow, Fs=hz)

        # stack-up octave band spectras
        try:
            coherences = np.vstack((coherences, coherence))
            fftfreqs = np.vstack((fftfreqs, freqs))
        except:
            coherences = coherence
            fftfreqs = freqs

    return coherences, fftfreqs

def thirdOctaveRootSum(spectras, fftfreqs, minFreq=0.25, maxFreq=25.0):
    """
    Calculates square root of sum of spectra over 1/3 octave bands

    Input
    -----
    * spectras : array-like

        Array or list of octave band spectras

    * fftfreqs : array-like

        Array or list of octave band FFT frequencies

    * minFreq : float

        Min frequency in 1/3 octave bands

    * maxFreq : float

        Max frequency in 1/3 octave bands

    Output
    ------
    * thirdOctaveRootSums : ndarray

        Square root of spectra sums over 1/3 octave intervals
    """
    ### Achmadi here
    sums = np.array([])
    thirdOctaveSums = np.array([])

    print("Calculating 1/3 octave square-rooted sums from")
    print(("%.2f to %.2f Hz") % (minFreq,maxFreq))

    thirdOctaveBands = thirdOctaves(minFreq, maxFreq)

    # loop over the spectras contained in 'spectras' and calculate 1/3 oct MTF
    for i,spectra in enumerate(spectras):
        freqs = fftfreqs[i]                # get fft frequencies for spectra

        # calculate the third octave sums
        for f13 in thirdOctaveBands:
            f131 = f13 / np.power(2, 1.0/6.0) # band start
            f132 = f13 * np.power(2, 1.0/6.0) # band end

            li = np.searchsorted(freqs, f131)
            ui = np.searchsorted(freqs, f132) + 1

            s = np.sum(spectra[li:ui]) # sum the spectral components in band
            s = np.sqrt(s)             # take square root of summed components

            try:
                sums = np.append(sums, s)
            except:
                sums = np.array([s])

        # stack-up third octave modulation transfer functions
        try:
            thirdOctaveSums = np.vstack((thirdOctaveSums, sums))
        except:
            thirdOctaveSums = sums

        # remove temp 'sum' and 'counts' variables for next octave band
        del(sums)

    return thirdOctaveSums

def thirdOctaveRMS(spectras, fftfreqs, minFreq=0.25, maxFreq=25.0):
    """
    Calculates RMS value of spectra over 1/3 octave bands

    Input
    -----
    * spectras : array-like

        Array or list of octave band spectras

    * fftfreqs : array-like

        Array or list of octave band FFT frequencies

    * minFreq : float

        Min frequency in 1/3 octave bands

    * maxFreq : float

        Max frequency in 1/3 octave bands

    Output
    ------
    * thirdOctaveRMSValues : ndarray

        RMS value of spectra over 1/3 octave intervals
    """

    ### Achmadi here
    sums = np.array([])
    thirdOctaveRMSValues = np.array([])

    print("Calculating 1/3 octave RMS values from")
    print(("%.2f to %.2f Hz") % (minFreq,maxFreq))

    thirdOctaveBands = thirdOctaves(minFreq, maxFreq)

    # loop over the spectras contained in 'spectras' and calculate 1/3 oct MTF
    for i,spectra in enumerate(spectras):
        freqs = fftfreqs[i]                # get fft frequencies for spectra

        # calculate the third octave sums
        for f13 in thirdOctaveBands:
            f131 = f13 / np.power(2, 1.0/6.0) # band start
            f132 = f13 * np.power(2, 1.0/6.0) # band end

            li = np.searchsorted(freqs, f131)
            ui = np.searchsorted(freqs, f132) + 1

            s = np.sum(spectra[li:ui]**2)  # sum the spectral components in band
            s = s / len(spectra[li:ui]) # divide by length of sum
            s = np.sqrt(s)                 # square root

            try:
                sums = np.append(sums, s)
            except:
                sums = np.array([s])

        # stack-up third octave modulation transfer functions
        try:
            thirdOctaveRMSValues = np.vstack((thirdOctaveRMSValues, sums))
        except:
            thirdOctaveRMSValues = sums

        # remove temp 'sum' and 'counts' variables for next octave band
        del(sums)

    return thirdOctaveRMSValues

def sti(modulations, coherences, minCoherence=0.8):
    """
    Calculate the speech transmission index from third octave modulation
    indices. The indices are truncated after coherence between clean and dirty
    audio falls below 'minCoherence' or 0.8, by default.

    Input
    -----
    * modulations : array-like

        Modulation indices spaced at 1/3 octaves within each octave band

    * coherences : array-like

        Coherence between clean and dirty octave band filtered audio

    * minCoherence : float

        The minimum coherence to include a mod index in the STI computation

    Output
    ------
    * index : float

        The speech transmission index (STI)
    """

    # create masking array of zeroes
    snrMask = np.zeros(modulations.shape, dtype=int)

    # sort through coherence array and mask corresponding SNRs where coherence
    # values fall below 'minCoherence' (0.8 in most cases and by default)
    for i,band in enumerate(coherences):
        lessThanMin = np.nonzero(band < minCoherence)[0]
        if len(lessThanMin) >= 1:
            discardAfter = min(lessThanMin)
            snrMask[i][discardAfter:] = np.ones((len(snrMask[i][discardAfter:])))

    modulations = np.clip(modulations, 0, 0.99)      # clip to [0, 0.99] (max: ~1)
    snr = 10*np.log10(modulations/(1 - modulations)) # estimate SNR
    snr = np.clip(snr, -15, 15)                      # clip to [-15,15]
    snr = npma.masked_array(snr, mask=snrMask)         # exclude values from sum
    snrCounts = (snr / snr).sum(axis=1)           # count SNRs
    snrCounts = snrCounts.data                    # remove masking
    octaveBandSNR = snr.sum(axis=1) / snrCounts   # calc average SNR
    alpha = 7 * (snrCounts / snrCounts.sum())     # calc alpha weight

    # octave band weighting factors, Steeneken and Houtgast (1985)
    w = [0.129, 0.143, 0.114, 0.114, 0.186, 0.171, 0.143]

    # calculate the STI measure
    snrp = alpha * w * octaveBandSNR
    snrp = snrp.sum()
    index = (snrp + 15) / 30.0

    print(("Speech Transmission Index (STI): %.2f") % index)
    return index

def stiFromAudio(reference, degraded, hz, calcref=False, downsample=None, name='unnamed', fftCohRes=0.122, fftSpecRes=0.06):
    """
    Calculate the speech transmission index (STI) from clean and dirty
    (ie: distorted) audio samples. The clean and dirty audio samples must have
    a common sample rate for successful use of this function.

    Input
    -----
    * reference : array-like

        Clean reference audio sample as an array of floating-point values

    * degraded : array-like

        Degraded audio sample as an array, or array of arrays for multiple
        samples, of floating-point values

    * hz : int

        Audio sample rate in hertz

    * calcref : boolean

        Calculate STI for reference signal alone

    * downsample : int or None

        Downsampling integer factor

    * name : string

        Name of sample set, for output tracking in larger runs

    Output
    ------
    * sti : array-like or float

        The calculated speech transmission index (STI) value(s)
    """
    ### Achmadi here
    thirdOctaveTemps = np.array([])
    stiValues = np.array([])

    # put single sample degraded array into another array so the loop works
    if type(degraded) is not type([]):
        degraded = [degraded]

    print ("--------------------------------------------------------")
    print ("Speech Transmission Index (STI) from speech waveforms")
    print ("--------------------------------------------------------")
    print ("")
    print (("Sample set:             %s") % name)
    print (("Number of samples:      %.2f") % len(degraded))
    print ("Calculate reference STI:")
    if calcref:
        print ("yes")
    else:
        print ("no")
    print ("")
    print (" Reference Speech ")

    refOctaveBands = octaveBandFilter(reference, hz)
    refRate = hz

    # downsampling, if desired
    if type(downsample) is type(1):
        refOctaveBands, refRate = downsampleBands(refOctaveBands, refRate,
                                                  downsample)

    # calculate STI for reference sample, if boolean set
    if calcref:
        # STI calc procedure
        spectras, sfreqs = octaveBandSpectra(refOctaveBands, refRate, fftSpecRes)
        coherences, cfreqs = octaveBandCoherence(refOctaveBands,refOctaveBands,
                                                 refRate, fftCohRes)
        thirdOctaveMTF = thirdOctaveRootSum(spectras, sfreqs)
        thirdOctaveCoherences = thirdOctaveRMS(coherences, cfreqs)

        # add to interim array for MTFs and coherences
        try:
            thirdOctaveTemps.append([thirdOctaveMTF, thirdOctaveCoherences])
        except:
            thirdOctaveTemps = [[thirdOctaveMTF, thirdOctaveCoherences]]

    # loop over degraded audio samples and calculate STIs
    for j,sample in enumerate(degraded):
    	### Achmadi here
    	# originally, print j is print j+1
        print((" Degraded Speech: Sample {%.2f}") % j)
        degrOctaveBands = octaveBandFilter(sample, hz)
        degrRate = hz

        # downsampling, if desired
        if type(downsample) is type(1):
            degrOctaveBands, degrRate = downsampleBands(degrOctaveBands,
                                                        degrRate, downsample)

        # STI calc procedure
        spectras, sfreqs = octaveBandSpectra(degrOctaveBands, degrRate, fftSpecRes)
        coherences, cfreqs = octaveBandCoherence(refOctaveBands,
                                                 degrOctaveBands, refRate, fftCohRes)
        thirdOctaveMTF = thirdOctaveRootSum(spectras, sfreqs)
        thirdOctaveCoherences = thirdOctaveRMS(coherences, cfreqs)

        # add to interim array for MTFs and coherences
        try:
            thirdOctaveTemps.append([thirdOctaveMTF, thirdOctaveCoherences])
        except:
            thirdOctaveTemps = [[thirdOctaveMTF, thirdOctaveCoherences]]

    # calculate the STI values
    print (" Speech Transmission Index ")
    for i in range(0,len(thirdOctaveTemps)):
        sampleSTI = sti(thirdOctaveTemps[i][0], thirdOctaveTemps[i][1])

        # add to STI output array
        try:
            stiValues.append(sampleSTI)
        except:
            stiValues = [sampleSTI]

    # unpack single value
    if len(stiValues) == 1:
        stiValues = stiValues[0]

    return stiValues

def readwav(path):
    """
    Reads Microsoft WAV format audio files, scales integer sample values and
    to [0,1]. Returns a tuple consisting of scaled WAV samples and sample rate
    in hertz.

    Input
    -----
    * path : string

        Valid system path to file

    Output
    ------
    * audio : array-like

        Array of scaled sampled

    * rate : int

        Audio sample rate in hertz
    """
    try:
        wav = scyio.wavfile.read(path)
        
    except:        
        print("error read wav file %s \n\n" % path)
        return 0, 0, 1
        
    status = 0
    rate = wav[0]
    audio = np.array(wav[1])

    scale = float(max(audio))
    audio = audio / scale

    return audio, rate, status
