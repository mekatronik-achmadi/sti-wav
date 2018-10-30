function [ coherences, fftfreqs ] = octaveBandSpectra(degrAudioBands, refAudioBands,hz, fftRes=0.122)
% Calculate coherence between clean and degraded octave band audio
% 
% Input
% -----
% * degrAudioBands : array-like
% 
%     Degraded octave band audio
% 
% * refAudioBands : array-like
% 
%     Reference (clean) octave band audio
% 
% * hz : float or int
% 
%     Audio sample rate. Must be common between clean and dirty audio
% 
% * fftRes : float or int
% 
%     Desired FFT frequency resolution
% 
% Output
% ------
% * coherences : ndarray
% 
%     Coherence values
% 
% * fftfreqs : ndarray
% 
%     Frequencies for FFT points
    if isempty(fftRes)
        fftRes=0.122;
    end

    psdWindow = fftWindowSize(fftRes, hz);
    
    fprintf('Calculating octave band power spectras\n');
    fprintf('FFT length: %d samples',psdWindow);
    
    for band=filteredAudioBands
        [spectra, freqs] = my_psd(band, psdWindow, hz);
        spectra = reshape(spectra, len(freqs));
        spectra = spectra / max(spectra);
        
        spectras = [spectras; spectra];
        fftfreqs = [fftfreqs; freqs];
    end

end

