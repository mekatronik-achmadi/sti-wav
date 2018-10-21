function [ coherences, fftfreqs ] = octaveBandCoherence(degrAudioBands, refAudioBands, hz, fftRes)
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
    
    fprintf('Calculating degraded and reference audio coherence\n');
    fprintf('FFT length: %d samples',psdWindow);
    
    for band=degrAudioBands
        refband = refAudioBands(find(degrAudioBands==band));
        [coherence, freqs] = my_cohere(band, refband , psdWindow, hz);
        coherences = [coherences, coherence];
        fftfreqs = [fftfreqs, freqs];
    end
end

