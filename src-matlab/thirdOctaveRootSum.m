function [ thirdOctaveSums ] = thirdOctaveRootSum(spectras, fftfreqs, minFreq, maxFreq)
% Calculates square root of sum of spectra over 1/3 octave bands
% 
% Input
% -----
% * spectras : array-like
% 
%     Array or list of octave band spectras
% 
% * fftfreqs : array-like
% 
%     Array or list of octave band FFT frequencies
% 
% * minFreq : float
% 
%     Min frequency in 1/3 octave bands
% 
% * maxFreq : float
% 
%     Max frequency in 1/3 octave bands
% 
% Output
% ------
% * thirdOctaveRootSums : ndarray
% 
%     Square root of spectra sums over 1/3 octave intervals
    if isempty(minFreq)
        minFreq=0.25;
    end
    
    if isempty(maxFreq)
        minFreq=25.0;
    end

    fprintf('Calculating 1/3 octave square-rooted sums from \n');
    fprintf('%d to %d Hz',minFreq,maxFreq);
    
    thirdOctaveBands = thirdOctaves(minFreq, maxFreq);
        
    for spectra = spectras
        freqs = fftfreqs(find(spectras==spectra));
        
        sums = 0;
        for f13=thirdOctaveBands
            f131 = f13 / pow(2, 1.0/6.0); 
            f132 = f13 * pow(2, 1.0/6.0);
            
            li = my_searchsorted(freqs, f131);
            ui = my_searchsorted(freqs, f132) + 1;
            
            s = sum(spectra(li:ui));
            s = sqrt(s);
            
            sums = [sums s];
            
        end
        
        thirdOctaveSums = [thirdOctaveSums; sums];
    end
        
end

