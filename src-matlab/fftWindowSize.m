function [ window ] = fftWindowSize(freqRes, hz)
% Calculate power of 2 window length for FFT to achieve specified frequency
% resolution. Useful for power spectra and coherence calculations.
% 
% Input
% -----
% * freqRes : float
% 
%     Desired frequency resolution in hertz
% 
% * hz : int
% 
%     Sample rate, in hertz, of signal undergoing FFT
% 
% Output
% ------
% * window : int

    freqRes = float(freqRes);
    pwr = 1;
    res = hz / float(pow(2,pwr));
    
    while res > freqRes
        pwr = pwr + 1;
        res = hz / float(pow(2,pwr));
    end
    
    window = pow(2,pwr);

end

