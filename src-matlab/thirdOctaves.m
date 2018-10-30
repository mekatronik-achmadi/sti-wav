function [ freqs ] = thirdOctaves(minFreq, maxFreq)
% Calculates a list of frequencies spaced 1/3 octave apart in hertz
% between minFreq and maxFreq
% 
% Input
% -----
% * minFreq : float or int
% 
%     Must be non-zero and non-negative
% 
% * maxFreq : float or int
% 
%     Must be non-zero and non-negative
% 
% Output
% ------
% * freqs : ndarray

    if (minFreq <=0) || (maxFreq <= 0)
        msg = 'minFreq and maxFreq must be non-zero and non-negative';
        error(msg)
    else
        maxFreq = float(maxFreq);
        f = float(minFreq);
        freqs = f;
        
        while f < maxFreq
            f = f * pow(10,0.1);
            % this array change size every loop. Bad choice !!!
            freqs = [freqs f];
        end
    end
    
end
