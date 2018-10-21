function [ output_args ] = octaveBandFilter( audio, hz, octaveBands,butterOrd, hammingTime)

% Octave band filter raw audio. The audio is filtered through butterworth
% filters of order 6 (by default), squared to obtain the envelope and finally
% low-pass filtered using a 'hammingTime' length Hamming filter at 25 Hz.
% 
% Input
% -----
% * audio : array-like
% 
%     Array of raw audio samples
% 
% * hz : float or int
% 
%     Audio sample rate in hertz
% 
% * octaveBands : array-like
% 
%     list or array of octave band center frequencies
% 
% * butterOrd : int
% 
%     butterworth filter order
% 
% * hammingTime : float or int
% 
%     Hamming window length, in milliseconds relative to audio sample rate
% 
% Output
% ------
% * octaveBandAudio : ndarray
% 
%     Octave band filtered audio
% 
% * hz : float or int
% 
%     Filtered audio sample rate

    if isempty(octaveBands)
        octaveBands=[125 250 500 1000 2000 4000 8000];
    end

    if isempty(butterOrd)
        butterOrd=6;
    end

    if isempty(hammingTime)
        hammingTime=16.6;
    end

    fprintf('Butterworth filter order: %d \n',butterOrd);
    fprintf('Hamming filter length:   %d ms \n',hammingTime);
    fprintf('print(("Audio sample rate:       %d \n',hz);
    
    nyquist = hz * 0.5;
    hammingLength = (hammingTime / 1000.0) * hz;
    
    bands = '';
    for f=octaveBands
        bands=[bands,' ',int2str(octaveBands(find(octaveBands==f)))];
        fprintf('bands at %s \n',bands);
        
        f1 = f / sqrt(2);
        f2 = f * sqrt(2);
        
        if f < max(octaveBands)
            [b1,a1] = butter(butterOrd, f1/nyquist, 'high');
            [b2,a2] = butter(butterOrd, f2/nyquist, 'low');
            
            filtOut = filter(b1, a1, audio);
            filtOut = filter(b2, a2, filtOut);
        else
            [b1,a1] = butter(butterOrd, f/nyquist, 'high');
            filtOut = filter(b1, a1, audio);
        end
        
        filtOut = pow(filtOut,2);
        
        % b = firwin(int(hammingLength), 25.0, window='hamming', nyq=int(nyquist))
        filtOut = filter(b, 1, filtOut);
        filtOut = filtOut * -1.0;
    end

end

