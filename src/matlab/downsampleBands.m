function [ dsAudio,hz ] = downsampleBands(audio, hz_in, downsampleFactor)
% Downsample audio by integer factor
% 
% Input
% -----
% * audio : array-like
% 
% Array of original audio samples
% 
% * hz : float or int
% 
% Original audio sample rate in hertz
% 
% * downsampleFactor : int
% 
% Factor to downsample audio by, if desired
% 
% Output
% ------
% * dsAudio : ndarray
% 
% Downsampled audio array
% 
% * hz : int
% 
% Downsampled audio sample rate in hertz

    downsampleFactor = int(downsampleFactor);
    hz = int(hz_in / downsampleFactor);
    
    for band = audio 
        ds = decimate(band,downsampleFactor,'fir');
        dsAudio = [dsAudio ds];
    end

end

