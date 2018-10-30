function [ audio,rate ] = readwav( path )
% Reads Microsoft WAV format audio files, scales integer sample values and
% to [0,1]. Returns a tuple consisting of scaled WAV samples and sample rate
% in hertz.
% 
% Input
% -----
% * path : string
% 
%     Valid system path to file
% 
% Output
% ------
% * audio : array-like
% 
%     Array of scaled sampled
% 
% * rate : int
% 
%     Audio sample rate in hertz

[audio,rate] = audioread(path);

end

