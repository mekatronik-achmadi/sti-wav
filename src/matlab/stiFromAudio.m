function [ stiValues ] = stiFromAudio(reference, degraded, hz, calcref, downsample, name)
% Calculate the speech transmission index (STI) from clean and dirty
% (ie: distorted) audio samples. The clean and dirty audio samples must have
% a common sample rate for successful use of this function.
% 
% Input
% -----
% * reference : array-like
% 
%     Clean reference audio sample as an array of floating-point values
% 
% * degraded : array-like
% 
%     Degraded audio sample as an array, or array of arrays for multiple
%     samples, of floating-point values
% 
% * hz : int
% 
%     Audio sample rate in hertz
% 
% * calcref : boolean
% 
%     Calculate STI for reference signal alone
% 
% * downsample : int or None
% 
%     Downsampling integer factor
% 
% * name : string
% 
%     Name of sample set, for output tracking in larger runs
% 
% Output
% ------
% * sti : array-like or float
% 
%     The calculated speech transmission index (STI) value(s)

    if isempty(calcref)
        calcref=false;
    end
    
    if isempty(downsample)
        downsample=null;
    end
    
    if isempty(name)
        name='untitled';
    end
    
    fprintf('--------------------------------------------------------\n');
    fprintf('Speech Transmission Index (STI) from speech waveforms\n');
    fprintf('--------------------------------------------------------\n');
    fprintf('\n');
    print ('Sample set:             %s \n', name);
    print ('Number of samples:      %d \n', len(degraded));
    fprintf('Calculate reference STI:\n');
    
    if calcref
        fprintf('yes\n');
    else
        fprintf('no\n');
    end
    fprintf('\n');
    fprintf('Reference Speech\n');
    
    refOctaveBands = octaveBandFilter(reference, hz);
    refRate = hz;
    
    [refOctaveBands, refRate] = downsampleBands(refOctaveBands, refRate, downsample);
    
    if calcref
        [spectras, sfreqs] = octaveBandSpectra(refOctaveBands, refRate);
        [coherences, cfreqs] = octaveBandCoherence(refOctaveBands, refOctaveBands,refRate);
        thirdOctaveMTF = thirdOctaveRootSum(spectras, sfreqs);
        thirdOctaveCoherences = thirdOctaveRMS(coherences, cfreqs);
        
        thirdOctaveTemps =[ thirdOctaveTemps thirdOctaveMTF thirdOctaveCoherences];
    end
    
    for sample=degraded
        j = find(degraded==sample);
        fprintf('Degraded Speech: Sample {%d}\n',j);
        degrOctaveBands = octaveBandFilter(sample, hz);
        degrRate = hz;
        
        [degrOctaveBands, degrRate] = downsampleBands(degrOctaveBands, degrRate, downsample);
        [spectras, sfreqs] = octaveBandSpectra(degrOctaveBands, degrRate);
        [coherences, cfreqs] = octaveBandCoherence(refOctaveBands, degrOctaveBands, refRate);
        thirdOctaveMTF = thirdOctaveRootSum(spectras, sfreqs);
        thirdOctaveCoherences = thirdOctaveRMS(coherences, cfreqs);
        
        thirdOctaveTemps = [ thirdOctaveTemps thirdOctaveMTF thirdOctaveCoherences];
    end
    
    fprintf(' Speech Transmission Index \n');
    for i = 1:len(thirdOctaveTemps)
        sampleSTI = sti(thirdOctaveTemps(i,1), thirdOctaveTemps(i,2));
        stiValues = [stiValues sampleSTI];
    end
    
    if len(stiValues) == 1
        stiValues = stiValues(1);
    end

end

