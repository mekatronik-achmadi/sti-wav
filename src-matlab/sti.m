function [ index ] = sti( modulations, coherences, minCoherence )
% Calculate the speech transmission index from third octave modulation
% indices. The indices are truncated after coherence between clean and dirty
% audio falls below 'minCoherence' or 0.8, by default.
% 
% Input
% -----
% * modulations : array-like
% 
%     Modulation indices spaced at 1/3 octaves within each octave band
% 
% * coherences : array-like
% 
%     Coherence between clean and dirty octave band filtered audio
% 
% * minCoherence : float
% 
%     The minimum coherence to include a mod index in the STI computation
% 
% Output
% ------
% * index : float
% 
%     The speech transmission index (STI)

    if isempty(minCoherence)
        minCoherence=0.8;
    end

    snrMask = zeros(size(modulations));
    for band=coherences
        i=find(coherences==band);
        nz = nonzero(band < minCoherence);
        lessThanMin = nz(0);
        
        if lessThanMin >= 1
            discardAfter = min(lessThanMin);
            snrMask(i,discardAfter:end) = ones(len(snrMask(i,discardAfter:end)));
        end
    end
    
    modulations = my_clip(modulations, 0, 0.99);
    snr = 10*log10(modulations/(1 - modulations));
    snr = my_clip(snr, -15, 15);
    snr = my_masked_array(snr, snrMask); 
    snrCounts = sum(snr/snr, 1);
    octaveBandSNR = sum(snr,1) / snrCounts;
    alpha = 7 * (snrCounts / sum(snrCounts));
    
    w = [0.129 0.143 0.114 0.114 0.186 0.171 0.143];
    
    snrp = alpha * w * octaveBandSNR;
    snrp = sum(snrp);
    index = (snrp + 15) / 30.0;
    
    fprintf('Speech Transmission Index (STI): %d',index);
end

