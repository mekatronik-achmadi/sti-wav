[refAudio, refRate] = readwav('../speech/eval1.wav');
[degrAudio, degrRate] = readwav('../speech/eval1_echo100.wav');

stis = stiFromAudio(refAudio, degrAudio, refRate, name='eval1.wav');

fprintf('Test Result: \n');

if abs(stis - 0.63) < 0.002:
    fprintf('OK\n');
    return 0
else:
    print('FAILED\n');
    return 1
end


