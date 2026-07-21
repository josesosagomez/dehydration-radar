% This script lets you run an FMCW Radar continuously for a certain number
% of steps.
% 
% % Setup:
%
% Connect the Vivaldi antenna to Phaser SMA Out2. Place the Vivaldi antenna
% in the field of view of the Phaser and point it at the Phaser.
%
% Notes:
%
% Run this script to continuously run the FMCW radar for demonstration.
% The first time this script is run, the data collection may not occur
% properly.
%
% Copyright 2023 The MathWorks, Inc.

% Carrier frequency                 = 10 GHz
% Light speed                       = 299 792 458 m/s
% Lambda                            = 3 cm
% Maximum range                     = 5 m
% Range resolution                  = 0.3 m
% Maximum velocity                  = 5 m/s
% Speed resolution                  = 0.5 m/s
% Ramp Bandwidth                    = 500 MHz
% Maximum Doppler Frequency shift   = 333.5641 Hz
% PRF (Pulse Repetition Frequency)  = 667.1282 Hz
% PRI (Pulse Repetition Interval)   = 1.5 ms
% Number of Pulses per Frame        = 20
% Pulse Duration                    = 2 ms
% Sweep Duration                    = 1 ms
% Sweep Slope                       = 488 MHz/ms
% Maximum Beat Frequency            = 16.287 KHz
% Sampling Frequency                = 520.834 KHz
% Number of frames                  = 100 frames
% Number of samples per pulse       = 534
% Time frame                        = 40 ms
    
%% Clear workspace and load calibration weights

clear; close all; clc;

%% First, setup the system, see fmcwDemo.m for more details

% Carrier frequency
fc = 10e9;
c = physconst("LightSpeed");
lambda = physconst("LightSpeed")/fc;

% Put some requirements on the system
maxRange = 5;
rangeResolution = 0.3;
maxSpeed = 5;
speedResolution = 1/2;

% Determine some parameter values
rampbandwidth = ceil(rangeres2bw(rangeResolution)/1e6)*1e6;
fmaxdop = speed2dop(2*maxSpeed,lambda);
prf = 2*fmaxdop;
pri = 1/prf;
nPulses = ceil(2*maxSpeed/speedResolution);
tpulse = ceil((1/prf)*1e3)*1e-3;
tsweep = getFMCWSweepTime(tpulse,tpulse);
sweepslope = rampbandwidth / tsweep;
fmaxbeat = sweepslope * range2time(maxRange);
fs = max(ceil(2*fmaxbeat),520834);

%%

% See fmcw demo for these setup steps
[rx,tx,bf,bf_TDD,model] = setupFMCWRadar(fc,fs,tpulse,tsweep,nPulses,rampbandwidth);

% Clear cache
rx();

% Use constant amplitude baseband transmit data
amp = 0.9 * 2^15;
txWaveform = amp*ones(rx.SamplesPerFrame,2);

%% Next, run continuously for nCaptures

nCaptures = 110;

framesRadar = zeros([534,20,nCaptures-10]);
framesRadarIQ = zeros([20834,2,nCaptures-10]);

% Create a range doppler plot
% rd = phased.RangeDopplerResponse(DopplerOutput="Speed",...
%     OperatingFrequency=fc,SampleRate=fs,RangeMethod="FFT",...
%    SweepSlope=sweepslope,PRFSource="Property",PRF=prf);

%ax = axes(figure);

fb = linspace(-0.5,0.5,534)*fs;
fR = c*fb*tpulse/(2*rampbandwidth);

for i = 1:nCaptures
    % capture data
    data = captureTransmitWaveform(txWaveform,rx,tx,bf);
   
    %Plot the data
    % rd.plotResponse(data);
    % xlim(ax,[-maxSpeed,maxSpeed]); ylim(ax,[0,maxRange]);
    % drawnow;
    if i > 10
        framesRadarIQ(:,:,i-10) = data;
        % Arrange data into pulses
        data = arrangePulseData(data,rx,bf,bf_TDD);
        framesRadar(:,:,i-10) = data;
        x = data;
        Xf = fftshift(fft(x-mean(x),534))*fs;
        hold on;
        plot(fR,abs(Xf));
        xlim([0, 5]);
        drawnow;
    end


    fprintf('Frame: %d \n', i);
end

% Disable TDD Trigger so we can operate 
% in Receive only mode
disableTddTrigger(bf_TDD)

%%

save("subject19_4pm_ov_1m.mat","framesRadar", "framesRadarIQ");

%save("subject17_5_9_10am_ov_05m.mat","framesRadar", "framesRadarIQ");


