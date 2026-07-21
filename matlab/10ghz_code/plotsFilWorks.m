%% Visual checks 1–4 for Butterworth filtering
% Assumes variables exist in workspace:
% data    : original matrix (534 x 20 x 100), complex double
% filData : filtered matrix (534 x 20 x 100), complex double

close all; clear; clc;

subject = 14;
time = "8am";
typeFil = "FIR";
frame_idx = 50;
chirp_idx = 10;

data = load("samples\subject_" + subject + "\subject" + subject + "_" + time + "_ov_1m.mat").framesRadar;
filData = load("samples_fil\filteredGPT" + typeFil + "\subject_" + subject + "_" + time + "_fil_gpt_"+ typeFil +".mat").filtered_data;

%% Radar / sampling parameters (edit if different)
fs       = 520834;        % sampling rate (Hz)
B        = 500e6;         % FMCW bandwidth (Hz)
T_chirp  = 1024e-6;       % chirp duration (s)
c        = physconst('LightSpeed');
Nsamples = size(data,1);
Nchirps  = size(data,2);
Nframes  = size(data,3);

% Range-of-interest markers
r_min = 0.9;      % m
r_max = 3.0;      % m

% Slope and freq->range mapping
S = B / T_chirp;                          % chirp slope (Hz/s)
f = (0:Nsamples-1).' * (fs/Nsamples);     % frequency bins (0..fs*(1-1/N))
R = (c * f) / (2*S);                      % meters per bin (one-sided)

% We will use one-sided spectra (up to Nyquist)
halfIdx = 1:floor(Nsamples/2);

%% 1) Range FFT (fast-time) averaged over chirps & frames
% Window to reduce leakage
w = hann(Nsamples,'periodic');

% Compute average magnitude spectrum BEFORE
spec_sum_orig = zeros(Nsamples,1);
for fr = 1:Nframes
    X = data(:,:,fr);                     % (Nsamples x Nchirps)
    Xw = X .* w;                          % window along fast-time
    F = fft(Xw, [], 1);                   % FFT along fast-time
    spec_sum_orig = spec_sum_orig + mean(abs(F), 2);  % average over chirps then accumulate frames
end
spec_avg_orig = spec_sum_orig / Nframes;

% Compute average magnitude spectrum AFTER
spec_sum_filt = zeros(Nsamples,1);
for fr = 1:Nframes
    X = filData(:,:,fr);
    Xw = X .* w;
    F = fft(Xw, [], 1);
    spec_sum_filt = spec_sum_filt + mean(abs(F), 2);
end
spec_avg_filt = spec_sum_filt / Nframes;

% Convert to one-sided and dB for visualization
R_plot = R(halfIdx);
P_orig = 20*log10( spec_avg_orig(halfIdx) / max(spec_avg_orig(halfIdx)) + eps );
P_filt = 20*log10( spec_avg_filt(halfIdx) / max(spec_avg_filt(halfIdx)) + eps );

figure('Name','(1) Range FFT Before vs After','Color','w');
plot(R_plot, P_orig, 'LineWidth',1.5); hold on;
plot(R_plot, P_filt, 'LineWidth',1.5);
xline(r_min,'--k','0.9 m'); xline(r_max,'--k','3.0 m');
grid on; xlabel('Range (m)'); ylabel('Normalized Magnitude (dB)');
title('Average Range Spectrum (Fast-Time FFT) — Before vs After');
legend('Original','Filtered','Location','best');

%% Select a chirp/frame for (2)-(4)
sig_orig = data(:, chirp_idx, frame_idx);
sig_filt = filData(:, chirp_idx, frame_idx);

%% 2) Time-domain signal inspection (magnitude)
t = (0:Nsamples-1).'/fs;   % seconds within chirp

figure('Name','(2) Time-Domain Magnitude (One Chirp)','Color','w');
plot(t*1e3, abs(sig_orig), 'LineWidth',1.2); hold on;
plot(t*1e3, abs(sig_filt), 'LineWidth',1.2);
grid on; xlabel('Fast-time within chirp (ms)'); ylabel('|s(t)|');
title(sprintf('Time-Domain Magnitude — Frame %d, Chirp %d', frame_idx, chirp_idx));
legend('Original','Filtered','Location','best');

%% 3) Spectrograms (same chirp/frame)
% Parameters for spectrogram (adjust if desired)
winLen   = 128;
overlap  = round(0.75*winLen);
nfft_sp  = 512;

figure('Name','(3) Spectrograms Before/After','Color','w');

subplot(2,1,1);
spectrogram(sig_orig, hamming(winLen), overlap, nfft_sp, fs, 'yaxis');
title(sprintf('Spectrogram - Original (Frame %d, Chirp %d)', frame_idx, chirp_idx));
ylim([0 20]); % show up to ~20 kHz to cover the 0.9–3 m band
colormap turbo; colorbar;

subplot(2,1,2);
spectrogram(sig_filt, hamming(winLen), overlap, nfft_sp, fs, 'yaxis');
title('Spectrogram - Filtered');
ylim([0 20]); % same scale
colormap turbo; colorbar;

%% 4) Overlay (normalized) time-domain waveforms
% Normalize to compare shapes clearly
sig_o_n = abs(sig_orig) / (max(abs(sig_orig)) + eps);
sig_f_n = abs(sig_filt) / (max(abs(sig_filt)) + eps);

figure('Name','(4) Overlay (Normalized) — Time-Domain Magnitude','Color','w');
plot(t*1e3, sig_o_n, 'LineWidth',1.3); hold on;
plot(t*1e3, sig_f_n, 'LineWidth',1.3);
grid on; xlabel('Fast-time within chirp (ms)'); ylabel('Normalized |s(t)|');
title(sprintf('Overlay (Normalized) — Frame %d, Chirp %d', frame_idx, chirp_idx));
legend('Original','Filtered','Location','best');