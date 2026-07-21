%% FMCW RADAR FILTER QUALITY ASSESSMENT SCRIPT
% Load your data files and run this script to evaluate filter performance
% 
% Required variables in workspace:
%   - data           : Original radar data (Nfast x Nchirps x Nframes)
%   - filtered_data  : Filtered radar data (same size)

%close all; 
clear; clc;

%% Load Data (modify paths as needed)
fprintf('Loading data...\n');

% Uncomment and modify these lines to load your data:
subject = 6;
time = "8am";
typeFil = "FIR"; %BW FIR FFT

data = load("samples\subject_" + subject + "\subject" + subject + "_" + time + "_ov_1m.mat").framesRadar;
filtered_data = load("samples_fil\filteredGPT" + typeFil + "\subject_" + subject + "_" + time + "_fil_gpt_"+ typeFil +".mat").filtered_data;


% Check if data exists in workspace
if ~exist('data', 'var') || ~exist('filtered_data', 'var')
    error('Please load ''data'' and ''filtered_data'' into workspace first!');
end

fprintf('Data loaded successfully.\n');
fprintf('  Original data size: %s\n', mat2str(size(data)));
fprintf('  Filtered data size: %s\n', mat2str(size(filtered_data)));

%% Radar Parameters
fs = 520834;               % Sampling frequency (Hz)
B = 500e6;                 % FMCW bandwidth (Hz)
T_chirp = 1024e-6;         % Chirp duration (s)
r_min = 0.9;               % Min range of interest (m)
r_max = 3.0;               % Max range of interest (m)
c = physconst('LightSpeed');

% Data dimensions
[Nfast, Nchirps, Nframes] = size(data);

% Range axis
delta_r = c / (2 * B);  % Range resolution
range_axis = (0:Nfast-1)' * delta_r;

% Find range bins corresponding to ROI
bin_min = max(1, round(r_min / delta_r));
bin_max = min(Nfast, round(r_max / delta_r));
roi_mask = false(Nfast, 1);
roi_mask(bin_min:bin_max) = true;

fprintf('\nRadar Configuration:\n');
fprintf('  Sampling freq: %.2f kHz\n', fs/1e3);
fprintf('  Bandwidth: %.0f MHz\n', B/1e6);
fprintf('  Range resolution: %.4f m\n', delta_r);
fprintf('  ROI: %.2f - %.2f m (bins %d to %d)\n', r_min, r_max, bin_min, bin_max);

%% Metric 1: Signal Preservation in ROI
fprintf('\n========== FILTER QUALITY ASSESSMENT ==========\n\n');
fprintf('1. SIGNAL PRESERVATION IN RANGE OF INTEREST (%.2f - %.2f m)\n', r_min, r_max);
fprintf('   ROI bins: %d to %d (out of %d)\n', bin_min, bin_max, Nfast);

% Compute range profiles (average over chirps and frames)
range_profile_orig = zeros(Nfast, Nframes);
range_profile_filt = zeros(Nfast, Nframes);

fprintf('   Computing range profiles...\n');
for fr = 1:Nframes
    for ch = 1:Nchirps
        range_profile_orig(:, fr) = range_profile_orig(:, fr) + ...
            abs(fft(data(:, ch, fr)));
        range_profile_filt(:, fr) = range_profile_filt(:, fr) + ...
            abs(fft(filtered_data(:, ch, fr)));
    end
end
range_profile_orig = range_profile_orig / Nchirps;
range_profile_filt = range_profile_filt / Nchirps;

% Average power in ROI vs outside
power_roi_orig = mean(range_profile_orig(roi_mask, :), 'all');
power_outside_orig = mean(range_profile_orig(~roi_mask, :), 'all');
power_roi_filt = mean(range_profile_filt(roi_mask, :), 'all');
power_outside_filt = mean(range_profile_filt(~roi_mask, :), 'all');

preservation_ratio = power_roi_filt / power_roi_orig;
suppression_ratio = power_outside_filt / power_outside_orig;

fprintf('   - Signal preservation in ROI: %.2f%% (want ~100%%)\n', preservation_ratio*100);
fprintf('   - Signal suppression outside ROI: %.2f%% (want ~0%%)\n', suppression_ratio*100);

%% Metric 2: SNR Analysis
fprintf('\n2. SIGNAL-TO-NOISE RATIO ANALYSIS\n');

% Estimate noise floor (assume lowest 10% of range bins are noise)
[~, sorted_idx] = sort(mean(range_profile_orig, 2));
noise_bins_orig = sorted_idx(1:round(0.1*Nfast));
noise_floor_orig = mean(range_profile_orig(noise_bins_orig, :), 'all');

[~, sorted_idx_filt] = sort(mean(range_profile_filt, 2));
noise_bins_filt = sorted_idx_filt(1:round(0.1*Nfast));
noise_floor_filt = mean(range_profile_filt(noise_bins_filt, :), 'all');

% Peak signal in ROI
signal_peak_orig = max(range_profile_orig(roi_mask, :), [], 'all');
signal_peak_filt = max(range_profile_filt(roi_mask, :), [], 'all');

snr_orig_db = 10*log10(signal_peak_orig / noise_floor_orig);
snr_filt_db = 10*log10(signal_peak_filt / noise_floor_filt);
snr_improvement_db = snr_filt_db - snr_orig_db;

fprintf('   - Original SNR: %.2f dB\n', snr_orig_db);
fprintf('   - Filtered SNR: %.2f dB\n', snr_filt_db);
fprintf('   - SNR improvement: %.2f dB (positive is good)\n', snr_improvement_db);

%% Metric 3: Spectral Leakage (Stopband Attenuation)
fprintf('\n3. STOPBAND ATTENUATION\n');

% Define stopband regions (outside ROI with some guard bands)
guard_bins = 10;
stopband_low = 1:(max(1, bin_min - guard_bins));
stopband_high = (min(Nfast, bin_max + guard_bins)):Nfast;
stopband_mask = false(Nfast, 1);
if ~isempty(stopband_low)
    stopband_mask(stopband_low) = true;
end
if ~isempty(stopband_high)
    stopband_mask(stopband_high) = true;
end

passband_power = mean(range_profile_orig(roi_mask, :), 'all');
stopband_power_orig = mean(range_profile_orig(stopband_mask, :), 'all');
stopband_power_filt = mean(range_profile_filt(stopband_mask, :), 'all');

attenuation_db = 10*log10(stopband_power_filt / passband_power);

fprintf('   - Stopband attenuation: %.2f dB (more negative is better)\n', attenuation_db);
fprintf('   - Stopband rejection: %.2f%% (lower is better)\n', ...
    (stopband_power_filt/passband_power)*100);

%% Metric 4: Signal Distortion (Correlation & MSE in ROI)
fprintf('\n4. SIGNAL DISTORTION IN ROI\n');

% Extract ROI signals for all frames
roi_orig = range_profile_orig(roi_mask, :);
roi_filt = range_profile_filt(roi_mask, :);

% Normalize for fair comparison
roi_orig_norm = roi_orig / max(roi_orig(:));
roi_filt_norm = roi_filt / max(roi_filt(:));

% Correlation
correlation = corr(roi_orig_norm(:), roi_filt_norm(:));

% Normalized MSE
mse_normalized = mean((roi_orig_norm(:) - roi_filt_norm(:)).^2);
psnr_db = 10*log10(1 / mse_normalized);

fprintf('   - Correlation with original: %.4f (want ~1.0)\n', correlation);
fprintf('   - Normalized MSE: %.6f (want ~0.0)\n', mse_normalized);
fprintf('   - PSNR: %.2f dB (higher is better, >40 dB excellent)\n', psnr_db);

%% Metric 5: Time-Domain Characteristics
fprintf('\n5. TIME-DOMAIN ANALYSIS\n');

% Select middle frame and chirp for analysis
mid_frame = round(Nframes/2);
mid_chirp = round(Nchirps/2);

sig_orig = data(:, mid_chirp, mid_frame);
sig_filt = filtered_data(:, mid_chirp, mid_frame);

% Energy preservation
energy_orig = sum(abs(sig_orig).^2);
energy_filt = sum(abs(sig_filt).^2);
energy_preservation = energy_filt / energy_orig;

% Peak preservation
peak_orig = max(abs(sig_orig));
peak_filt = max(abs(sig_filt));
peak_preservation = peak_filt / peak_orig;

fprintf('   - Energy preservation: %.2f%% (want ~100%%)\n', energy_preservation*100);
fprintf('   - Peak amplitude preservation: %.2f%%\n', peak_preservation*100);

%% Metric 6: Phase Distortion
fprintf('\n6. PHASE ANALYSIS\n');

phase_orig = angle(sig_orig);
phase_filt = angle(sig_filt);
phase_diff = wrapToPi(phase_filt - phase_orig);

% Phase error in ROI (convert to range domain first)
range_sig_orig = fft(sig_orig);
range_sig_filt = fft(sig_filt);
phase_diff_range = wrapToPi(angle(range_sig_filt(roi_mask)) - angle(range_sig_orig(roi_mask)));

mean_phase_error = mean(abs(phase_diff_range));
max_phase_error = max(abs(phase_diff_range));

fprintf('   - Mean phase error in ROI: %.4f rad (%.2f deg)\n', ...
    mean_phase_error, rad2deg(mean_phase_error));
fprintf('   - Max phase error in ROI: %.4f rad (%.2f deg)\n', ...
    max_phase_error, rad2deg(max_phase_error));

%% Overall Quality Score
fprintf('\n========== OVERALL QUALITY SCORE ==========\n');

% Weighted scoring (0-100 scale)
score_preservation = max(0, min(100, preservation_ratio * 100));
score_suppression = max(0, min(100, 100 - suppression_ratio * 100));
score_snr = max(0, min(100, 50 + snr_improvement_db * 5));  % 0dB improvement = 50pts
score_correlation = max(0, min(100, correlation * 100));
score_phase = max(0, min(100, 100 - rad2deg(mean_phase_error) * 10));

overall_score = 0.25*score_preservation + 0.25*score_suppression + ...
                0.20*score_snr + 0.20*score_correlation + 0.10*score_phase;

fprintf('Quality Scores (0-100 scale):\n');
fprintf('  - Signal preservation:    %.1f/100\n', score_preservation);
fprintf('  - Stopband suppression:   %.1f/100\n', score_suppression);
fprintf('  - SNR improvement:        %.1f/100\n', score_snr);
fprintf('  - Signal correlation:     %.1f/100\n', score_correlation);
fprintf('  - Phase preservation:     %.1f/100\n', score_phase);
fprintf('\n  >>> OVERALL QUALITY: %.1f/100 <<<\n', overall_score);

if overall_score >= 90
    fprintf('      Rating: EXCELLENT - Filter is highly effective\n');
elseif overall_score >= 75
    fprintf('      Rating: GOOD - Filter performs well\n');
elseif overall_score >= 60
    fprintf('      Rating: ACCEPTABLE - Filter works but has issues\n');
else
    fprintf('      Rating: POOR - Filter needs improvement\n');
end

fprintf('\n===============================================\n\n');

%% Save Results to Structure
results = struct();
results.preservation_ratio = preservation_ratio;
results.suppression_ratio = suppression_ratio;
results.snr_original_db = snr_orig_db;
results.snr_filtered_db = snr_filt_db;
results.snr_improvement_db = snr_improvement_db;
results.stopband_attenuation_db = attenuation_db;
results.stopband_rejection_pct = (stopband_power_filt/passband_power)*100;
results.correlation = correlation;
results.mse_normalized = mse_normalized;
results.psnr_db = psnr_db;
results.energy_preservation = energy_preservation;
results.peak_preservation = peak_preservation;
results.mean_phase_error_rad = mean_phase_error;
results.max_phase_error_rad = max_phase_error;
results.overall_score = overall_score;
results.component_scores = struct('preservation', score_preservation, ...
                                  'suppression', score_suppression, ...
                                  'snr', score_snr, ...
                                  'correlation', score_correlation, ...
                                  'phase', score_phase);

fprintf('Results saved to ''results'' structure in workspace.\n\n');

%% Generate Visualization Plots
fprintf('Generating visualization plots...\n');

figure('Position', [100 100 1400 900], 'Name', 'Filter Quality Analysis');

% Plot 1: Range Profile Comparison
subplot(3,3,1);
avg_orig = mean(range_profile_orig, 2);
avg_filt = mean(range_profile_filt, 2);
plot(range_axis, 20*log10(avg_orig), 'b', 'LineWidth', 1.5); hold on;
plot(range_axis, 20*log10(avg_filt), 'r', 'LineWidth', 1.5);
xline(range_axis(bin_min), 'g--', 'LineWidth', 1);
xline(range_axis(bin_max), 'g--', 'LineWidth', 1);
xlabel('Range (m)'); ylabel('Magnitude (dB)');
title('Average Range Profile');
legend('Original', 'Filtered', 'ROI bounds', 'Location', 'best');
grid on;

% Plot 2: Range Profile (ROI zoom)
subplot(3,3,2);
roi_range = range_axis(roi_mask);
plot(roi_range, 20*log10(avg_orig(roi_mask)), 'b.-', 'LineWidth', 1.5); hold on;
plot(roi_range, 20*log10(avg_filt(roi_mask)), 'r.-', 'LineWidth', 1.5);
xlabel('Range (m)'); ylabel('Magnitude (dB)');
title('Range Profile (ROI Zoomed)');
legend('Original', 'Filtered', 'Location', 'best');
grid on;

% Plot 3: Stopband Analysis
subplot(3,3,3);
stopband_low_idx = 1:max(1, bin_min-10);
stopband_high_idx = min(length(range_axis), bin_max+10):length(range_axis);

semilogy(range_axis(stopband_low_idx), avg_orig(stopband_low_idx), 'b.-'); hold on;
semilogy(range_axis(stopband_low_idx), avg_filt(stopband_low_idx), 'r.-');
if ~isempty(stopband_high_idx)
    semilogy(range_axis(stopband_high_idx), avg_orig(stopband_high_idx), 'b.-');
    semilogy(range_axis(stopband_high_idx), avg_filt(stopband_high_idx), 'r.-');
end
xlabel('Range (m)'); ylabel('Magnitude (linear)');
title('Stopband Suppression');
legend('Original', 'Filtered', 'Location', 'best');
grid on;

% Plot 4: Range-Time Map (Original)
subplot(3,3,4);
imagesc(1:size(range_profile_orig,2), range_axis, 20*log10(range_profile_orig));
xlabel('Frame'); ylabel('Range (m)');
title('Range-Time Map: Original');
colorbar; colormap('jet');
hold on;
yline(range_axis(bin_min), 'g--', 'LineWidth', 1.5);
yline(range_axis(bin_max), 'g--', 'LineWidth', 1.5);

% Plot 5: Range-Time Map (Filtered)
subplot(3,3,5);
imagesc(1:size(range_profile_filt,2), range_axis, 20*log10(range_profile_filt));
xlabel('Frame'); ylabel('Range (m)');
title('Range-Time Map: Filtered');
colorbar; colormap('jet');
hold on;
yline(range_axis(bin_min), 'g--', 'LineWidth', 1.5);
yline(range_axis(bin_max), 'g--', 'LineWidth', 1.5);

% Plot 6: Difference Map
subplot(3,3,6);
diff_map = 20*log10(range_profile_filt) - 20*log10(range_profile_orig);
imagesc(1:size(diff_map,2), range_axis, diff_map);
xlabel('Frame'); ylabel('Range (m)');
title('Difference: Filtered - Original (dB)');
colorbar; colormap('jet');
caxis([-20 20]);  % Symmetric color scale

% Plot 7: Time-domain comparison
subplot(3,3,7);
t_axis = 0:length(sig_orig)-1;
plot(t_axis, abs(sig_orig), 'b', 'LineWidth', 1); hold on;
plot(t_axis, abs(sig_filt), 'r', 'LineWidth', 1);
xlabel('Fast-time (samples)'); ylabel('Magnitude');
title(sprintf('Time-Domain (Frame %d, Chirp %d)', mid_frame, mid_chirp));
legend('Original', 'Filtered', 'Location', 'best');
grid on;

% Plot 8: Quality Metrics Bar Chart
subplot(3,3,8);
scores = [score_preservation, score_suppression, score_snr, ...
          score_correlation, score_phase];
labels = {'Preservation', 'Suppression', 'SNR', 'Correlation', 'Phase'};
bar(scores);
set(gca, 'XTickLabel', labels, 'XTickLabelRotation', 45);
ylabel('Score (0-100)');
title('Quality Component Scores');
ylim([0 100]);
yline(75, 'g--', 'Good', 'LineWidth', 1.5);
grid on;

% Plot 9: Overall Score Gauge
subplot(3,3,9);
theta = linspace(0, pi, 100);
r = ones(size(theta));
polarplot(theta, r, 'k', 'LineWidth', 2); hold on;

score_angle = (overall_score / 100) * pi;
polarplot([0 score_angle], [0 1], 'r', 'LineWidth', 4);

ax = gca;
ax.ThetaZeroLocation = 'bottom';
ax.ThetaDir = 'counterclockwise';
ax.RLim = [0 1];
ax.ThetaLim = [0 180];
title(sprintf('Overall Quality: %.1f/100', overall_score), ...
      'FontSize', 12, 'FontWeight', 'bold');

sgtitle('FMCW Radar Filter Quality Assessment', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('Plots generated successfully.\n');
fprintf('\n=== ANALYSIS COMPLETE ===\n');