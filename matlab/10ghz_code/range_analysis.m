% ========================================================================
% Batch Range Analysis - All Subjects and Times
% Purpose: Process all radar files and generate comprehensive report
% ========================================================================
close all;
clear;
clc;

% Radar parameters
fs = 520834; % Sampling frequency (Hz)
B = 500e6; % Bandwidth (Hz)
T_chirp = 1024e-6; % Chirp duration (s)
c = 3e8; % Speed of light (m/s)

% Range limits and bin size
R_min = 0.9; % meters
R_max = 3.0; % meters
bin_size = 0.3; % meters (range resolution)

% Define range bins
range_bin_edges = R_min:bin_size:R_max;
range_bin_centers = range_bin_edges(1:end-1) + bin_size/2;
num_bins = length(range_bin_centers);

% Subject and time configuration
subjects = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
times = {'8am', '10am', '12pm', '2pm', '4pm'};
base_folder = 'samples';

% Expected subject position
expected_range = 1.8; % meters
tolerance = 0.3; % meters

% Open report file
report_filename = 'range_analysis_report.txt';
fid = fopen(report_filename, 'w');

% Write header
fprintf(fid, '================================================================================\n');
fprintf(fid, '                    RADAR RANGE ANALYSIS - COMPREHENSIVE REPORT\n');
fprintf(fid, '================================================================================\n');
fprintf(fid, 'Generated: %s\n', datestr(now));
fprintf(fid, 'Range window: %.1f - %.1f m\n', R_min, R_max);
fprintf(fid, 'Bin size: %.1f m\n', bin_size);
fprintf(fid, 'Bin centers: ');
fprintf(fid, '%.1f ', range_bin_centers);
fprintf(fid, 'm\n');
fprintf(fid, 'Expected subject position: %.1f m (±%.1f m)\n', expected_range, tolerance);
fprintf(fid, '================================================================================\n\n');

% Also print to console
fprintf('================================================================================\n');
fprintf('Processing all files...\n');
fprintf('Subjects: %s\n', mat2str(subjects));
fprintf('Times: %s\n', strjoin(times, ', '));
fprintf('================================================================================\n\n');

% Statistics accumulator for summary
all_files_stats = struct();
file_count = 0;
total_files = length(subjects) * length(times);

% Process each subject and time
for subj_idx = 1:length(subjects)
    subject_num = subjects(subj_idx);
    
    for time_idx = 1:length(times)
        time_str = times{time_idx};
        filename = sprintf('subject%d_%s_ov_1m.mat', subject_num, time_str);
        filepath = fullfile(base_folder, filename);
        
        file_count = file_count + 1;
        
        % Check if file exists
        if ~exist(filepath, 'file')
            fprintf('  [%d/%d] WARNING: File not found: %s\n', file_count, total_files, filepath);
            fprintf(fid, '\n--------------------------------------------------------------------------------\n');
            fprintf(fid, 'File: %s\n', filename);
            fprintf(fid, 'Status: FILE NOT FOUND\n');
            fprintf(fid, '--------------------------------------------------------------------------------\n');
            continue;
        end
        
        fprintf('  [%d/%d] Processing: %s\n', file_count, total_files, filename);
        
        % Load data
        try
            data_struct = load(filepath);
            radar_data = data_struct.framesRadar;
        catch ME
            fprintf('  ERROR loading file: %s\n', ME.message);
            fprintf(fid, '\n--------------------------------------------------------------------------------\n');
            fprintf(fid, 'File: %s\n', filename);
            fprintf(fid, 'Status: ERROR LOADING FILE - %s\n', ME.message);
            fprintf(fid, '--------------------------------------------------------------------------------\n');
            continue;
        end
        
        % Create range axis
        range_axis = ((0:size(radar_data, 1)-1) * c * T_chirp*1000) / (2 * B);
        
        % Initialize counters
        num_chirps = size(radar_data, 2);
        num_frames = size(radar_data, 3);
        total_chirps = num_chirps * num_frames;
        
        bin_counts = zeros(num_bins, 1);
        peak_ranges_all = zeros(total_chirps, 1);
        
        % Process each frame and chirp
        chirp_idx = 1;
        for frame = 1:num_frames
            for chirp = 1:num_chirps
                % Extract signal for this chirp and apply FFT
                chirp_signal = radar_data(:, chirp, frame);
                range_profile = fft(chirp_signal, size(radar_data, 1));
                
                % Find indices in range window
                range_mask = (range_axis >= R_min) & (range_axis <= R_max);
                range_profile_window = range_profile(range_mask);
                range_values_window = range_axis(range_mask);
                
                % Find peak amplitude and its range
                [~, max_idx] = max(abs(range_profile_window));
                peak_range = range_values_window(max_idx);
                peak_ranges_all(chirp_idx) = peak_range;
                
                % Determine which bin this peak belongs to
                bin_idx = discretize(peak_range, range_bin_edges);
                if ~isnan(bin_idx)
                    bin_counts(bin_idx) = bin_counts(bin_idx) + 1;
                end
                
                chirp_idx = chirp_idx + 1;
            end
        end
        
        % Calculate statistics
        mean_range = mean(peak_ranges_all);
        median_range = median(peak_ranges_all);
        std_range = std(peak_ranges_all);
        min_range = min(peak_ranges_all);
        max_range = max(peak_ranges_all);
        mode_range = mode(round(peak_ranges_all/bin_size)*bin_size);
        
        % Find dominant bin
        [max_count, max_bin_idx] = max(bin_counts);
        dominant_range_center = range_bin_centers(max_bin_idx);
        
        % Check concentration around expected position
        chirps_near_expected = sum(abs(peak_ranges_all - expected_range) <= tolerance);
        percentage_near_expected = (chirps_near_expected / total_chirps) * 100;
        
        % Distribution metrics
        entropy = -sum((bin_counts(bin_counts>0)/total_chirps) .* log2(bin_counts(bin_counts>0)/total_chirps));
        max_entropy = log2(num_bins);
        concentration_index = 1 - (entropy / max_entropy);
        
        % Position assessment
        if percentage_near_expected > 80
            position_status = 'GOOD - Consistently at expected position';
        elseif mean_range > expected_range + 0.2
            position_status = sprintf('WARNING - Further than expected (%.2f m)', mean_range);
        elseif mean_range < expected_range - 0.2
            position_status = sprintf('WARNING - Closer than expected (%.2f m)', mean_range);
        else
            position_status = 'ACCEPTABLE - Close to expected position';
        end
        
        % Movement assessment
        if concentration_index > 0.7
            movement_status = 'Stationary (highly concentrated)';
        elseif concentration_index > 0.4
            movement_status = 'Some movement (moderately concentrated)';
        else
            movement_status = 'Significant movement or clutter (dispersed)';
        end
        
        % Store for summary
        all_files_stats(file_count).filename = filename;
        all_files_stats(file_count).subject = subject_num;
        all_files_stats(file_count).time = time_str;
        all_files_stats(file_count).mean_range = mean_range;
        all_files_stats(file_count).median_range = median_range;
        all_files_stats(file_count).std_range = std_range;
        all_files_stats(file_count).dominant_range = dominant_range_center;
        all_files_stats(file_count).concentration = concentration_index;
        all_files_stats(file_count).near_expected_pct = percentage_near_expected;
        
        % Write to report
        fprintf(fid, '\n================================================================================\n');
        fprintf(fid, 'FILE: %s\n', filename);
        fprintf(fid, '================================================================================\n');
        fprintf(fid, 'Subject: %d | Time: %s\n', subject_num, time_str);
        fprintf(fid, 'Data size: %dx%dx%d (samples x chirps x frames)\n', size(radar_data));
        fprintf(fid, 'Total chirps analyzed: %d (%d frames × %d chirps/frame)\n\n', total_chirps, num_frames, num_chirps);
        
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, 'CHIRP COUNT PER RANGE BIN\n');
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, '%-15s | %-10s | %-12s | %-30s\n', 'Range Bin (m)', 'Count', 'Percentage', 'Bar Chart');
        fprintf(fid, '%-15s-+-%-10s-+-%-12s-+-%-30s\n', '---------------', '----------', '------------', '------------------------------');
        
        for i = 1:num_bins
            percentage = (bin_counts(i) / total_chirps) * 100;
            bar_length = round(percentage / 2);
            bar_str = repmat('█', 1, bar_length);
            
            fprintf(fid, '%.1f - %.1f     | %6d     | %7.2f%%     | %s\n', ...
                    range_bin_edges(i), range_bin_edges(i+1), ...
                    bin_counts(i), percentage, bar_str);
        end
        
        fprintf(fid, '%-15s-+-%-10s-+-%-12s-+-%-30s\n', '---------------', '----------', '------------', '------------------------------');
        fprintf(fid, '%-15s | %6d     | %7.2f%%     |\n\n', 'TOTAL', total_chirps, 100.0);
        
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, 'DOMINANT RANGE BIN\n');
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, 'Range: %.1f - %.1f m (center: %.1f m)\n', ...
                range_bin_edges(max_bin_idx), range_bin_edges(max_bin_idx+1), dominant_range_center);
        fprintf(fid, 'Count: %d chirps (%.2f%% of total)\n\n', max_count, (max_count/total_chirps)*100);
        
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, 'RANGE STATISTICS\n');
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, 'Mean:       %8.3f m\n', mean_range);
        fprintf(fid, 'Median:     %8.3f m\n', median_range);
        fprintf(fid, 'Mode:       %8.3f m\n', mode_range);
        fprintf(fid, 'Std Dev:    %8.3f m\n', std_range);
        fprintf(fid, 'Min:        %8.3f m\n', min_range);
        fprintf(fid, 'Max:        %8.3f m\n\n', max_range);
        
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, 'SUBJECT POSITION ANALYSIS\n');
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, 'Expected position: %.1f m (±%.1f m)\n', expected_range, tolerance);
        fprintf(fid, 'Chirps within tolerance: %d / %d (%.2f%%)\n', ...
                chirps_near_expected, total_chirps, percentage_near_expected);
        fprintf(fid, 'Status: %s\n\n', position_status);
        
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, 'DISTRIBUTION METRICS\n');
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        fprintf(fid, 'Entropy:              %.3f bits (max: %.3f bits)\n', entropy, max_entropy);
        fprintf(fid, 'Concentration Index:  %.3f (0=uniform, 1=single bin)\n', concentration_index);
        fprintf(fid, 'Assessment:           %s\n\n', movement_status);
        
    end
end

% ========================================================================
% Summary Section
% ========================================================================

fprintf(fid, '\n\n');
fprintf(fid, '################################################################################\n');
fprintf(fid, '#                                                                              #\n');
fprintf(fid, '#                            SUMMARY STATISTICS                                #\n');
fprintf(fid, '#                                                                              #\n');
fprintf(fid, '################################################################################\n\n');

fprintf(fid, 'Total files processed: %d\n', file_count);
fprintf(fid, 'Subjects: %s\n', mat2str(subjects));
fprintf(fid, 'Times: %s\n\n', strjoin(times, ', '));

% Calculate overall statistics
if file_count > 0
    all_means = [all_files_stats.mean_range];
    all_medians = [all_files_stats.median_range];
    all_stds = [all_files_stats.std_range];
    all_concentrations = [all_files_stats.concentration];
    all_near_expected = [all_files_stats.near_expected_pct];
    
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, 'OVERALL STATISTICS (across all files)\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, 'Mean Range:\n');
    fprintf(fid, '  Average:   %.3f m\n', mean(all_means));
    fprintf(fid, '  Std Dev:   %.3f m\n', std(all_means));
    fprintf(fid, '  Min:       %.3f m\n', min(all_means));
    fprintf(fid, '  Max:       %.3f m\n\n', max(all_means));
    
    fprintf(fid, 'Concentration Index:\n');
    fprintf(fid, '  Average:   %.3f\n', mean(all_concentrations));
    fprintf(fid, '  Std Dev:   %.3f\n', std(all_concentrations));
    fprintf(fid, '  Min:       %.3f\n', min(all_concentrations));
    fprintf(fid, '  Max:       %.3f\n\n', max(all_concentrations));
    
    fprintf(fid, 'Percentage Near Expected Position (%.1f m):\n', expected_range);
    fprintf(fid, '  Average:   %.2f%%\n', mean(all_near_expected));
    fprintf(fid, '  Std Dev:   %.2f%%\n', std(all_near_expected));
    fprintf(fid, '  Min:       %.2f%%\n', min(all_near_expected));
    fprintf(fid, '  Max:       %.2f%%\n\n', max(all_near_expected));
    
    % Per-subject summary
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, 'PER-SUBJECT SUMMARY\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, '%-8s | %-10s | %-10s | %-15s | %-20s\n', ...
            'Subject', 'Mean Range', 'Std Range', 'Concentration', 'Near Expected (avg)');
    fprintf(fid, '%-8s-+-%-10s-+-%-10s-+-%-15s-+-%-20s\n', ...
            '--------', '----------', '----------', '---------------', '--------------------');
    
    for subj = subjects
        subj_mask = [all_files_stats.subject] == subj;
        if any(subj_mask)
            subj_means = [all_files_stats(subj_mask).mean_range];
            subj_stds = [all_files_stats(subj_mask).std_range];
            subj_conc = [all_files_stats(subj_mask).concentration];
            subj_near = [all_files_stats(subj_mask).near_expected_pct];
            
            fprintf(fid, '%4d     | %8.3f m | %8.3f m | %13.3f   | %17.2f%%\n', ...
                    subj, mean(subj_means), mean(subj_stds), mean(subj_conc), mean(subj_near));
        end
    end
    
    % Per-time summary
    fprintf(fid, '\n--------------------------------------------------------------------------------\n');
    fprintf(fid, 'PER-TIME SUMMARY\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, '%-8s | %-10s | %-10s | %-15s | %-20s\n', ...
            'Time', 'Mean Range', 'Std Range', 'Concentration', 'Near Expected (avg)');
    fprintf(fid, '%-8s-+-%-10s-+-%-10s-+-%-15s-+-%-20s\n', ...
            '--------', '----------', '----------', '---------------', '--------------------');
    
    for t = times
        time_str = t{1};
        time_mask = strcmp({all_files_stats.time}, time_str);
        if any(time_mask)
            time_means = [all_files_stats(time_mask).mean_range];
            time_stds = [all_files_stats(time_mask).std_range];
            time_conc = [all_files_stats(time_mask).concentration];
            time_near = [all_files_stats(time_mask).near_expected_pct];
            
            fprintf(fid, '%-8s | %8.3f m | %8.3f m | %13.3f   | %17.2f%%\n', ...
                    time_str, mean(time_means), mean(time_stds), mean(time_conc), mean(time_near));
        end
    end
    
    % Files with best concentration
    fprintf(fid, '\n--------------------------------------------------------------------------------\n');
    fprintf(fid, 'TOP 5 FILES WITH HIGHEST CONCENTRATION (most stationary)\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    [~, sorted_idx] = sort(all_concentrations, 'descend');
    top_n = min(5, length(sorted_idx));
    for i = 1:top_n
        idx = sorted_idx(i);
        fprintf(fid, '%d. %s - Concentration: %.3f, Mean: %.3f m\n', ...
                i, all_files_stats(idx).filename, ...
                all_files_stats(idx).concentration, all_files_stats(idx).mean_range);
    end
    
    % Files closest to expected position
    fprintf(fid, '\n--------------------------------------------------------------------------------\n');
    fprintf(fid, 'TOP 5 FILES CLOSEST TO EXPECTED POSITION (%.1f m)\n', expected_range);
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    distance_from_expected = abs(all_means - expected_range);
    [~, sorted_idx] = sort(distance_from_expected, 'ascend');
    top_n = min(5, length(sorted_idx));
    for i = 1:top_n
        idx = sorted_idx(i);
        fprintf(fid, '%d. %s - Mean: %.3f m (%.3f%% within tolerance)\n', ...
                i, all_files_stats(idx).filename, ...
                all_files_stats(idx).mean_range, all_files_stats(idx).near_expected_pct);
    end
end

fprintf(fid, '\n\n');
fprintf(fid, '################################################################################\n');
fprintf(fid, '#                          END OF REPORT                                       #\n');
fprintf(fid, '################################################################################\n');

% Close file
fclose(fid);

% Also save MATLAB data
save('batch_analysis_results.mat', 'all_files_stats', 'subjects', 'times', ...
     'range_bin_centers', 'range_bin_edges');

% Print completion message
fprintf('\n================================================================================\n');
fprintf('PROCESSING COMPLETE!\n');
fprintf('================================================================================\n');
fprintf('Total files processed: %d\n', file_count);
fprintf('Report saved to: %s\n', report_filename);
fprintf('MATLAB data saved to: batch_analysis_results.mat\n');
fprintf('================================================================================\n');