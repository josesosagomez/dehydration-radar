function filter_gpt_butterworth(data, output_folder, subject, time)
  
    fs = 520834;               % Sampling frequency (Hz)
    B = 500e6;                 % FMCW bandwidth (Hz)
    T_chirp = 1024e-6;         % Chirp duration (s)
    c = physconst('LightSpeed');
    r_min = 0.9;               % Min range (meters)
    r_max = 3.0;               % Max range (meters)
    filt_order = 4;            % Butterworth filter order
    S = B / T_chirp;           % Slope

    % Convert range to beat frequencies
    f_min = 2 * S * r_min / c;  % Beat frequency corresponding to r_min
    f_max = 2 * S * r_max / c;  % Beat frequency corresponding to r_max

    % Normalize to Nyquist frequency
    Wn = [f_min f_max] / (fs / 2);
    if any(Wn >= 1)
        error('Normalized cutoff frequency exceeds Nyquist. Check filter parameters.');
    end

    % Design SOS Butterworth bandpass filter
    [sos, g] = butter(filt_order, Wn, 'bandpass');

    % Preallocate filtered data
    [Nfast, Nchirps, Nframes] = size(data);
    filtered_data = complex(zeros(Nfast, Nchirps, Nframes));

    % Apply filtering
    for frame = 1:Nframes
        for chirp = 1:Nchirps
            signal = double(data(:, chirp, frame));
            filtered_data(:, chirp, frame) = filtfilt(sos, g, signal);
        end
    end

    % Ensure output folder exists
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Save filtered result
    file_name = "subject_" + subject + "_" + time + "_fil_BW.mat";
    save(fullfile(output_folder, file_name), 'filtered_data');
end
