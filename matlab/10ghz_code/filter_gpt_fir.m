function filter_gpt_fir(data, output_folder, subject, time)
%FILTER_GPT_FIR  Short-order linear-phase FIR bandpass for 0.9–3.0 m.
%   Uses order=160 Kaiser-window FIR + filtfilt (safe for 534-sample chirps).
%
%   data:          534 x 20 x 100 complex double (fast-time x chirps x frames)
%   output_folder: folder where 'filtered_fir.mat' will be saved

    % ---------- Radar / sampling params ----------
    fs       = 520834;                 % Sampling rate [Hz]
    B        = 500e6;                  % FMCW bandwidth [Hz]
    T_chirp  = 1024e-6;                % Chirp duration [s]
    c        = physconst('LightSpeed');

    r_min_m  = 0.9;                    % Range gate [m]
    r_max_m  = 3.0;

    % ---------- FIR design parameters (short order) ----------
    N        = 160;                    % Filter order (must satisfy 3*N < 534)
    As_dB    = 60;                     % ~stopband attenuation target
    % Kaiser beta from As (approx): beta = 0.1102*(As-8.7) for As>50
    beta     = 0.1102*(As_dB - 8.7);   % ≈ 5.65

    % ---------- Range -> beat frequency mapping ----------
    S = B / T_chirp;                   % Chirp slope [Hz/s]
    k = 2*S/c;                         % Hz per meter
    f_min = k * r_min_m;               % ~ 2.93 kHz
    f_max = k * r_max_m;               % ~ 9.77 kHz

    % Normalized band edges for fir1
    Wn = [f_min, f_max] / (fs/2);
    if any(Wn <= 0) || any(Wn >= 1)
        error('FIR design: invalid normalized band edges. Check parameters.');
    end

    % ---------- Design short-order linear-phase FIR (Kaiser window) ----------
    b = fir1(N, Wn, 'bandpass', kaiser(N+1, beta), 'scale');  % symmetric taps

    % ---------- Filtering ----------
    [Nfast, Nchirps, Nframes] = size(data);
    if 3*N >= Nfast
        error('filtfilt requirement not met: need 3*order < Nfast (got %d >= %d).', 3*N, Nfast);
    end

    filtered_data = complex(zeros(Nfast, Nchirps, Nframes));
    for fr = 1:Nframes
        for ch = 1:Nchirps
            x = double(data(:, ch, fr));           % complex ok
            y = filtfilt(b, 1, x);                 % zero-phase
            filtered_data(:, ch, fr) = y;
        end
    end

    % ---------- Save ----------
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    file_name = "subject_" + subject + "_" + time + "_fil_gpt_FIR.mat";
    save(fullfile(output_folder, file_name), 'filtered_data');

    % Optional summary
    % fprintf('[FIR-short] Order=%d, PB=[%.1f %.1f] kHz, As≈%d dB\n', ...
    %     N, f_min/1e3, f_max/1e3, As_dB);
end

