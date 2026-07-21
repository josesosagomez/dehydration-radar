function filter_gpt_butterworth77(data, output_folder, subject, time)
    % 77 GHz FMCW Butterworth bandpass (2-4 m) per (frame, chirp, rx).
    % data: [Nframes x Nfast x Nchirps x Nrx]
    % Saves: <output_folder>/subject_<subject>_<time>_fil_BW.mat

    % Radar params
    fs      = 500e3;
    B       = 2e9;
    T_chirp = 512e-6;
    c       = physconst('LightSpeed');
    r_min   = 2.0;
    r_max   = 4.0;

    % Beat frequency band
    S     = B / T_chirp;
    f_min = 2 * S * r_min / c;
    f_max = 2 * S * r_max / c;

    % Normalize and design filter
    % butter(4,'bandpass') -> 8th-order; filtfilt doubles to effective order 16
    Wn = [f_min f_max] / (fs / 2);
    if any(Wn <= 0) || any(Wn >= 1) || Wn(1) >= Wn(2)
        error('Normalized cutoffs invalid: Wn = [%.4f %.4f].', Wn(1), Wn(2));
    end
    [sos, g] = butter(4, Wn, 'bandpass');

    % Validate input
    if ndims(data) ~= 4
        error('Expected 4-D data [Nframes x Nfast x Nchirps x Nrx]. Got: %s', mat2str(size(data)));
    end
    [Nframes, Nfast, Nchirps, Nrx] = size(data);
    assert(size(data,2) == Nfast && size(data,3) == Nchirps, ...
        'Dimension mismatch: check data axis ordering.');

    filtered_data = zeros(Nframes, Nfast, Nchirps, Nrx, 'like', double(0));

    fprintf('Butterworth bandpass: [%.1f kHz, %.1f kHz]  Wn=[%.3f, %.3f]  fs=%.0f kHz\n', ...
        f_min/1e3, f_max/1e3, Wn(1), Wn(2), fs/1e3);
    tic;
    for fr = 1:Nframes
        for rx = 1:Nrx
            X = double(squeeze(data(fr, :, :, rx)));  % [Nfast x Nchirps]

            % Static clutter removal: subtract slow-time mean per fast-time bin
            X = X - mean(X, 2);

            % Zero-phase bandpass along fast-time (first dimension)
            Y = filtfilt(sos, g, X);
            filtered_data(fr, :, :, rx) = reshape(Y, [1, Nfast, Nchirps, 1]);
        end

        if mod(fr, max(1, floor(Nframes/10))) == 0
            fprintf('  Frame %d/%d (%.0f%%)\n', fr, Nframes, 100*fr/Nframes);
        end
    end
    fprintf('Filtering done in %.2f s\n', toc);

    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    file_name = "subject_" + string(subject) + "_" + string(time) + "_fil_BW.mat";
    save(fullfile(output_folder, file_name), 'filtered_data', '-v7.3');
    fprintf('Saved: %s  [%s]\n', fullfile(output_folder, file_name), mat2str(size(filtered_data)));
end