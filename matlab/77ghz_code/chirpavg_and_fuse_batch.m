function chirpavg_and_fuse_batch(inFolder, outFolder)
    % Chirp average, Rx fusion, and range-Doppler maps for all filtered files.
    % Input:  filtered_data [Nframes x Nfast x Nchirps x Nrx]
    % Output per file:
    %   chirpAvg   [Nframes x Nfast x Nrx]     time-domain chirp average
    %   fused_mean   [Nframes x Nfast]          Rx-fused mean  (time-domain)
    %   fused_median [Nframes x Nfast]          Rx-fused median (time-domain)
    %   rd_mean      [Nframes x Nrange x Nchirps]  Rx-fused mean  range-Doppler map
    %   rd_median    [Nframes x Nrange x Nchirps]  Rx-fused median range-Doppler map
    %   range_axis   [1 x Nrange]              range in metres for rd_* bins

    if nargin < 1 || isempty(inFolder),  inFolder  = 'samples';        end
    if nargin < 2 || isempty(outFolder), outFolder = 'samples_prewst'; end
    if ~exist(outFolder, 'dir'), mkdir(outFolder); end

    % Radar params (must match filter step)
    B = 2e9;
    c = physconst('LightSpeed');
    r_min = 2.0;
    r_max = 4.0;

    files = dir(fullfile(inFolder, 'subject_*_*.mat'));
    if isempty(files)
        error('No files found in %s matching subject_*_*.mat', inFolder);
    end
    fprintf('Found %d files.\n', numel(files));

    for k = 1:numel(files)
        inPath = fullfile(files(k).folder, files(k).name);
        base   = files(k).name;

        % Parse subject ID (digits only) and time label — unambiguous for your naming convention
        toks = regexp(base, '^subject_(\d+)_(\w+?)_.*\.mat$', 'tokens', 'once');
        if isempty(toks)
            warning('Cannot parse subject/time from %s — skipping.', base);
            continue;
        end
        outName = sprintf('subject_%s_%s_prewst.mat', toks{1}, toks{2});
        outPath = fullfile(outFolder, outName);

        if exist(outPath, 'file')
            fprintf('  [%2d/%2d] Skipping (already exists): %s\n', k, numel(files), outName);
            continue;
        end
        fprintf('  [%2d/%2d] %s\n', k, numel(files), base);

        S = load(inPath);
        if     isfield(S, 'filtered_data'), X = S.filtered_data;
        elseif isfield(S, 'framesRadar'),   X = S.framesRadar;
            warning('Using framesRadar (unfiltered) from %s.', base);
        else
            warning('Skipping %s: no filtered_data or framesRadar.', base);
            continue;
        end

        if ndims(X) ~= 4
            warning('Skipping %s: expected 4-D, got %s.', base, mat2str(size(X)));
            continue;
        end

        [Nframes, Nfast, Nchirps, Nrx] = size(X);
        X = double(X);

        % ── Time-domain chirp average ─────────────────────────────────────────
        % Average across chirps (dim 3); explicit reshape avoids squeeze pitfalls
        chirpAvg     = reshape(mean(X, 3), Nframes, Nfast, Nrx);  % [Nframes x Nfast x Nrx]
        fused_mean   = mean(chirpAvg,   3);                        % [Nframes x Nfast]
        fused_median = median(chirpAvg, 3);                        % [Nframes x Nfast]

        % ── Range-Doppler map ─────────────────────────────────────────────────
        % Hann windows on both axes to suppress sidelobes
        win_fast = reshape(hann(Nfast),   [1, Nfast,   1,      1]);
        win_slow = reshape(hann(Nchirps), [1, 1,       Nchirps,1]);
        Xw = X .* win_fast .* win_slow;

        % Range FFT (fast-time, dim 2) then Doppler FFT (slow-time, dim 3)
        rd = fft(Xw, Nfast,   2);           % [Nframes x Nfast   x Nchirps x Nrx]
        rd = fft(rd, Nchirps, 3);           % [Nframes x Nfast   x Nchirps x Nrx]
        rd = abs(fftshift(rd, 3));          % magnitude; centre zero-Doppler on dim 3

        % Trim to the range bins that fall within [r_min, r_max]
        % Range axis: bin k (1-indexed) covers range (k-1)*dr metres
        dr         = c / (2 * B);                      % range resolution [m] = 0.075 m
        full_range = (0:Nfast-1) * dr;                 % [1 x Nfast]
        bin_min    = find(full_range >= r_min, 1, 'first');
        bin_max    = find(full_range <= r_max, 1, 'last');
        if isempty(bin_min) || isempty(bin_max) || bin_min > bin_max
            error('Range [%.1f %.1f] m produces no valid bins. Check radar params.', r_min, r_max);
        end

        rd        = rd(:, bin_min:bin_max, :, :);       % [Nframes x Nrange x Nchirps x Nrx]
        rd_mean   = mean(rd,   4);                       % [Nframes x Nrange x Nchirps]
        rd_median = median(rd, 4);                       % [Nframes x Nrange x Nchirps]
        range_axis = full_range(bin_min:bin_max);        % [1 x Nrange] in metres

        save(outPath, 'chirpAvg', 'fused_mean', 'fused_median', ...
             'rd_mean', 'rd_median', 'range_axis', '-v7.3');
        fprintf('      -> %s | chirpAvg%s | rd_mean%s | range [%.2f %.2f] m (%d bins)\n', ...
            outName, mat2str(size(chirpAvg)), mat2str(size(rd_mean)), ...
            range_axis(1), range_axis(end), numel(range_axis));
    end

    fprintf('Done. Outputs in: %s\n', outFolder);
end