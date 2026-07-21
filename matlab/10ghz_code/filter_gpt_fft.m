function filter_gpt_fft(data, output_folder, subject, time)
%FILTER_GPT_FFT  Frequency-domain (FFT) range gating with tapered mask.
%   Keeps only beat freqs corresponding to 0.9–3.0 m (≈2.93–9.77 kHz),
%   applied per chirp (FFT on fast-time), then IFFT back to time domain.
%
%   data:          534 x 20 x 100 complex double (fast-time x chirps x frames)
%   output_folder: folder where 'filtered_fft.mat' will be saved

    % ---------- Radar / sampling params ----------
    fs       = 520834;                 % Sampling rate [Hz]
    B        = 500e6;                  % FMCW bandwidth [Hz]
    T_chirp  = 1024e-6;                % Chirp duration [s]
    c        = physconst('LightSpeed');

    r_min_m  = 0.9;                    % Range gate [m]
    r_max_m  = 3.0;

    % ---------- Mask edge taper (transition) ----------
    tw_hz    = 500;                    % Transition width [Hz] for smooth skirts

    % ---------- Range -> beat frequency ----------
    S = B / T_chirp;                   % Chirp slope [Hz/s]
    k = 2*S/c;                         % Hz per meter
    f_min = k * r_min_m;               % ~ 2.93 kHz
    f_max = k * r_max_m;               % ~ 9.77 kHz

    % ---------- Precompute frequency axis (fftshifted) ----------
    [Nfast, Nchirps, Nframes] = size(data);
    df   = fs / Nfast;
    f_ax = (-floor(Nfast/2):ceil(Nfast/2)-1).' * df;   % column, length Nfast (fftshifted axis)

    % Build symmetric, tapered passband mask on |f|
    pass_lo = max(0,     f_min);
    pass_hi = min(fs/2,  f_max);
    trans   = max(0, tw_hz);

    % Helper to make a 1D raised-cosine (Hann) edge from f1->f2 (monotone slope)
    hann_up = @(x,f1,f2) 0.5*(1 - cos(pi*min(max((x-f1)/(f2-f1),0),1)));
    hann_dn = @(x,f1,f2) 0.5*(1 + cos(pi*min(max((x-f1)/(f2-f1),0),1)));

    % Use absolute frequency so mask is symmetric about DC (works for complex data too)
    af = abs(f_ax);

    M = zeros(size(f_ax));  % start with stopband

    % 1) Lower transition: [pass_lo-trans, pass_lo]
    if trans > 0 && pass_lo > 0
        idx = af >= (pass_lo - trans) & af < pass_lo;
        M(idx) = hann_up(af(idx), pass_lo - trans, pass_lo);
    end

    % 2) Passband: [pass_lo, pass_hi]
    idx = af >= pass_lo & af <= pass_hi;
    M(idx) = 1;

    % 3) Upper transition: [pass_hi, pass_hi+trans]
    if trans > 0
        idx = af > pass_hi & af <= (pass_hi + trans);
        M(idx) = hann_dn(af(idx), pass_hi, pass_hi + trans);
    end

    % Clip (numerical safety)
    M = min(max(M,0),1);

    % ---------- Apply mask per chirp & frame ----------
    filtered_data = complex(zeros(Nfast, Nchirps, Nframes));

    for fr = 1:Nframes
        for ch = 1:Nchirps
            x  = double(data(:, ch, fr));      % complex ok
            X  = fftshift(fft(x, Nfast, 1), 1); % fast-time FFT then shift
            Y  = X .* M;                        % apply tapered mask
            y  = ifft(ifftshift(Y, 1), Nfast, 1);
            filtered_data(:, ch, fr) = y;
        end
    end

    % ---------- Save ----------
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    file_name = "subject_" + subject + "_" + time + "_fil_gpt_FFT.mat";
    save(fullfile(output_folder, file_name), 'filtered_data');

    % Optional summary
    % fprintf('[FFT-gate] PB=[%.1f %.1f] kHz, TW=±%.1f kHz; N=%d; df=%.1f Hz\n', ...
    %     f_min/1e3, f_max/1e3, trans/1e3, Nfast, df);
end
