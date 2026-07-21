function wst_extract77(chirpAvg, fused_mean, fused_median, rd_mean, rd_median, ...
                       subjectID, timeLabel, filterTag, varargin)
% WST feature extraction for 77 GHz FMCW radar data.
%
% 6 branches x 3 tilings = 18 feature sets saved (pooled + raw each).
%
% Fast-time branches (WST on ADC range profile, fs=500 kHz):
%   feat_fmean_T#      / featuresWST_fmean_T#      — Rx-fused mean
%   feat_fmed_T#       / featuresWST_fmed_T#       — Rx-fused median
%   feat_prxMean_T#    / featuresWST_prxMean_T#    — per-Rx WST, feature-space mean
%   feat_prxMed_T#     / featuresWST_prxMed_T#     — per-Rx WST, feature-space median
%
% Doppler branches (WST on Doppler spectrum, fs=PRF=1/Tchirp):
%   feat_dopMean_T#    / featuresWST_dopMean_T#    — Rx-fused mean RD map
%   feat_dopMed_T#     / featuresWST_dopMed_T#     — Rx-fused median RD map
%
% Inputs:
%   chirpAvg   [Nframes x Nfast x Nrx]
%   fused_mean [Nframes x Nfast]
%   fused_median [Nframes x Nfast]
%   rd_mean    [Nframes x Nrange x Nchirps]   range-Doppler map, Rx-fused mean
%   rd_median  [Nframes x Nrange x Nchirps]   range-Doppler map, Rx-fused median
%
% Output: samples_wst_77/wst77_features_subject_<ID>_<TIME>_<TAG>.mat

    % ── Input parser ────────────────────────────────────────────────────────
    p = inputParser;
    p.addParameter('fs',      500e3,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addParameter('B',       2e9,    @(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addParameter('Tchirp',  512e-6, @(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addParameter('EdgeTrim',8,      @(x)isnumeric(x)&&isscalar(x)&&x>=0);
    p.addParameter('PadTo',   512,    @(x)isnumeric(x)&&isscalar(x)&&x>=1);

    % Fast-time tilings: signals at fs=500 kHz, InvScale in ms
    defaultFtTilings = struct( ...
        'Q',           {[8 4], [6 4], [4 2]}, ...
        'InvScale_ms', {0.08,  0.16,  0.20 });

    % Doppler tilings: signals at PRF~1953 Hz, InvScale in ms
    % Max InvScale = 0.5 * (Nchirps/PRF) = 0.5*(256/1953) ≈ 65 ms
    defaultDopTilings = struct( ...
        'Q',           {[8 4], [6 4], [4 2]}, ...
        'InvScale_ms', {20,    40,    60   });

    p.addParameter('Tilings',       defaultFtTilings,  @(t)isstruct(t)&&all(isfield(t,{'Q','InvScale_ms'})));
    p.addParameter('DopplerTilings',defaultDopTilings, @(t)isstruct(t)&&all(isfield(t,{'Q','InvScale_ms'})));
    p.parse(varargin{:});
    S = p.Results;

    PRF = 1 / S.Tchirp;   % Doppler sampling frequency [Hz] ≈ 1953.125 Hz

    % ── Validate input dimensions ────────────────────────────────────────────
    [Nframes, Nfast] = size(fused_mean);
    Nrx              = size(chirpAvg,  3);
    Nrange           = size(rd_mean,   2);
    Nchirps          = size(rd_mean,   3);

    assert(isequal(size(fused_median), [Nframes Nfast]), ...
        'fused_median must be [%d x %d].', Nframes, Nfast);
    assert(size(chirpAvg,1)==Nframes && size(chirpAvg,2)==Nfast, ...
        'chirpAvg first two dims must be [%d x %d].', Nframes, Nfast);
    assert(isequal(size(rd_median), [Nframes Nrange Nchirps]), ...
        'rd_median must match rd_mean size [%d x %d x %d].', Nframes, Nrange, Nchirps);

    % ── Overwrite check ──────────────────────────────────────────────────────
    outFolder = 'samples_wst_77';
    if ~exist(outFolder, 'dir'), mkdir(outFolder); end
    outName = sprintf('wst77_features_subject_%s_%s_%s.mat', ...
        string(subjectID), string(timeLabel), string(filterTag));
    outPath = fullfile(outFolder, outName);
    if exist(outPath, 'file')
        fprintf('[WST] Skipping (already exists): %s\n', outName);
        return;
    end

    out = struct();

    % ── Main loop: one scattering object pair per tiling ────────────────────
    for t = 1:numel(S.Tilings)
        fprintf('[WST] Tiling %d/%d  Q=%s  InvFt=%.2fms  InvDop=%.0fms\n', ...
            t, numel(S.Tilings), mat2str(S.Tilings(t).Q), ...
            S.Tilings(t).InvScale_ms, S.DopplerTilings(t).InvScale_ms);

        % Fast-time scattering object
        effLen      = Nfast - 2*min(S.EdgeTrim, floor(Nfast/4));
        assert(effLen >= 32, 'EdgeTrim too large; effective length %d < 32.', effLen);
        Nsig_ft     = max(S.PadTo, effLen);
        invScale_ft = min(S.Tilings(t).InvScale_ms*1e-3, 0.5*(Nsig_ft/S.fs));
        scObj_ft    = waveletScattering( ...
            'SignalLength',      Nsig_ft, ...
            'SamplingFrequency', S.fs, ...
            'InvarianceScale',   invScale_ft, ...
            'QualityFactors',    S.Tilings(t).Q);

        % Doppler scattering object
        % Signal is the 256-point Doppler spectrum, sampled at PRF
        invScale_dop = min(S.DopplerTilings(t).InvScale_ms*1e-3, 0.5*(Nchirps/PRF));
        scObj_dop    = waveletScattering( ...
            'SignalLength',      Nchirps, ...
            'SamplingFrequency', PRF, ...
            'InvarianceScale',   invScale_dop, ...
            'QualityFactors',    S.DopplerTilings(t).Q);

        % ── Branch 1: fast-time WST on Rx-fused mean ────────────────────────
        [F1p, F1r, M1] = wst_on_signal_matrix(fused_mean, scObj_ft, S.EdgeTrim, S.PadTo, S);
        out.(sprintf('feat_fmean_T%d',        t)) = F1p;
        out.(sprintf('featuresWST_fmean_T%d', t)) = F1r;
        out.(sprintf('meta_fmean_T%d',        t)) = M1;
        fprintf('  Branch 1 (fmean):  pooled [%dx%d]  raw [%dx%d]\n', ...
            size(F1p,1),size(F1p,2), size(F1r,1),size(F1r,2));

        % ── Branch 2: fast-time WST on Rx-fused median ──────────────────────
        [F2p, F2r, M2] = wst_on_signal_matrix(fused_median, scObj_ft, S.EdgeTrim, S.PadTo, S);
        out.(sprintf('feat_fmed_T%d',        t)) = F2p;
        out.(sprintf('featuresWST_fmed_T%d', t)) = F2r;
        out.(sprintf('meta_fmed_T%d',        t)) = M2;
        fprintf('  Branch 2 (fmed):   pooled [%dx%d]  raw [%dx%d]\n', ...
            size(F2p,1),size(F2p,2), size(F2r,1),size(F2r,2));

        % ── Branches 3 & 4: per-Rx fast-time WST, feature-space fusion ──────
        % Pre-allocate using sizes from Branch 1 (same scObj, same Nfast → same dims)
        pool3D = zeros(Nframes, size(F1p,2), Nrx);
        raw3D  = zeros(Nframes, size(F1r,2), Nrx);
        for rx = 1:Nrx
            Xrx = chirpAvg(:, :, rx);   % [Nframes x Nfast] — no squeeze needed
            [Fp, Fr, ~] = wst_on_signal_matrix(Xrx, scObj_ft, S.EdgeTrim, S.PadTo, S);
            pool3D(:, :, rx) = Fp;
            raw3D(:,  :, rx) = Fr;
        end
        F3p = mean(pool3D,   3);   F3r = mean(raw3D,   3);
        F4p = median(pool3D, 3);   F4r = median(raw3D, 3);

        Mprx         = M1;
        Mprx.Channel = 'perRx';
        Mprx.Fusion  = 'feature-space';

        Tm = Mprx; Tm.FusionOp = 'mean';
        out.(sprintf('feat_prxMean_T%d',        t)) = F3p;
        out.(sprintf('featuresWST_prxMean_T%d', t)) = F3r;
        out.(sprintf('meta_prxMean_T%d',        t)) = Tm;

        Tm = Mprx; Tm.FusionOp = 'median';
        out.(sprintf('feat_prxMed_T%d',         t)) = F4p;
        out.(sprintf('featuresWST_prxMed_T%d',  t)) = F4r;
        out.(sprintf('meta_prxMed_T%d',         t)) = Tm;
        fprintf('  Branch 3&4 (prx):  pooled [%dx%d]  raw [%dx%d]\n', ...
            size(F3p,1),size(F3p,2), size(F3r,1),size(F3r,2));

        % ── Branch 5: Doppler WST on rd_mean ────────────────────────────────
        [F5p, F5r, M5] = wst_on_doppler_matrix(rd_mean, scObj_dop, S, PRF);
        out.(sprintf('feat_dopMean_T%d',        t)) = F5p;
        out.(sprintf('featuresWST_dopMean_T%d', t)) = F5r;
        out.(sprintf('meta_dopMean_T%d',        t)) = M5;
        fprintf('  Branch 5 (dopMean): pooled [%dx%d]  raw [%dx%d]\n', ...
            size(F5p,1),size(F5p,2), size(F5r,1),size(F5r,2));

        % ── Branch 6: Doppler WST on rd_median ──────────────────────────────
        [F6p, F6r, M6] = wst_on_doppler_matrix(rd_median, scObj_dop, S, PRF);
        out.(sprintf('feat_dopMed_T%d',        t)) = F6p;
        out.(sprintf('featuresWST_dopMed_T%d', t)) = F6r;
        out.(sprintf('meta_dopMed_T%d',        t)) = M6;
        fprintf('  Branch 6 (dopMed):  pooled [%dx%d]  raw [%dx%d]\n', ...
            size(F6p,1),size(F6p,2), size(F6r,1),size(F6r,2));
    end

    save(outPath, '-struct', 'out', '-v7.3');
    fprintf('[WST] Saved %s  (frames=%d, 6 branches x 3 tilings = 18 sets)\n', ...
        outPath, Nframes);
end


% ═══════════════════════════════════════════════════════════════════════════
% Helper: fast-time WST on a matrix of range profiles
% ═══════════════════════════════════════════════════════════════════════════
function [features_pool, features_raw, meta] = wst_on_signal_matrix(X, scObj, EdgeTrim, PadTo, S)
% X: [Nframes x Nfast]  one range profile per row
% Returns:
%   features_pool [Nframes x 6*P]   mean+std over global and two halves
%   features_raw  [Nframes x T*P]   full flattened scattering matrix

    [Nframes, Nfast] = size(X);
    trim   = min(EdgeTrim, floor(Nfast/4));
    effLen = Nfast - 2*trim;

    features_pool = [];
    features_raw  = [];
    P = 0; T = 0;

    for fr = 1:Nframes
        s = double(X(fr, :).');
        s = s(1+trim : end-trim);
        s = standardize_robust(s);
        if numel(s) < PadTo
            s = [s; zeros(PadTo - numel(s), 1)];
        else
            s = s(1:PadTo);
        end

        Sx = featureMatrix(scObj, s);   % [P x T]

        if fr == 1
            [P, T] = size(Sx);
            features_pool = zeros(Nframes, 6*P);
            features_raw  = zeros(Nframes, T*P);
        end

        features_pool(fr, :) = pool_mean_std_halves(Sx);
        features_raw(fr,  :) = reshape(Sx.', 1, []);
    end

    meta = struct();
    meta.fs           = S.fs;
    meta.B            = S.B;
    meta.Tchirp       = S.Tchirp;
    meta.EdgeTrim     = trim;
    meta.EffectiveLen = effLen;
    meta.PadTo        = PadTo;
    meta.SignalLength  = scObj.SignalLength;
    meta.Q            = scObj.QualityFactors;
    meta.InvScale_sec = scObj.InvarianceScale;
    meta.WSTShape     = [P, T];   % paths x time-steps
    meta.Channel      = 'fused';
    meta.Fusion       = 'signal-space';
    meta.FusionOp     = '';
end


% ═══════════════════════════════════════════════════════════════════════════
% Helper: Doppler WST on a range-Doppler cube
% ═══════════════════════════════════════════════════════════════════════════
function [features_pool, features_raw, meta] = wst_on_doppler_matrix(rd, scObj_dop, S, PRF)
% rd: [Nframes x Nrange x Nchirps]
%
% For each frame:
%   1. For each of the Nrange range bins, apply WST to the Nchirps-point
%      Doppler magnitude profile.
%   2. Average the scattering matrices across range bins.
%      This equally weights all range bins regardless of reflection strength.
%   3. Pool the averaged scattering matrix into a compact feature vector.
%
% Averaging in scattering space (step 2) rather than averaging raw Doppler
% profiles (then one WST) is important: the WST is nonlinear, so the order
% of operations matters. Averaging after WST preserves per-bin structure
% while still combining information across the target extent.

    [Nframes, Nrange, Nchirps] = size(rd);

    features_pool = [];
    features_raw  = [];
    P = 0; T = 0;

    for fr = 1:Nframes
        Sx_sum = zeros(0);   % will be sized on first range bin

        for r = 1:Nrange
            s = double(reshape(rd(fr, r, :), [], 1));   % [Nchirps x 1]
            s = standardize_robust(s);
            Sx = featureMatrix(scObj_dop, s);            % [P x T]

            if r == 1
                Sx_sum = Sx;
            else
                Sx_sum = Sx_sum + Sx;
            end
        end

        Sx_avg = Sx_sum / Nrange;   % [P x T] averaged across range bins

        if fr == 1
            [P, T] = size(Sx_avg);
            features_pool = zeros(Nframes, 6*P);
            features_raw  = zeros(Nframes, T*P);
        end

        features_pool(fr, :) = pool_mean_std_halves(Sx_avg);
        features_raw(fr,  :) = reshape(Sx_avg.', 1, []);
    end

    meta = struct();
    meta.fs           = S.fs;
    meta.B            = S.B;
    meta.Tchirp       = S.Tchirp;
    meta.PRF          = PRF;
    meta.Nrange       = Nrange;
    meta.Nchirps      = Nchirps;
    meta.SignalLength  = scObj_dop.SignalLength;
    meta.Q            = scObj_dop.QualityFactors;
    meta.InvScale_sec = scObj_dop.InvarianceScale;
    meta.WSTShape     = [P, T];
    meta.Channel      = 'Doppler';
    meta.Fusion       = 'range-bin averaged scattering';
end


% ═══════════════════════════════════════════════════════════════════════════
% Helper: robust standardization — median centering, MAD scaling
% ═══════════════════════════════════════════════════════════════════════════
function y = standardize_robust(x)
% Uses median for both location and scale to be consistent.
% The 1.4826 factor makes MAD a consistent estimator of std for Gaussian data.
    med  = median(x);
    madv = median(abs(x - med)) + eps;
    y    = (x - med) / (1.4826 * madv);
end


% ═══════════════════════════════════════════════════════════════════════════
% Helper: pool scattering matrix into mean+std over global and two halves
% ═══════════════════════════════════════════════════════════════════════════
function f = pool_mean_std_halves(Sx)
% Sx: [P x T]
% Returns [1 x 6*P]: for each path, [global_mean, global_std,
%                                     first_half_mean, first_half_std,
%                                     second_half_mean, second_half_std]
    [P, T] = size(Sx);
    h = floor(T / 2);

    muG = mean(Sx,  2);   sdG = std(Sx,  0, 2);

    if h > 0
        mu1 = mean(Sx(:, 1:h),   2);   sd1 = std(Sx(:, 1:h),   0, 2);
    else
        mu1 = zeros(P, 1);              sd1 = zeros(P, 1);
    end

    if T - h > 0
        mu2 = mean(Sx(:, h+1:T), 2);   sd2 = std(Sx(:, h+1:T), 0, 2);
    else
        mu2 = zeros(P, 1);              sd2 = zeros(P, 1);
    end

    f = reshape([muG, sdG, mu1, sd1, mu2, sd2].', 1, []);   % [1 x 6*P]
end