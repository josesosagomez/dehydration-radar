function dsReport = wst_integrity_check_dataset(data, varargin)
%WST_INTEGRITY_CHECK_DATASET  Run integrity checks over all frames of a filtered cube.
%   DSREPORT = wst_integrity_check_dataset(DATA, 'Name', Value, ...)
%
%   DATA: complex double, size [Nsamples x Nchirps x Nframes] (default expected [534 x 20 x 100])
%
%   Name-Value (all optional; tuned to your setup):
%     'ExpectedSize'     [Nsamples Nchirps Nframes], default [534 20 100]
%     'fs'               sampling rate (Hz), default 520834
%     'B'                FMCW bandwidth (Hz), default 500e6
%     'Tchirp'           chirp duration (s), default 1024e-6
%     'rmin'             min range of interest (m), default 0.9
%     'rmax'             max range of interest (m), default 3.0
%     'BandMarginHz'     band margin per side for energy ratio, default 500
%     'RMSOutlierZ'      robust z-score threshold for chirp RMS outliers, default 4
%     'FlatlineFrac'     fraction of identical-magnitude samples to flag flatlines, default 0.25
%
%   DSREPORT fields:
%     .overall_pass           logical, true if all frames passed basic checks
%     .frame_pass             1xN logical per-frame pass
%     .messages               1xN cell arrays of strings (per-frame)
%     .numNaN                 1xN
%     .numInf                 1xN
%     .maxAbs                 1xN
%     .medianAbs              1xN
%     .rmsPerChirp            1xN cell (each 1xNchirps)
%     .rmsMedian              1xN
%     .rmsMAD                 1xN
%     .rmsOutlierIdx          1xN cell (indices per frame)
%     .flatlineChirps         1xN cell (indices per frame)
%     .inBandEnergyRatio      1xN (0..1)
%     .bandHz                 [fmin fmax]
%     .bandWithMarginHz       [lo hi]
%     .specCentroidHz         1xN
%     .fs, .B, .Tchirp, .rmin, .rmax, .df   echoed params

% ---------- Parse inputs ----------
p = inputParser;
p.addParameter('ExpectedSize', [534 20 100], @(x)isnumeric(x)&&numel(x)==3);
p.addParameter('fs', 520834, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('B', 500e6, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('Tchirp', 1024e-6, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('rmin', 0.9, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('rmax', 3.0, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('BandMarginHz', 1000, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('RMSOutlierZ', 4.5, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FlatlineFrac', 0.25, @(x)isnumeric(x)&&isscalar(x)&&x>=0&&x<=1);
p.parse(varargin{:});
S = p.Results;

% ---------- Basic dataset checks ----------
sz = size(data);
if numel(sz) ~= 3
    error('DATA must be a 3D matrix [Nsamples x Nchirps x Nframes].');
end
sizeOK = all(sz == S.ExpectedSize);
if ~sizeOK
    warning('Size mismatch: got [%d x %d x %d], expected [%d x %d x %d].', ...
        sz(1), sz(2), sz(3), S.ExpectedSize(1), S.ExpectedSize(2), S.ExpectedSize(3));
end

Nsamples = sz(1); Nchirps = sz(2); Nframes = sz(3);

% ---------- Precompute constant mapping ----------
c0     = physconst('LightSpeed');
Slope  = S.B / S.Tchirp;          % Hz/s
k_hzpm = (2*Slope) / c0;          % Hz per meter
fmin   = k_hzpm * S.rmin;
fmax   = k_hzpm * S.rmax;
lo     = max(0,     fmin - S.BandMarginHz);
hi     = min(S.fs/2, fmax + S.BandMarginHz);

df     = S.fs / Nsamples;
half   = 1:floor(Nsamples/2);
f1     = ((half-1)') * df;        % one-sided freq axis (column)

% ---------- Allocate outputs ----------
frame_pass          = false(1, Nframes);
messages            = cell(1, Nframes);
numNaN              = zeros(1, Nframes);
numInf              = zeros(1, Nframes);
maxAbs              = zeros(1, Nframes);
medianAbs           = zeros(1, Nframes);
rmsPerChirp_cell    = cell(1, Nframes);
rmsMedian           = zeros(1, Nframes);
rmsMAD              = zeros(1, Nframes);
rmsOutlierIdx_cell  = cell(1, Nframes);
flatlineChirps_cell = cell(1, Nframes);
inBandEnergyRatio   = zeros(1, Nframes);
specCentroidHz      = zeros(1, Nframes);

w = hann(Nsamples,'periodic');

% ---------- Iterate frames ----------
for fr = 1:Nframes
    msgs = {};
    F = data(:,:,fr); % [Nsamples x Nchirps]

    % NaN/Inf
    numNaN(fr) = sum(isnan(F(:)));
    numInf(fr) = sum(isinf(F(:)));
    if numNaN(fr)>0, msgs{end+1} = sprintf('Found %d NaN samples.', numNaN(fr)); end
    if numInf(fr)>0, msgs{end+1} = sprintf('Found %d Inf samples.', numInf(fr)); end

    % Magnitudes & per-chirp RMS
    absF = abs(F);
    maxAbs(fr)    = max(absF(:));
    medianAbs(fr) = median(absF(:));
    rmsCh = sqrt(mean(absF.^2, 1));             % 1 x Nchirps
    rmsCh = reshape(rmsCh, 1, []);
    rmsPerChirp_cell{fr} = rmsCh;
    medR = median(rmsCh);
    madR = median(abs(rmsCh - medR)) + eps;
    rmsMedian(fr) = medR;
    rmsMAD(fr)    = madR;
    robZ = abs(rmsCh - medR) / (1.4826*madR);
    outIdx = find(robZ > S.RMSOutlierZ);
    rmsOutlierIdx_cell{fr} = outIdx;
    if ~isempty(outIdx)
        msgs{end+1} = sprintf('RMS outliers at chirps: %s', mat2str(outIdx));
    end

    % Flatline heuristic (magnitude histogram)
    flatCh = [];
    nbins = max(10, min(200, round(Nsamples/2)));
    for ch = 1:Nchirps
        mag = abs(F(:, ch));
        [counts, ~] = histcounts(mag, nbins);
        if max(counts) >= S.FlatlineFrac * Nsamples
            flatCh(end+1) = ch; %#ok<AGROW>
        end
    end
    flatlineChirps_cell{fr} = flatCh;
    if ~isempty(flatCh)
        msgs{end+1} = sprintf('Possible flatline/saturation in chirps: %s', mat2str(flatCh));
    end

    % In-band energy ratio and spectral centroid (avg over chirps)
    Psum = zeros(numel(f1),1);
    cNum = 0; cDen = 0;
    for ch = 1:Nchirps
        x = F(:, ch) .* w;
        X = fft(x);
        P = abs(X(half)).^2;             % one-sided power (relative scale ok)
        Psum = Psum + P;
        cNum = cNum + sum(f1 .* P);
        cDen = cDen + sum(P);
    end
    Pavg = Psum / Nchirps;
    mask = (f1 >= lo) & (f1 <= hi);
    inBandEnergyRatio(fr) = sum(Pavg(mask)) / max(sum(Pavg), eps);
    specCentroidHz(fr)    = cNum / max(cDen, eps);

    if inBandEnergyRatio(fr) < 0.3
        msgs{end+1} = sprintf('Low in-band energy ratio: %.2f', inBandEnergyRatio(fr));
    end
    if specCentroidHz(fr) < (fmin*0.7) || specCentroidHz(fr) > (fmax*1.3)
        msgs{end+1} = sprintf('Spectral centroid %.1f Hz outside expected band [%.1f–%.1f] Hz.', ...
            specCentroidHz(fr), fmin, fmax);
    end

    % Per-frame pass decision (basic)
    frame_pass(fr) = (numNaN(fr)==0) && (numInf(fr)==0) && isempty(flatCh);
    messages{fr}   = msgs(:);
end

dsReport = struct();
dsReport.overall_pass        = all(frame_pass);
dsReport.frame_pass          = frame_pass;
dsReport.messages            = messages;

dsReport.numNaN              = numNaN;
dsReport.numInf              = numInf;
dsReport.maxAbs              = maxAbs;
dsReport.medianAbs           = medianAbs;

dsReport.rmsPerChirp         = rmsPerChirp_cell;
dsReport.rmsMedian           = rmsMedian;
dsReport.rmsMAD              = rmsMAD;
dsReport.rmsOutlierIdx       = rmsOutlierIdx_cell;
dsReport.flatlineChirps      = flatlineChirps_cell;

dsReport.inBandEnergyRatio   = inBandEnergyRatio;
dsReport.bandHz              = [fmin fmax];
dsReport.bandWithMarginHz    = [lo hi];
dsReport.specCentroidHz      = specCentroidHz;

dsReport.fs                  = S.fs;
dsReport.B                   = S.B;
dsReport.Tchirp              = S.Tchirp;
dsReport.rmin                = S.rmin;
dsReport.rmax                = S.rmax;
dsReport.df                  = df;
dsReport.ExpectedSize        = S.ExpectedSize;

% Dataset size sanity
dsReport.sizeOK              = sizeOK;
dsReport.dataSize            = sz;

end
