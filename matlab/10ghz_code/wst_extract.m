    function wst_extract(filtered_data, subjectID, timeLabel, filterTag, varargin)
%WST_EXTRACT  Extract 12 WST feature sets (A/B × mag/IQ × 3 tilings) with rich pooling.
%
%   wst_extract(FILTERED_DATA, SUBJECTID, TIMELABEL, FILTERTAG, 'Name',Value,...)
%
%   FILTERED_DATA : complex double [Nsamples x Nchirps x Nframes] (534 x 20 x 100)
%   SUBJECTID     : scalar or string/char (e.g., 6)
%   TIMELABEL     : string/char (e.g., "8am", "10am", ...)
%   FILTERTAG     : string/char (e.g., "Butterworth", "FIR", "FFT")
%
%   Name-Value options (defaults tuned to your setup):
%     'fs'            : 520834        % Hz
%     'B'             : 500e6         % Hz
%     'Tchirp'        : 1024e-6       % s
%     'rmin'          : 0.9           % m
%     'rmax'          : 3.0           % m
%     'EdgeTrim'      : 64            % samples trimmed at both ends before WST
%     'PeakNeighbors' : 1             % ±bins kept around peak for Option B
%     'MaskTaper'     : true          % tapered mask for Option B reconstruction
%     'Tilings'       : struct array with fields:
%                         .Q  (scalar or [Q1 Q2])
%                         .InvScale_ms  (invariance scale in ms)
%                       Default (3 tilings):
%                         (1) Q=[16 8], InvScale_ms=0.10
%                         (2) Q=[10 4], InvScale_ms=0.20
%                         (3) Q=[8  8], InvScale_ms=0.35
%
%   Pooling:
%     - ScatteringOrders: [0 1 2] (keep S0,S1,S2)
%     - Stats per path: mean & std
%     - Regions: global, first half, second half  => 3 regions × 2 stats = 6 numbers per path
%
%   Output:
%     Saves file: ./wst_features_fil_<FILTERTAG>/wst_features_subject_<ID>_<TIME>_filtered_<FILTERTAG>.mat
%     Variables inside (12 feature matrices + 12 meta structs):
%       feat_A_mag_T1, feat_A_mag_T2, feat_A_mag_T3
%       feat_A_iq_T1,  feat_A_iq_T2,  feat_A_iq_T3
%       feat_B_mag_T1, feat_B_mag_T2, feat_B_mag_T3
%       feat_B_iq_T1,  feat_B_iq_T2,  feat_B_iq_T3
%       meta_A_mag_T1, ..., meta_B_iq_T3

    % ---------- Parse inputs ----------
    p = inputParser;
    p.addParameter('fs', 520834, @(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addParameter('B', 500e6, @(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addParameter('Tchirp', 1024e-6, @(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addParameter('rmin', 0.9, @(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addParameter('rmax', 3.0, @(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addParameter('EdgeTrim', 32, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
    p.addParameter('PeakNeighbors', 1, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
    p.addParameter('MaskTaper', true, @(x)islogical(x)||ismember(x,[0 1]));
    p.addParameter('StoreSx', true, @(x)islogical(x) || ismember(x,[0 1]));
    % Default tilings: three complementary configs
    defaultTilings = struct( ...
        'Q',          { [10 4], [8 2], [6 2] }, ...
        'InvScale_ms',{ 0.20,   0.30,   0.40  } );
    p.addParameter('Tilings', defaultTilings, @(t)isstruct(t) && all(isfield(t,{'Q','InvScale_ms'})));
    p.parse(varargin{:});
    S = p.Results;

    sz = size(filtered_data);
    Nsamples = sz(1); Nchirps = sz(2); Nframes = sz(3);

    % ---------- Build 12 sets: 4 variants × |Tilings| ----------
    variants = { ...
        'A','mag'; ...
        'A','iq' ; ...
        'B','mag'; ...
        'B','iq'  ...
    };

    % Collect outputs to save
    outVars = struct();

    for t = 1:numel(S.Tilings)
        Qcfg   = S.Tilings(t).Q;
        Inv_ms = S.Tilings(t).InvScale_ms;

        for v = 1:size(variants,1)
            modeStr = lower(variants{v,1});  % 'a' or 'b'
            chanStr = lower(variants{v,2});  % 'mag' or 'iq'
            % map 'a'/'b' to internal 'avg'/'peak'
            modeMap = struct('a','avg','b','peak');
            modeUse = modeMap.(modeStr);

            [features, featuresWST, meta] = extract_variant_tiled(filtered_data, modeUse, chanStr, S, Qcfg, Inv_ms);

            % Name features/metadata for saving
            varNameFeat = sprintf('feat_%s_%s_T%d', upper(modeStr), lower(chanStr), t);
            varNameWST  = sprintf('featuresWST_%s_%s_T%d', upper(modeStr), lower(chanStr), t);
            varNameMeta = sprintf('meta_%s_%s_T%d', upper(modeStr), lower(chanStr), t);
            outVars.(varNameFeat) = features;
            outVars.(varNameWST)  = featuresWST;
            outVars.(varNameMeta) = meta;
        end
    end

    % ---------- Make folder and save ----------
    folderName = sprintf('samples_wst');
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
    fileName = sprintf('wst_features_subject_%s_%s_filtered_%s.mat', string(subjectID), string(timeLabel), string(filterTag));

    % Save all fields of outVars into the MAT-file
    save(fullfile(folderName, fileName), '-struct', 'outVars');

    fprintf('[WST] Saved %s with %d tilings × 4 variants = %d sets (frames=%d).\n', ...
        fullfile(folderName, fileName), numel(S.Tilings), 4*numel(S.Tilings), Nframes);

    % ===================== NESTED HELPER (tiling-specific) =====================
    function [features, featuresWST, meta] = extract_variant_tiled(dataCube, modeStr, chanStr, Slocal, Qcfg, Inv_ms)
        % Build one feature matrix for a given (Mode × Channel × Tiling)
        N = size(dataCube,1);
        trim = min(Slocal.EdgeTrim, floor(N/4));
        effLen = N - 2*trim;
        if effLen < 32
            error('EdgeTrim too large; effective length %d is too short.', effLen);
        end

        % Range->beat mapping (for Option B bin search)
        c0 = physconst('LightSpeed');
        slope = Slocal.B / Slocal.Tchirp;      % Hz/s
        HzPerM = (2*slope)/c0;                 % Hz per meter
        fmin = HzPerM * Slocal.rmin;           % ~2.93 kHz
        fmax = HzPerM * Slocal.rmax;           % ~9.77 kHz
        df = Slocal.fs / N;
        halfIdx = 1:floor(N/2);
        f1 = (halfIdx - 1).' * df;
        roi = (f1 >= fmin) & (f1 <= fmax);
        if ~any(roi)
            error('No frequency bins fall in ROI [%.1f, %.1f] Hz. Check parameters.', fmin, fmax);
        end

        % Scattering object for this tiling
        invScale_sec = min(Inv_ms*1e-3, 0.5*(N/Slocal.fs));  % ensure < chirp (safer upper bound)
        scObj = waveletScattering( ...
            'SignalLength', effLen, ...
            'SamplingFrequency', Slocal.fs, ...
            'InvarianceScale', invScale_sec, ...
            'QualityFactors', Qcfg);

        Nf = size(dataCube,3);
        featCell = cell(1, Nf);
        nChan = strcmpi(chanStr, 'iq') + 1;
        Sx_store = cell(Nf, nChan);

        % Optional: capture peak info for 'peak' mode
        peakInfo.framePeakBin = nan(1, Nf);
        peakInfo.framePeakHz  = nan(1, Nf);

        for fr = 1:Nf
            X = dataCube(:,:,fr);   % [Nsamples x Nchirps]

            switch lower(modeStr)
                case 'avg'   % Option A
                    s = mean(X, 2);

                case 'peak'  % Option B
                    % Find dominant bin in ROI (avg over chirps)
                    w = hann(N,'periodic');
                    Psum = zeros(numel(f1),1);
                    for ch = 1:size(X,2)
                        Xi = fft(X(:,ch).*w, N, 1);
                        Psum = Psum + abs(Xi(halfIdx)).^2;
                    end
                    Pavg = Psum / size(X,2);
                    Pavg(~roi) = 0;
                    [~, idxRel] = max(Pavg);
                    peakBin = halfIdx(idxRel);
                    peakHz  = (peakBin-1)*df;
                    peakInfo.framePeakBin(fr) = peakBin;
                    peakInfo.framePeakHz(fr)  = peakHz;

                    % ±PeakNeighbors mask (two-sided, fftshifted)
                    nb = Slocal.PeakNeighbors;
                    keep = false(N,1);
                    posBins = peakBin + (0:nb);
                    posBins = posBins(posBins <= floor(N/2));
                    negBins = N - posBins + 2;
                    idxKeep = unique([posBins, negBins]);
                    keep(idxKeep) = true;

                    % Optional taper over kept region
                    M = double(keep);
                    if Slocal.MaskTaper
                        idx = find(keep);
                        if numel(idx) > 2
                            M(idx) = hann(numel(idx));
                        end
                    end

                    % Apply mask to each chirp, IFFT, then average chirps
                    ysum = zeros(N,1);
                    for ch = 1:size(X,2)
                        Xc = fftshift(fft(X(:,ch), N, 1), 1);
                        Yc = Xc .* fftshift(M,1);
                        yc = ifft(ifftshift(Yc,1), N, 1);
                        ysum = ysum + yc;
                    end
                    s = ysum / size(X,2);

                otherwise
                    error('Unknown Mode: %s. Use ''avg'' or ''peak''.', modeStr);
            end

            % Edge trim
            s = s(1+trim:end-trim);

            % Channel mapping to real signals for WST
            switch lower(chanStr)
                case 'mag'
                    xlist = { standardize_robust(abs(s)) };
                case 'iq'
                    xr = standardize_robust(real(s));
                    xi = standardize_robust(imag(s));
                    xlist = { xr, xi }; % concatenate later
                otherwise
                    error('Unknown Channel: %s. Use ''mag'' or ''iq''.', chanStr);
            end

            % Scattering -> feature matrix -> pooled stats
            featParts = cell(1, numel(xlist));
            Sx_list   = cell(1, numel(xlist));
            for cc = 1:numel(xlist)
                Sx = featureMatrix(scObj, xlist{cc});
                Sx_list{cc} = Sx;
                featParts{cc} = pool_mean_std_3regions(Sx);
            end
            % Concatenate channels if I/Q
            featCell{fr} = horzcat(featParts{:});

            vI = reshape(Sx_list{1}.', 1, []);
            
            if strcmpi(chanStr,'iq')
                vQ = reshape(Sx_list{2}.', 1, []);
                flatCell{fr} = [vI, vQ];
            else
                flatCell{fr} = vI;
            end
        end

        % Pack to matrix [Nframes x Nfeat]
        Lmax = max(cellfun(@numel, featCell));
        features = zeros(Nf, Lmax);
        for fr = 1:Nf
            v = featCell{fr};
            features(fr,1:numel(v)) = v;
        end

        Lflat = max(cellfun(@numel, flatCell));
        featuresWST = zeros(Nf, Lflat);
        for fr = 1:Nf
            vf = flatCell{fr};
            featuresWST(fr,1:numel(vf)) = vf;
        end

        % Meta
        meta = struct();
        meta.Mode           = modeStr;
        meta.Channel        = chanStr;
        meta.fs             = Slocal.fs;
        meta.B              = Slocal.B;
        meta.Tchirp         = Slocal.Tchirp;
        meta.rmin           = Slocal.rmin;
        meta.rmax           = Slocal.rmax;
        meta.EdgeTrim       = trim;
        meta.Q              = Qcfg;
        meta.InvScale_sec   = Inv_ms*1e-3;
        meta.SignalLength   = effLen;
        meta.df             = df;
        meta.RangeBandHz    = [fmin fmax];
        meta.Sx             = Sx_store; 
        meta.PeakNeighbors  = Slocal.PeakNeighbors;
        meta.MaskTaper      = logical(Slocal.MaskTaper);
        meta.peakInfo       = [];
        if strcmpi(modeStr,'peak')
            meta.peakInfo = peakInfo;
        end
    end

    % ===================== NESTED UTILITIES =====================
    function y = standardize_robust(x)
        mu = mean(x);
        med = median(x);
        madv = median(abs(x - med)) + eps;
        y = (x - mu) / (1.4826*madv);
    end

    function f = pool_mean_std_3regions(Sx)
        % Sx: [Npaths x T] scattering time series
        [P, T] = size(Sx);
        h = floor(T/2);
        idx1 = 1:h;
        idx2 = (h+1):T;
        % Means / STDs per region (columns)
        muG = mean(Sx, 2);         sdG = std(Sx, 0, 2);
        if h > 0
            mu1 = mean(Sx(:,idx1),2); sd1 = std(Sx(:,idx1),0,2);
        else
            mu1 = zeros(P,1); sd1 = zeros(P,1);
        end
        if T-h > 0
            mu2 = mean(Sx(:,idx2),2); sd2 = std(Sx(:,idx2),0,2);
        else
            mu2 = zeros(P,1); sd2 = zeros(P,1);
        end
        % Concatenate in fixed order: [Gmean, Gstd, H1mean, H1std, H2mean, H2std] per path
        Fmat = [muG, sdG, mu1, sd1, mu2, sd2]; % [P x 6]
        f = reshape(Fmat.', 1, []);            % row vector: 6*P
    end
end
