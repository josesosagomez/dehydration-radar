% validate_wst_features.m
% Checks all WST feature files in samples_wst_77/ for:
%   - Missing files
%   - Wrong number of rows (frames)
%   - NaN / Inf values
%   - All-zero rows or columns
%   - Near-constant columns (no variance)
%   - Unexpected feature dimensions
% Prints a per-file, per-variable report and a final summary.

clear; clc;

WST_FOLDER    = 'samples_wst_77';
EXPECTED_ROWS = 125;     % frames per recording
N_TILINGS     = 3;

% All variable prefixes expected in each file
FAST_BRANCHES = {'feat_fmean',   'feat_fmed',   'feat_prxMean',  'feat_prxMed'};
DOP_BRANCHES  = {'feat_dopMean', 'feat_dopMed'};
RAW_PREFIX    = 'featuresWST_';

ALL_POOL_BRANCHES = [FAST_BRANCHES, DOP_BRANCHES];
ALL_BRANCHES      = ALL_POOL_BRANCHES;   % raw names derived below

% ── Find files ──────────────────────────────────────────────────────────────
files = dir(fullfile(WST_FOLDER, 'wst77_features_subject_*.mat'));
if isempty(files)
    error('No files found in %s. Check the folder path.', WST_FOLDER);
end
fprintf('Found %d files in %s\n\n', numel(files), WST_FOLDER);

% ── Per-file tracking ───────────────────────────────────────────────────────
total_vars    = 0;
total_issues  = 0;
files_with_issues = {};

for k = 1:numel(files)
    fpath = fullfile(files(k).folder, files(k).name);
    fname = files(k).name;
    fprintf('══════════════════════════════════════════════════════\n');
    fprintf('[%2d/%2d] %s\n', k, numel(files), fname);

    try
        S = load(fpath);
    catch ME
        fprintf('  !! FAILED TO LOAD: %s\n', ME.message);
        files_with_issues{end+1} = fname;
        total_issues = total_issues + 1;
        continue;
    end

    present_vars = fieldnames(S);
    file_issues  = 0;

    for t = 1:N_TILINGS
        for b = 1:numel(ALL_POOL_BRANCHES)
            branch = ALL_POOL_BRANCHES{b};

            % ── Pooled variable ──────────────────────────────────────────────
            pool_name = sprintf('%s_T%d', branch, t);
            raw_name  = sprintf('%s%s_T%d', RAW_PREFIX, branch(6:end), t);
            % e.g. feat_fmean_T1 -> featuresWST_fmean_T1

            for pass = 1:2
                if pass == 1
                    vname = pool_name;
                else
                    vname = raw_name;
                end

                total_vars = total_vars + 1;

                if ~isfield(S, vname)
                    fprintf('  [T%d] %-40s  MISSING\n', t, vname);
                    file_issues = file_issues + 1;
                    continue;
                end

                M = double(S.(vname));
                [nr, nc] = size(M);

                issues = {};

                % Wrong row count
                if nr ~= EXPECTED_ROWS
                    issues{end+1} = sprintf('rows=%d (expected %d)', nr, EXPECTED_ROWS); %#ok<AGROW>
                end

                % NaN
                n_nan = sum(isnan(M(:)));
                if n_nan > 0
                    issues{end+1} = sprintf('%d NaN', n_nan); 
                end

                % Inf
                n_inf = sum(isinf(M(:)));
                if n_inf > 0
                    issues{end+1} = sprintf('%d Inf', n_inf);
                end

                % All-zero rows
                zero_rows = sum(all(M == 0, 2));
                if zero_rows > 0
                    issues{end+1} = sprintf('%d all-zero rows', zero_rows);
                end

                % All-zero columns
                zero_cols = sum(all(M == 0, 1));
                if zero_cols > 0
                    issues{end+1} = sprintf('%d all-zero cols', zero_cols); 
                end

                % Near-constant columns (std < 1e-8 across frames)
                % These add no discriminative information
                col_std     = std(M, 0, 1);
                const_cols  = sum(col_std < 1e-8);
                if const_cols > 0
                    issues{end+1} = sprintf('%d near-constant cols', const_cols); 
                end

                % Global stats for context
                mu_val  = mean(M(:));
                std_val = std(M(:));
                rng_val = [min(M(:)), max(M(:))];

                if isempty(issues)
                    fprintf('  [T%d] %-40s  OK   [%dx%d]  mean=%.3g  std=%.3g  range=[%.3g, %.3g]\n', ...
                        t, vname, nr, nc, mu_val, std_val, rng_val(1), rng_val(2));
                else
                    issue_str = strjoin(issues, ' | ');
                    fprintf('  [T%d] %-40s  !! ISSUES: %s\n', t, vname, issue_str);
                    fprintf('         [%dx%d]  mean=%.3g  std=%.3g  range=[%.3g, %.3g]\n', ...
                        nr, nc, mu_val, std_val, rng_val(1), rng_val(2));
                    file_issues = file_issues + 1;
                end
            end
        end
    end

    % ── Check for unexpected extra variables ─────────────────────────────────
    expected_prefixes = [cellfun(@(b)sprintf('%s_T', b), ALL_POOL_BRANCHES, 'UniformOutput',false), ...
                         cellfun(@(b)sprintf('%s%s_T', RAW_PREFIX, b(6:end)), ALL_POOL_BRANCHES, 'UniformOutput',false), ...
                         cellfun(@(b)sprintf('meta_%s_T', b(6:end)), ALL_POOL_BRANCHES, 'UniformOutput',false)];
    for v = 1:numel(present_vars)
        vn = present_vars{v};
        matched = any(cellfun(@(p)startsWith(vn, p), expected_prefixes));
        if ~matched
            fprintf('  [??] Unexpected variable: %s\n', vn);
        end
    end

    if file_issues == 0
        fprintf('  >> All variables OK.\n');
    else
        fprintf('  >> %d issue(s) found.\n', file_issues);
        files_with_issues{end+1} = fname; 
    end
    total_issues = total_issues + file_issues;
end

% ── Final summary ────────────────────────────────────────────────────────────
fprintf('\n══════════════════════════════════════════════════════\n');
fprintf('SUMMARY\n');
fprintf('  Files checked  : %d\n', numel(files));
fprintf('  Variables checked: %d\n', total_vars);
fprintf('  Total issues   : %d\n', total_issues);

if isempty(files_with_issues)
    fprintf('  Result: ALL FILES CLEAN.\n');
else
    fprintf('  Files with issues (%d):\n', numel(files_with_issues));
    for i = 1:numel(files_with_issues)
        fprintf('    - %s\n', files_with_issues{i});
    end
end
fprintf('══════════════════════════════════════════════════════\n');