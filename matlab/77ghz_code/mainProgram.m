close all;
clear;
clc;

%%

subjects = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
times = ["8am", "10am", "12pm", "2pm", "4pm"];
filterEnable = 1;
fs = 500000;
t_s = 512e-6;
B = 2e9;
c = physconst('LightSpeed');
sample_folder = "samples";
r_min = 2.0;
r_max = 4.0;
range_res = 0.3;
Nsamples = 256;
Nchirps = 256;
Nframes = 125;
NRx = 16;
filtersNames = ["BW", "FIR", "FFT"];

%% Filtering

% for i = 1:length(subjects)
%     for j = 1: length(times)
% 
%         file_name = sample_folder + "\subject_" + subjects(i) + "_" + times(j) + ".mat";
%         data = load(file_name).framesRadar;
% 
%         filter_gpt_butterworth77(data, "samples_fil", subjects(i), times(j));
%         % filter_gpt_fir(data, "samples_fil\filteredGPTFIR", subjects(i), times(j));
%         % filter_gpt_fft(data, "samples_fil\filteredGPTFFT", subjects(i), times(j));
% 
%     end
% end

%% Integrity check
% 
% 
% for k = 1: 1
%     for i = 1:length(subjects)
%         for j = 1: length(times)
% 
%             file_name = "samples_fil\subject_" + subjects(i) + "_" + times(j) + "_fil_" + filtersNames(k) + ".mat";
%             data = load(file_name).filtered_data;
% 
%             report = wst_integrity_check_dataset77(data);
%             report_name = "samples_report\report_" + subjects(i) + "_" + times(j) + "_" + filtersNames(k) + ".mat";
%             save(report_name,"report");
% 
%         end
%     end
% end
% 
%% Pre WST

% chirpavg_and_fuse_batch("samples_fil", "samples_prewst");

%% WST Features

counter = 1;

for k = 1: 1
    for i = 1:length(subjects)
        for j = 1: length(times)

            file_name = "samples_prewst\subject_" + subjects(i) + "_" + times(j) + "_prewst.mat";
            data = load(file_name);
            fused_mean = data.fused_mean;
            fused_median = data.fused_median;
            chirpAvg = data.chirpAvg;
            rd_mean = data.rd_mean;
            rd_median = data.rd_median;
            fprintf("(%d/80) ", counter);
            wst_extract77(chirpAvg, fused_mean, fused_median, rd_mean, rd_median, subjects(i), times(j), filtersNames(k));
            counter = counter + 1;
        end
    end
end