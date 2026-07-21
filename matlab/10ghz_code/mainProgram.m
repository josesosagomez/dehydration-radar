close all;
clear;
clc;

%%

subjects = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
times = ["8am", "10am", "12pm", "2pm", "4pm"];
filterEnable = 1;
fs = 520834;
t_s = 1024e-6;
B = 500e6;
c = physconst('LightSpeed');
sample_folder = "samples";
r_min = 0.9;
r_max = 3.0;
range_res = 0.3;
Nsamples = 534;
Nchirps = 20;
Nframes = 100;

%% Filtering

% for i = 1:length(subjects)
%     for j = 1: length(times)
% 
%         file_name = sample_folder + "\subject" + subjects(i) + "_" + times(j) + "_ov_1m.mat";
%         data = load(file_name).framesRadar;
% 
%         filter_gpt_butterworth(data, "samples_fil", subjects(i), times(j));
%         % filter_gpt_fir(data, "samples_fil\filteredGPTFIR", subjects(i), times(j));
%         % filter_gpt_fft(data, "samples_fil\filteredGPTFFT", subjects(i), times(j));
% 
%     end
% end

%% Integrity check

filtersNames = ["BW", "FIR", "FFT"];
% 
% for k = 1: 1
%     for i = 1:length(subjects)
%         for j = 1: length(times)
% 
%             file_name = "samples_fil\subject_" + subjects(i) + "_" + times(j) + "_fil_" + filtersNames(k) + ".mat";
%             data = load(file_name).filtered_data;
% 
%             report = wst_integrity_check_dataset(data);
%             report_name = "samples_report\report_" + subjects(i) + "_" + times(j) + "_" + filtersNames(k) + ".mat";
%             save(report_name,"report");
% 
%         end
%     end
% end

%% WST Features

% for k = 1: 1
%     for i = 1:length(subjects)
%         for j = 1: length(times)
% 
%             file_name = "samples_fil\subject_" + subjects(i) + "_" + times(j) + "_fil_" + filtersNames(k) + ".mat";
%             data = load(file_name).filtered_data;
% 
%             wst_extract(data, subjects(i), times(j), filtersNames(k));
% 
%         end
%     end
% end