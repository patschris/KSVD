%% fMRI & k-SVD test file
% This file automatically calculates the speedup,
% the speedup percentage as well as the difference in
% the final error between CPU and GPU execution of the
% k-SVD algorithm.
%
%% Initializations %%
% load data
load('300_3600.mat');
clearvars -except X

% FMRI data dimensions
% Rows(data) = number_of_time_components
% Columns(data) = number_of_signals
params.data = X;

% Dictionary dimensions
% Rows(Dictionary) = Rows(data)
% Columns(Dictionary) = number_of_sources
sources = 9;
D = randn(size(X,1),sources);

% Sparsity level target
k = 6;

%% Run k-SVD training %%
disp('============================================');
disp('================= MATLAB ===================');
disp('============================================');

params.data = X;
params.Tdata = k;
params.dictsize = sources;
params.iternum = 30;
params.muthresh = 0.8;
params.memusage = 'high';
params.codemode = 'sparsity';
params.initdict = D;

tic;
[Dksvd,g,err] = ksvd(params,'it');
time = toc;
fprintf('\n>Time for k-SVD training in CPU: %.6f\n',time);


%% Write input to folder that CUDA project will use %%
save('input\X.mat','X');
save('input\D.mat','D');

%% Run CUDA
disp('============================================');
disp('================== CUDA ====================');
disp('============================================');
path = 'C:\Users\Thesis\Dropbox\Visual Studio 2015\Projects\K-SVD\x64\Release\';
[status,cmdout] = system(strcat(path,'K-SVD.exe'));
disp(cmdout);

%% Display statistics

disp('============================================');
disp('================ RESULTS ===================');
disp('============================================');

k = strfind(cmdout,'RMSE = ');
cuda_err = str2double(cmdout(k(end) + 7:k(end) + 14));
fprintf('>Final errors\n  MATLAB = %.5f, CUDA = %.5f\n',err(end),cuda_err);
fprintf('>Difference = %.5f\n',abs(err(end) - cuda_err));
k = strfind(cmdout,'(');
speedup = time / str2double(cmdout(k(1) + 1:k(1) + 5));
fprintf('>Speedup = %.2f (%.2f %%)\n\n', speedup, (1 - 1/speedup)*100);

%% END
