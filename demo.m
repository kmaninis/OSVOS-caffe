clear; close all; clc;
addpath(genpath('src'));
addpath(genpath('lib'));
cd(osvos_root);

% Set name and number of online training iters
name = 'drift-chicane';
num_iters = 500;

% Set the OSVOS parameters.
set_params(name);

% Train the model
if osvos_params.useTrainOnline
    caffe_train(osvos_params, num_iters);
end


% Test the model
[pred, vid] = caffe_test(osvos_params);


% Vizualize the results
for ii=1:size(pred,3)
    vid{ii}(:,:,2) = min(255, vid{ii}(:,:,2) + 100*uint8(pred(:,:,ii)>0.757));
    imshow(vid{ii});
    pause(0.01);
end
