function osvos_params = set_params(name)

if evalin('base','exist(''osvos_params'',''var'')')
    if evalin('base','strcmp(osvos_params.name,name)')
        % Recover the parameters
        osvos_params = evalin('base', 'osvos_params');
        return;
    end
end

disp('Setting OSVOS parameters')

%% ATTENTION: Parameters to be modifed by the user

% Specify /path/to/caffe/matlab (no need to change if didn't move caffe-osvos)
osvos_params.caffe_path = './caffe-osvos/matlab/';

% For CPU mode, set to 0 (WARNING: 0 is slow)
osvos_params.useGPU = 1;

% Set the ID of your GPU (default 0)
osvos_params.gpu_id = 0;

% Specify if you want to use the second contour branch
useCont = 0;

% Train online or use pre-trained model?(1=Yes,0=No)
osvos_params.useTrainOnline = 1;

%% Other parameters
osvos_params.name = name;

if ~exist(osvos_params.caffe_path,'dir')
    error(['Caffe path ''' osvos_params.caffe_path ''' not found'])
end

addpath(osvos_params.caffe_path);
caffe.reset_all();

% Network  test model and weights
osvos_params.test_model = fullfile(osvos_root,'models','deploy.prototxt');
osvos_params.test_weights = fullfile(osvos_root,'models',[name '.caffemodel']);

% Network train model and weights
osvos_params.solver_path = fullfile(osvos_root,'models','solvers','solver_online.prototxt');
osvos_params.train_weights = fullfile(osvos_root,'models','OSVOS_parent.caffemodel');

% Load the two-branch network for using the contour model
if useCont
    osvos_params.test_model = fullfile(osvos_root,'models','deploy_cont.prototxt');
    osvos_params.cont_weights = fullfile(osvos_root,'models','OSVOS_contour.caffemodel');
end


% Prepare the data if training online
if osvos_params.useTrainOnline
    im = im2double(imread(fullfile(osvos_root, 'DAVIS', 'JPEGImages', '480p', name, '00000.jpg')));
    gt = im2double(imread(fullfile(osvos_root, 'DAVIS', 'Annotations', '480p', name, '00000.png')));
    osvos_params.data = prepare_data(im, gt, name);
end

% Copy to base workspace
assignin('base', 'osvos_params', osvos_params);

% For GPU users
if osvos_params.useGPU
    evalin('base', 'caffe.set_mode_gpu()');
    evalin('base', 'caffe.set_device(osvos_params.gpu_id)')
end

end

function data = prepare_data(im, gt, name)
scales = [0.5 0.8 1];
k=1;
for jj=1:length(scales)
    % scale the images
    im_sc = imresize(im, scales(jj)); data.im{k}=single(255*im_sc);
    gt_sc = imresize(gt, scales(jj)); data.gt{k}=single(gt_sc); k=k+1;
    
    % flip the scaled images
    im_sc_fl = fliplr(im_sc); data.im{k} = single(255*im_sc_fl);
    gt_sc_fl = fliplr(gt_sc); data.gt{k} = single(gt_sc_fl); k=k+1;
    
    % save the data (useful for the python interface)
    save_aug_data(name, im_sc, gt_sc, im_sc_fl, gt_sc_fl, scales(jj));
end

% Prepare data (useful for the python interface)
fid = fopen(fullfile(osvos_root, 'aug_data', name,'train_pairs.txt'), 'w');
for jj=1:length(scales)
    fprintf(fid,[fullfile('im',['im_' num2str(scales(jj)) '_0.png'])...
        ' ' fullfile('gt',['gt_' num2str(scales(jj)) '_0.png\n'])]);
    fprintf(fid,[fullfile('im',['im_' num2str(scales(jj)) '_1.png'])...
        ' ' fullfile('gt',['gt_' num2str(scales(jj)) '_1.png\n'])]);
end
end

function save_aug_data(name, im, gt, im_fl, gt_fl, scale)
save_dir = fullfile(osvos_root, 'aug_data', name);
if ~exist(save_dir, 'dir')
    mkdir(fullfile(save_dir, 'im'));
    mkdir(fullfile(save_dir, 'gt'));
end
imwrite(im, fullfile(save_dir, 'im', ['im_' num2str(scale) '_0.png']));
imwrite(im_fl, fullfile(save_dir, 'im', ['im_' num2str(scale) '_1.png']));
imwrite(gt, fullfile(save_dir, 'gt', ['gt_' num2str(scale) '_0.png']));
imwrite(gt_fl, fullfile(save_dir, 'gt', ['gt_' num2str(scale) '_1.png']));
end

