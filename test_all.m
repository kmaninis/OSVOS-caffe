clear; close all; clc;
addpath(genpath('src'));
addpath(genpath('lib'));

cd(osvos_root);

save_dir = '/path/to/save/results';
names = importdata('src/gt_sets/val_categories.txt');

for ii=1:length(names)
    display(['Processing ' num2str(ii)]);
    name = names{ii};
    
    set_params(name);
    
    [pred, vid] = caffe_test(osvos_params);
    if ~exist(fullfile(save_dir,name),'dir')
        mkdir(fullfile(save_dir,name));
    end
    for jj=1:size(pred,3)
        imwrite(pred(:,:,jj), fullfile(save_dir,name, [num2str(jj-1,'%05d') '.png']));
    end
end
