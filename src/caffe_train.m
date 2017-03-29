function caffe_train(osvos_params, num_iters)

solver = caffe.Solver(osvos_params.solver_path);
solver.net.copy_from(osvos_params.train_weights);

num = length(osvos_params.data.im);

for k=1:num_iters
    ind = ceil(num*rand(1,1));
    im = osvos_params.data.im{ind};
    gt = osvos_params.data.gt{ind};
    im = prepare_data(im);
    gt = prepare_gt(gt);
    
    % Reshape the layers and feed the data
    solver.net.blobs('data').reshape([size(im) 1]);
    solver.net.blobs('label').reshape([size(gt) 1 1]);
    solver.net.blobs('data').set_data(im);
    solver.net.blobs('label').set_data(gt);
    solver.net.reshape();
    
    % Perform one forward-backward iteration
    solver.step(1);
end

% Save the result
solver.net.save(fullfile(osvos_root,'models',[osvos_params.name '.caffemodel']));

end


function data = prepare_data(data)

data = single(data);
mval = [122.67892 116.66877 104.00699];
dim = size(data);

% mean substraction
for ch=1:dim(3)
    data(:,:,ch) = data(:,:,ch) - mval(ch);
end

% BGR format
data = data(:,:,[3 2 1]);

% permute width and height (Caffe thing)
data = permute(data, [2, 1, 3]);
end


function gt = prepare_gt(gt)
gt = single(gt);
gt = permute(gt, [2, 1]);
end