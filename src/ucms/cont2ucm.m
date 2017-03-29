function ucm2 = cont2ucm(cont)

% Compute ucms at multiple scales
[ucms_pre] = img2ucm_scale_fast(cont);

% Align ucms
ucm2 = ucms_pre{1};%project_ucms_wrap_fast(ucms_pre, osvos_params.align_thr);



function [ucm2] = img2ucm_scale_fast(I)

% Detect multiscale contours with a single fw pass
[E,O] = cont_and_orient(I);

n_scales = length(E);
ucm2 = cell(n_scales,1);
for s=1:n_scales
    % Continuous oriented watershed transform
    [owt2, superpixels] = contours2OWT(E{s}, O.angle);
    % Ultrametric contour map
    ucm2{s} = double(ucm_mean_pb( (owt2), superpixels) );
end


