function [ total_weights ] = count_weights_from_vector( nodeLayers )
%count_weights_from_vector SUMMARY
%   Note: we assume all layers are fully connected (e.g. no dropout)
bias_weights = 0;
total_weights = 0;
for idx = 1:length(nodeLayers)-1
    element = nodeLayers(idx);
    next_element = nodeLayers(idx+1);
    layer_weight_count = element * next_element;
    bias_weights = bias_weights + next_element;
    total_weights = total_weights + layer_weight_count;
end
total_weights = total_weights + bias_weights;
end




