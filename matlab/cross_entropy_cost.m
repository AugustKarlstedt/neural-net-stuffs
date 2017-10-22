function [ out ] = cross_entropy_cost( a, y )
out = nan_to_num(sum(-y .* log(a) - (1 - y) .* log(1 - a)));
end
