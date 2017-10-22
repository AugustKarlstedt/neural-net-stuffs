function [ out ] = cross_entropy_cost( a, y )
out = sum(nan_to_num(-y .* log(a) - (1 - y) .* log(1 - a)));
end
