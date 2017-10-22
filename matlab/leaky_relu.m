function [ out ] = leaky_relu( x )
out = max(0.1 * x, x);
end

