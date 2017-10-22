function [ out ] = leaky_relu_prime( x )
if (x <= 0)
    out = 0.1;
else
    out = 1;
end
end

