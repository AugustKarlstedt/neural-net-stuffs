function [ out ] = relu_prime( n )
if (n <= 0)
    out = 0;
else
    out = 1;
end
end

