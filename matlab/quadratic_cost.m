function [ out ] = quadratic_cost( a, y )
out = 0.5 * sum(norm(y-a) .^ 2);
end

