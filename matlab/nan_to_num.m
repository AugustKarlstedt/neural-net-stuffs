function [ out ] = nan_to_num( x )
    if (isinf(x))
        if (sign(x) == 1)
            out = realmax;
        elseif (sign(x) == -1)
            out = -realmax;
        end
        
        return;
    end

    if (isnan(x))
        out = 0;
        return;
    end

    out = x;
end

