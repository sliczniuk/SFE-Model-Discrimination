function [r] = corr2(a,b)
    
    a = a - mean(a);
    b = b - mean(b);
    r = sum(sum(a.*b))/sqrt(sum(sum(a.*a))*sum(sum(b.*b)));

end