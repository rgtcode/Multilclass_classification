function out = myNorm(in)

minVal = min(in(:));
maxVal = max(in(:));

out = (in - minVal)/(maxVal - minVal);

end