function out = Get6DFeature(x,y)


% Use only x,y cooridates
% out = [x;y]';
% return;

% Feature extraction based on following paper
% Writing Speed Normalization for On-Line Handwritten Text Recognition
eps = 1e-6;

% 1: Normalized vertical position
yN = myNorm(y);

% 2,3: First derivatives
ws = 6;
den = 0;
for i = 1:ws
   x_tm(i,:) = [zeros(1,i) x(1:end-i)];
   x_tp(i,:) = [x(1+i:end) zeros(1,i)]; 
   y_tm(i,:) = [zeros(1,i) y(1:end-i)];
   y_tp(i,:) = [y(1+i:end) zeros(1,i)];
   den = den + i*i;
end

% 3, 4: Second Derivatives
xt_dash = zeros(size(x));
yt_dash = zeros(size(x));
for i = 1:ws
    xt_dash = xt_dash + (i * (x_tp(i,:) - x_tm(i,:)))/den;
    yt_dash = yt_dash + (i * (y_tp(i,:) - y_tm(i,:)))/den;    
end

for i = 1:ws
   xt_dash_tm(i,:) = [zeros(1,i) xt_dash(1:end-i)];
   xt_dash_tp(i,:) = [xt_dash(1+i:end) zeros(1,i)]; 
   yt_dash_tm(i,:) = [zeros(1,i) yt_dash(1:end-i)];
   yt_dash_tp(i,:) = [yt_dash(1+i:end) zeros(1,i)];
   den = den + i*i;
end
xt_ddash = zeros(size(x));
yt_ddash = zeros(size(x));
for i = 1:ws
    xt_ddash = xt_ddash + (i * (xt_dash_tp(i,:) - xt_dash_tm(i,:)))/den;
    yt_ddash = yt_ddash + (i * (yt_dash_tp(i,:) - yt_dash_tm(i,:)))/den;    
end

% Curvature
den = xt_dash.*xt_dash + yt_dash.*yt_dash;
den = den .^ (3/2);
kt = xt_dash .* yt_ddash - xt_ddash .* yt_dash;
% kt = kt ./ (eps + den);
kt = kt ./den;
kt(isinf(kt)) = 0;
kt(isnan(kt)) = 0;

% Final Feature Vector
% Normalize all the features between 0 and 1
xN = myNorm(x);
yN = myNorm(y);
xt_dash = myNorm(xt_dash);
yt_dash = myNorm(yt_dash);
xt_ddash = myNorm(xt_ddash);
yt_ddash = myNorm(yt_ddash);
kt = myNorm(kt);
out = [ yN; xt_dash; yt_dash; xt_ddash; yt_ddash; kt];
out = out';

end