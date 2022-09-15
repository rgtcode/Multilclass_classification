function prob=sigmoid_multc(X)
%X is the row vector
tot_prob=0;
for i=1:length(X)
    tot_prob=tot_prob+exp(X(i));
end

for  i=1:length(X)
    prob(i)=exp(X(i))/tot_prob;
end
end