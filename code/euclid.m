function dist=euclid(a,b)
count=0;
for i=1:length(a)
    c=(a(1,i)-b(1,i))^2;
    count=count+c;
end
dist=sqrt(count);
  
end