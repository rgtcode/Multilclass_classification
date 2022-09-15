function out=fext(a)
dim=a(1,1);
out=zeros(dim+1,6);
b=zeros(dim,6);
c=a(2:end);
x=zeros(dim,1);
y=zeros(dim,1);



%separating the x-coordinate and y-coordinate in separate vector
for i=2:dim
    x(i,1)=c(2*i-1);
    y(i,1)=c(2*i);
end

%extracting all the features
v=Get6DFeature(x',y');
out(2:end,1:end)=v;
out(1,1)=6;
out(1,2)=a(1,1);

end
