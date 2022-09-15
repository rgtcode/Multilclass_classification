function opt_path=mydtw2(a,b)
nc=a(1,1);
n=a(1,2);
m=b(1,2);
dis=zeros(n,m);
tra=zeros(n,nc);
tst=zeros(m,nc);
tra=a(2:n+1,1:nc);
tst=b(2:m+1,1:nc);
dis(1,1)=euclid(tra(1,:),tst(1,:));
for j=2:m
    dis(1,j)=euclid(tra(1,:),tst(j,:))+dis(1,j-1);
end
for i=2:n
    dis(i,1)=euclid(tra(i,:),tst(1,:))+dis(i-1,1);
end
for i=2:n
    for j=2:m
         c= euclid(tra(i,:),tst(j,:));
         a=[dis(i-1,j-1) dis(i,j-1)];
         dis(i,j)=c+min(a);
    end
end
 opt_path=dis(n,m);
end
