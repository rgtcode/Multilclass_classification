function [i_u a]=M_step_I(F,N,ls,ws)
epsi=0.001;
[k,~,f_l]=size(F);
c_i=F(1,:,1)'*ws';
a_i=N(1,1)*(inv(ls)+ws*ws');
for i=2:f_l
    b_i=F(1,:,i)'*ws';
    c_i=c_i+b_i;
    d_i=N(i,1)*(inv(ls)+ws*ws');
    a_i=a_i+d_i;
end
i_u=c_i*inv(a_i);
for i=2:k
    c_i=F(k,:,1)'*ws';
    a_i=N(1,i)*(inv(ls)+ws*ws');
    for j=2:f_l
        b_i=F(k,:,j)'*ws';
        c_i=c_i+b_i;
        d_i=N(j,i)*(inv(ls)+ws*ws');
        a_i=a_i+d_i;
    end
 a=c_i*inv(a_i);
 i_u=[i_u;a];
 a=size(i_u);
end
end

    
    
