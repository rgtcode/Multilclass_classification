function I=I_vec(utter,covariances,f,n,F,N,rank)

%Intialisation of the Total  variability matrix

%t=total variability matrix
[k,dim,~]=size(F);

t = zeros(k*dim,rank);
for i=1:k*dim
    for j=1:rank
        t(i,j)= normrnd(0,1);
    end
end
q=size(t);
%storing the N_s in the form that can be useful for the expectation stage
N_s=n(1,1)*ones(1,dim);
for j=2:k
    a=n(1,j)*ones(1,dim);
    N_s=[N_s a];
end
N_s=diag(N_s);
b=size(N_s);
%concatnating all the means to get the supervector that can be used 
A=f(:,:,1);
R_a=reshape(A,[],1);

%iteration it is required to  for the convergence of the EM algorithm
iter=20;
[Ls Ws]=E_step_I(t,covariances,R_a,N_s,rank);
 for i=1:iter
     [Ls,Ws]=E_step_I(t,covariances,R_a,N_s,rank);
     [T a]=M_step_I(F,N,Ls,Ws);
 end   
 I=Ws;
end