clear 
close all


load text1_uni.mat
X=mapstd(X);  %可调整为X=mapstd(X')
y=double(Y);

%% Initialization

maxIter = 5;
c = length(unique(Y));                      % number of cluster
[n,d] = size(X);                         % number of samples
X=X';
lambdalist=[1e-2];           % 可以调整
detalist=[1e2];              % 可以调整 
RESULT=[];T=[];OBJ=[]; 
for JJ = 1:1
for III=1:size(lambdalist,2)
     lambda=lambdalist(III);
     for IIII=1:size(detalist,2)
     detamul=detalist(IIII);
tic
%% 初始化F

F=orth(randn(n,c));
G=orth(randn(n,c));

%% 初始化高斯核带宽
temp = (X-X*F*G').^2;
deta =detamul*sqrt(sum(sum(temp)/(2*n))); 
OBJJ=[];
%% Optimization
for Iter = 1:maxIter

    % Update W

    temp1 = (X-X*F*G').^2;
    temp2 = (sum(temp1,2))./(2*deta^2);
    W = sparse(diag(exp(-temp2)./(deta^2)));
%      W = diag(exp(-temp2)./(deta^2));


    %update G

    A = X'*W*X*F;
    [AA,~,CC] = svd(A,'econ');
    G = AA*CC;        

    
    % update F
    A = 0;    
    U = X'*W*X-2*lambda*X'*X;
    H = X'*W*X*G;
    QQ= 2*U*F-2*H;
    [DD,~,EE]=svd(QQ,'econ');
    F = DD*EE;
  
    %objective value
    
    obj = trace((X-X*F*G')'*W*(X-X*F*G'))+lambda*trace((X'*X-F*F')'*(X'*X-F*F'));
    OBJJ=[OBJJ, obj];

end


[maxv,ind]=max(G,[],2);
t=toc;
Result = ClusteringMeasure(y, ind);
% Result=[lambda,Result,t]
Result=[Result,t]
RESULT=[RESULT;Result];T=[T,t];
% end
end
end
end
record=[mean(RESULT(:,1)),std(RESULT(:,1));
 mean(RESULT(:,2)),std(RESULT(:,2));
 mean(RESULT(:,3)),std(RESULT(:,3));
 mean(RESULT(:,4)),std(RESULT(:,4));
 mean(RESULT(:,5)),std(RESULT(:,5));
 mean(RESULT(:,6)),std(RESULT(:,6));
 mean(RESULT(:,7)),std(RESULT(:,7));
 mean(T),std(T)];
record = record'


