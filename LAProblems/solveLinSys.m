clc;
clear all;
MATRIX_SIZE=30;
COND_NUMBER=200;
MAX_ITER=200;

HIDDEN_LAYER_NODE_NUM=20; 
NUM_OUTPUT_NODE=MATRIX_SIZE;
NUM_EXAMPLES=MATRIX_SIZE+40;

%A=diag(2*ones(1,MATRIX_SIZE))+diag(-ones(1,MATRIX_SIZE-1),-1)+diag(ones(1,MATRIX_SIZE-1),1);
%A=sprand(MATRIX_SIZE,MATRIX_SIZE,3*1/MATRIX_SIZE);
%A=rand(MATRIX_SIZE,MATRIX_SIZE);

A=2*rand(MATRIX_SIZE,MATRIX_SIZE)-ones(MATRIX_SIZE,MATRIX_SIZE);

[U,S,V]=svd(A);
S=diag(linspace(1,COND_NUMBER,MATRIX_SIZE));
A=U*S*V';

% 
% examples_out=zeros(MATRIX_SIZE,NUM_EXAMPLES);
% examples_in=zeros(MATRIX_SIZE,NUM_EXAMPLES);
% temp=x;
% for jj=1:MATRIX_SIZE
%    examples_out(:,jj)=temp;
%    examples_in(:,jj)=A*temp;
%    temp=A*temp;
% end

 examples_out=2*rand(MATRIX_SIZE,NUM_EXAMPLES)-ones(MATRIX_SIZE,NUM_EXAMPLES);
% examples_in=A*examples_out;
examples_in=2*rand(MATRIX_SIZE,NUM_EXAMPLES)-1*ones(MATRIX_SIZE,NUM_EXAMPLES);
featureNum=size(examples_in,1);
%normalize
normalizer=1./max(abs(examples_in));
examples_out=examples_out.*repmat(normalizer,NUM_OUTPUT_NODE,1);
normalizer=repmat(normalizer,featureNum,1);
examples_in=examples_in.*normalizer;

epsilon=sqrt(6)/sqrt(2*MATRIX_SIZE);

Theta=2*epsilon*rand(MATRIX_SIZE,MATRIX_SIZE)-epsilon*rand(MATRIX_SIZE,MATRIX_SIZE);

nn_params=Theta(:);
lambda=0;
options = optimset('MaxIter', MAX_ITER);

costFunction = @(p) solveLinSysCostFunction(p, ...
                                    MATRIX_SIZE,...
                                   examples_in, examples_out, A,lambda)
fprintf('Cost before training:');
costFunction(nn_params)

B=examples_in*examples_in';
maxIter=1000;
timeStep=0.00001;
for ii=1:maxIter
   [J grad]=solveLinSysCostFunction(nn_params, ...
                                    MATRIX_SIZE,...
                                   examples_in, examples_out, A,lambda);
     Theta=reshape(nn_params,MATRIX_SIZE,MATRIX_SIZE);
     numerator=sum(diag(examples_in'*(1-Theta'*A')*A*A'*(1-A*Theta)*B*examples_in));
     denominator=sum(diag(examples_in'*B'*(1-Theta'*A')*A*A'*A*A'*(1-A*Theta)*B*examples_in));
     
     timeStep=0.5*numerator/denominator;
   nn_params=nn_params-timeStep*grad;
   fprintf('Cost : %f\r',J);
    



end

[nn_params, cost] = fmincg(costFunction, nn_params, options);

Theta=reshape(nn_params,MATRIX_SIZE,MATRIX_SIZE);