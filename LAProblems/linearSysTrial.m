MATRIX_SIZE=25;

NUM_HIDDEN_LAYERS=2;
HIDDEN_LAYER_NODE_NUM=20; % morning, 13th, June,2017: for 196 features ,this may be small
NUM_OUTPUT_NODE=MATRIX_SIZE;
NUM_EXAMPLES=350;

A=diag(2*ones(1,MATRIX_SIZE))+diag(-ones(1,MATRIX_SIZE-1),-1)+diag(-ones(1,MATRIX_SIZE-1),1);

% x=rand(MATRIX_SIZE,1);
% 
% examples_out=zeros(MATRIX_SIZE,NUM_EXAMPLES);
% examples_in=zeros(MATRIX_SIZE,NUM_EXAMPLES);
% temp=x;
% for jj=1:MATRIX_SIZE
%    examples_out(:,jj)=temp;
%    examples_in(:,jj)=A*temp;
%    temp=A*temp;
% end

examples_out=rand(MATRIX_SIZE,NUM_EXAMPLES);
examples_in=A*examples_out;
featureNum=size(examples_in,1);
%normalize
normalizer=1./max(abs(examples_in));
examples_out=examples_out.*repmat(normalizer,NUM_OUTPUT_NODE,1);
normalizer=repmat(normalizer,featureNum,1);
examples_in=examples_in.*normalizer;
%initialize
epsilon=ones(1,NUM_HIDDEN_LAYERS+1);
epsilon=sqrt(6)/sqrt(HIDDEN_LAYER_NODE_NUM*2+1)*epsilon;
epsilon(1)=sqrt(6)/sqrt(featureNum+HIDDEN_LAYER_NODE_NUM+1);
epsilon(end)=sqrt(6)/sqrt(HIDDEN_LAYER_NODE_NUM+NUM_OUTPUT_NODE+1);

Theta={rand(HIDDEN_LAYER_NODE_NUM,featureNum+1)*2*epsilon(1)-epsilon(1)};
for kk=2:NUM_HIDDEN_LAYERS
   Theta{kk}=rand(HIDDEN_LAYER_NODE_NUM,HIDDEN_LAYER_NODE_NUM+1)*2*epsilon(kk)-epsilon(kk);
end
Theta{NUM_HIDDEN_LAYERS+1}=rand(NUM_OUTPUT_NODE,HIDDEN_LAYER_NODE_NUM+1)*2*epsilon(end)-epsilon(end);

lambda=0.05;
options = optimset('MaxIter', 1500);
nn_params=Theta{1}(:);
for kk=2:NUM_HIDDEN_LAYERS+1
   nn_params=[nn_params;Theta{kk}(:)];

end

costFunction = @(p) MLSolveLinearSysCostFunction(p, ...
                                   featureNum, ...
                                   HIDDEN_LAYER_NODE_NUM, ...
                                   NUM_OUTPUT_NODE,...
                                   NUM_HIDDEN_LAYERS,...
                                   examples_in, examples_out, lambda);
fprintf('Cost before training:');
costFunction(nn_params)

% maxIter=500;
% timeStep=0.0002;
% for ii=1:maxIter
%     [J grad]=MLSolveLinearSysCostFunction(nn_params, ...
%         featureNum, ...
%         HIDDEN_LAYER_NODE_NUM, ...
%         NUM_OUTPUT_NODE,...
%         NUM_HIDDEN_LAYERS,...
%         examples_in, examples_out, lambda);
%     nn_params=nn_params-grad*timeStep;
%    fprintf('Cost : %f\r',J);
%     
% 
% 
% 
% end
[nn_params, cost] = fmincg(costFunction, nn_params, options);