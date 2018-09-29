MATRIX_SIZE=50;
%CONSTANT=5;
NUM_HIDDEN_LAYERS=2;
HIDDEN_LAYER_NODE_NUM=10; % morning, 13th, June,2017: for 196 features ,this may be small
NUM_OUTPUT_NODE=1;
NUM_EXAMPLES=400;

examples_in=zeros(MATRIX_SIZE,MATRIX_SIZE,NUM_EXAMPLES);
for jj=1:NUM_EXAMPLES 
   examples_in(:,:,jj)=diag(rand(1,MATRIX_SIZE))+diag(rand(1,MATRIX_SIZE-1),-1)+diag(rand(1,MATRIX_SIZE-1),1);
end

%examples_in=rand(MATRIX_SIZE,MATRIX_SIZE,NUM_EXAMPLES);
examples_in_reshaped=reshape(examples_in(:),MATRIX_SIZE^2,NUM_EXAMPLES);
examples_out=zeros(NUM_EXAMPLES,1)';
for jj=1:NUM_EXAMPLES
    examples_out(jj)=abs(eigs(examples_in(:,:,jj),1));
end



featureNum=size(examples_in_reshaped,1);
%normalize
% normalizer=1./max(abs(examples_in_reshaped));
% examples_out=examples_out.*repmat(normalizer,NUM_OUTPUT_NODE,1);
% normalizer=repmat(normalizer,featureNum,1);
% examples_in_reshaped=examples_in_reshaped.*normalizer;
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

lambda=10;
options = optimset('MaxIter', 2500);
nn_params=Theta{1}(:);
for kk=2:NUM_HIDDEN_LAYERS+1
   nn_params=[nn_params;Theta{kk}(:)];

end

costFunction = @(p)  MLSolveLinearSysCostFunction(p, ...
                                   featureNum, ...
                                   HIDDEN_LAYER_NODE_NUM, ...
                                   NUM_OUTPUT_NODE,...
                                   NUM_HIDDEN_LAYERS,...
                                   examples_in_reshaped, examples_out, lambda);
fprintf('Cost before training:');
costFunction(nn_params)

[nn_params, cost] = fmincg(costFunction, nn_params, options);
% maxIter=1000;
% timeStep=0.000001;
% for ii=1:maxIter
%     [J grad]= MLSolveLinearSysCostFunction(nn_params, ...
%         featureNum, ...
%         HIDDEN_LAYER_NODE_NUM, ...
%         NUM_OUTPUT_NODE,...
%         NUM_HIDDEN_LAYERS,...
%         examples_in_reshaped, examples_out, lambda);
%     nn_params=nn_params-grad*timeStep;
%     fprintf('Cost : %f\r',J);
%     
% 
% 
% 
% end