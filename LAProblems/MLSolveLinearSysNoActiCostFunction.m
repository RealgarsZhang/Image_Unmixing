function [ J,grad ] = MLSolveLinearSysNoActiCostFunction(nn_params, ...                                 
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   NUM_HIDDEN_LAYERS,...
                                   X, Y, lambda)
%UNTITLED2 Summary of this function goes here
%   This function is only suitable for the simple linear case of one
%   hidden layer and 14x14 training example case.
Theta = {reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, input_layer_size+1 )};
currentProg=hidden_layer_size * (input_layer_size + 1);
for kk=2:NUM_HIDDEN_LAYERS
    Theta{kk}=reshape(nn_params(currentProg+1:currentProg+hidden_layer_size*(hidden_layer_size+1)),...
                      hidden_layer_size,hidden_layer_size+1);
    currentProg=currentProg+hidden_layer_size*(hidden_layer_size+1);

end
Theta{NUM_HIDDEN_LAYERS+1}=reshape(nn_params(currentProg+1:end),...
                      num_labels,hidden_layer_size+1);


        
m=size(X,2);

a={[ones(1,m);X]};
z=a;
for kk=2:NUM_HIDDEN_LAYERS+1
   z{kk}=Theta{kk-1}*a{kk-1};
   a{kk}=(z{kk});
   a{kk}=[ones(1,m);a{kk}];
end
%Here the function on output layer is 
output=Theta{NUM_HIDDEN_LAYERS+1}*a{NUM_HIDDEN_LAYERS+1};
z{NUM_HIDDEN_LAYERS+2}=output;
a{NUM_HIDDEN_LAYERS+2}=output;

diff=output-Y;
%using 2-norm for distance

J=sum(sum(diff.^2))/2;


%regularization:
correction=0;
for kk=1:NUM_HIDDEN_LAYERS+1
    correction=correction+sum(sum(Theta{kk}(:,2:end).^2));
end
correction=lambda/2*correction;



%correction=lambda/2*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
J=J+correction;
% then do the backpropogation  
delta(1:NUM_HIDDEN_LAYERS+2)={1};
Theta_grad(1:NUM_HIDDEN_LAYERS+1)={1};
delta{end}=output-Y;
%delta{end}=delta{end}(2:end,:);
Theta_grad{end}=delta{end}*a{NUM_HIDDEN_LAYERS+1}';



for kk=1:NUM_HIDDEN_LAYERS
    tempkk=NUM_HIDDEN_LAYERS-kk+1;
    
    %delta{tempkk+1}=Theta{tempkk+1}'*delta{tempkk+2}.*diffSelu([zeros(1,m);z{tempkk+1}]);%(1-a{tempkk+1}.^2);
    delta{tempkk+1}=Theta{tempkk+1}'*delta{tempkk+2};%.*diffRelu([zeros(1,m);z{tempkk+1}]);%(1-a{tempkk+1}.^2);
    %delta{tempkk+1}=Theta{tempkk+1}'*delta{tempkk+2}.*(1-a{tempkk+1}.^2);
    %delta{tempkk+1}=Theta{tempkk+1}'*delta{tempkk+2}.*(1-a{tempkk+1}).*a{tempkk+1};
    delta{tempkk+1}=delta{tempkk+1}(2:end,:);
    Theta_grad{tempkk}=delta{tempkk+1}*a{tempkk}';
    
    %regularization
    tempMatrix=zeros(size(Theta_grad{tempkk}));
    tempMatrix(:,2:end)=Theta{tempkk}(:,2:end);
    Theta_grad{tempkk}=Theta_grad{tempkk}+lambda*tempMatrix;
end









grad=[Theta_grad{1}(:)];
for kk=2:NUM_HIDDEN_LAYERS+1
    grad=[grad;Theta_grad{kk}(:)];
end
    


             
%J=1/m*J;
%grad=1/m*grad;
             
             

end