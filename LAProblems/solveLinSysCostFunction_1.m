function [ J,grad ] = solveLinSysCostFunction_1(nn_params, ...                                 
                                   MATRIX_SIZE,...
                                   X, Y, A,lambda)

Theta=reshape(nn_params,MATRIX_SIZE,MATRIX_SIZE);
        
m=size(X,2);


output=Theta*X;
diff=output-Y;
%diff=A*output-X;
J=sum(sum(diff.^2))/2;



%regularization:
%correction=0;
% for kk=1:NUM_HIDDEN_LAYERS+1
%     correction=correction+sum(sum(Theta{kk}(:,2:end).^2));
% end
% correction=lambda/2*correction;



%correction=lambda/2*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
%J=J+correction;
% then do the backpropogation  

%grad=A'*(A*Theta*X-X)*X';
grad=diff*X';
    
%regularization
% tempMatrix=zeros(size(grad));
% tempMatrix(:,2:end)=Theta(:,2:end);
% grad=grad+lambda*tempMatrix;
% 

grad=grad(:);

             
%J=1/m*J;
%grad=1/m*grad;
             
             

end