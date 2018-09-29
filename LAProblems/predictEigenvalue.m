MATRIX_SIZE=50;
F=diag(3*ones(1,MATRIX_SIZE))+diag(3*ones(1,MATRIX_SIZE-1),-1)+diag(ones(1,MATRIX_SIZE-1),1);
%X=examples_in(:,:,1);
X=rand(MATRIX_SIZE,MATRIX_SIZE);
X=F;
EV=abs(eigs(X,1));

X=X(:);
normm=max(abs(X));
X=X/normm;
EV=EV/normm;
hidden_layer_size=HIDDEN_LAYER_NODE_NUM;
input_layer_size=featureNum;
num_labels=NUM_OUTPUT_NODE;

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
   a{kk}=selu(z{kk});
   a{kk}=[ones(1,m);a{kk}];
end
%Here the function on output layer is 
output=Theta{NUM_HIDDEN_LAYERS+1}*a{NUM_HIDDEN_LAYERS+1};
z{NUM_HIDDEN_LAYERS+2}=output;
a{NUM_HIDDEN_LAYERS+2}=output;

output