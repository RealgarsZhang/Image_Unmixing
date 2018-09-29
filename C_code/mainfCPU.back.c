#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> 
          
#include "mkl.h"
#include "mkl_lapacke.h"         

#include "NoNoiseNoPrintCPU.h"
int main(){

    int TRAINING_SIZE = 200;
    int CV_SIZE = 20;
    int TEST_SIZE = 20;
    double RANDOM_MIN = -1;
    double RANDOM_MAX =  1;
    int NUM_HIDDEN_LAYERS = 2;
    int HIDDEN_LAYER_NODE_NUM = 15; 
    int NUM_OUTPUT_NODE = 3;
    int featureNum = 196 ; 
    
    unsigned int maxIter;
    int length;
    int index;
    int nn_paramsLength;
    MATRIX *Theta;
    MATRIX coefMatrix_train, coefMatrix_cv, coefMatrix_test;
    MATRIX examples_train,   examples_cv,   examples_test;
    double *epsilon, *nn_params, *grad;
    double lambda, J; 
    int i,j;
    i=readinExamples(&coefMatrix_train, 
                     &coefMatrix_cv,
		     &coefMatrix_test,
		     &examples_train,
		     &examples_cv,
		     &examples_test);
    
    //get epsilon and nn_paramsLength
    epsilon = (double*) malloc(sizeof(double)*(NUM_HIDDEN_LAYERS + 1));
    nn_paramsLength = 0;


    epsilon[0] = sqrt( 6.0 ) / sqrt( featureNum + HIDDEN_LAYER_NODE_NUM+1 );
    nn_paramsLength += (featureNum+1) * (HIDDEN_LAYER_NODE_NUM);
    
    for( i=1; i<NUM_HIDDEN_LAYERS; i++){
        epsilon[i] = sqrt( 6.0 ) / sqrt( HIDDEN_LAYER_NODE_NUM*2+1);
	nn_paramsLength += (HIDDEN_LAYER_NODE_NUM + 1)*
	                    HIDDEN_LAYER_NODE_NUM;  
    }
    
    epsilon[NUM_HIDDEN_LAYERS] = sqrt( 6.0 ) / 
                             sqrt(HIDDEN_LAYER_NODE_NUM+NUM_OUTPUT_NODE+1);
    nn_paramsLength += (HIDDEN_LAYER_NODE_NUM + 1) *
                       NUM_OUTPUT_NODE;
    
    //I will first malloc for nn_params, a linear array.
    //Then the pointers in Theta will point to corresponding address.
    nn_params = (double*) malloc(sizeof(double) * nn_paramsLength);    
    Theta     = (MATRIX*) malloc(sizeof(MATRIX) * (NUM_HIDDEN_LAYERS + 1));
    
    
    Theta[0].col = featureNum + 1;
    Theta[0].row = HIDDEN_LAYER_NODE_NUM;
    Theta[0].data= &nn_params[0];
    index = (featureNum+1) * HIDDEN_LAYER_NODE_NUM ;
    
    for( i=1; i<NUM_HIDDEN_LAYERS; i++){ 
        Theta[i].col = HIDDEN_LAYER_NODE_NUM + 1;
	Theta[i].row = HIDDEN_LAYER_NODE_NUM;
        Theta[i].data= &nn_params[index];
	index += (HIDDEN_LAYER_NODE_NUM + 1)*
	          HIDDEN_LAYER_NODE_NUM;  
    }
    
    Theta [NUM_HIDDEN_LAYERS].col = HIDDEN_LAYER_NODE_NUM;
    Theta [NUM_HIDDEN_LAYERS].row = NUM_OUTPUT_NODE;
    Theta [NUM_HIDDEN_LAYERS].data= &nn_params[index];
    
    //Initialize
    for( i=0; i<NUM_HIDDEN_LAYERS+1; i++){
        length = Theta[i].col * Theta[i].row;
	for( j=0; j<length; j++){
	    Theta[i].data[j] = random_generate( -epsilon[i] , epsilon[i] ); 
	}
    }
    
    lambda = 0.01;
    maxIter= 1500;
    grad = (double*) malloc(sizeof(double)*nn_paramsLength);
    
    J = costFunction( grad,
                      nn_params,
		      featureNum,
		      HIDDEN_LAYER_NODE_NUM,
		      NUM_OUTPUT_NODE,
		      NUM_HIDDEN_LAYERS,
		      examples_train,
		      coefMatrix_train,
		      lambda);


    printf( "%lf\n", J);
    //free everything
    free(epsilon);
    free(Theta);
    free(grad);
    free(nn_params);
    free(coefMatrix_train.data);
    free(coefMatrix_cv.data);
    free(coefMatrix_test.data);
    free(examples_train.data);
    free(examples_cv.data);
    free(examples_test.data);
    return 0;
}
