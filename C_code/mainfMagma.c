#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> 
#include <cuda.h>
#include <time.h>
//#include "testings.h"
          
#include "magma.h"
#include "magma_lapack.h"         
#include "mkl.h"
#include "mkl_lapacke.h"
#include "NoNoiseNoPrintMagma.h"
int main(){
    printf("On GPU:\n");
    magma_init();
    int TRAINING_SIZE = 100;
    int CV_SIZE = 20;
    int TEST_SIZE = 20;
    double RANDOM_MIN = -1;
    double RANDOM_MAX =  1;
    int NUM_HIDDEN_LAYERS = 2;
    int HIDDEN_LAYER_NODE_NUM = 15; 
    int NUM_OUTPUT_NODE = 3;
    int featureNum = 196 ; 
    double lambda = 0.05;
    unsigned int maxIter= 2500;
    //clock_t start, diff;


    int m, n;
    int length;
    int index;
    int mallocSize;
    int nn_paramsLength;
    MATRIX *Theta;  //It seems that, essentially, Theta is only used for initial
    MATRIX coefMatrix_train, coefMatrix_cv, coefMatrix_test;
    MATRIX examples_train,   examples_cv,   examples_test;
    MATRIX d_coefMatrix_train, d_coefMatrix_cv, d_coefMatrix_test;
    MATRIX d_examples_train,   d_examples_cv,   d_examples_test;
    MATRIX *d_a, *a; 
    MATRIX *d_delta,*d_Theta,*d_Theta_grad;
    double *epsilon, *nn_params, *grad;
    double *d_nn_params, *d_grad;
    double *p1, *p2,*d_p1,*d_p2,*p0;
    double  J, J1, J2, ep;
    double correction; 
    int i,j;
    int gradientCheckIndex;
    i=readinExamples(&coefMatrix_train, 
                     &coefMatrix_cv,
		     &coefMatrix_test,
		     &examples_train,
		     &examples_cv,
		     &examples_test);
    m = coefMatrix_train.row;
    n = coefMatrix_train.col;
    d_coefMatrix_train.row = coefMatrix_train.row;
    d_coefMatrix_train.col = coefMatrix_train.col;
    magma_dmalloc(  &(d_coefMatrix_train.data), m*n );
    magma_dsetmatrix ( m , n , coefMatrix_train.data , 
                       m , d_coefMatrix_train.data , m);
   
    m = coefMatrix_cv.row;
    n = coefMatrix_cv.col;
    d_coefMatrix_cv.row = coefMatrix_cv.row;
    d_coefMatrix_cv.col = coefMatrix_cv.col;
    magma_dmalloc(  &(d_coefMatrix_cv.data), m*n );
    magma_dsetmatrix ( m , n , coefMatrix_cv.data , 
                       m , d_coefMatrix_cv.data , m);
    
    m = coefMatrix_test.row;
    n = coefMatrix_test.col;
    d_coefMatrix_test.row = coefMatrix_test.row;
    d_coefMatrix_test.col = coefMatrix_test.col;
    magma_dmalloc(  &(d_coefMatrix_test.data), m*n );
    magma_dsetmatrix ( m , n , coefMatrix_test.data , 
                       m , d_coefMatrix_test.data , m);
 
    m = examples_train.row;
    n = examples_train.col;
    d_examples_train.row = examples_train.row;
    d_examples_train.col = examples_train.col;
    magma_dmalloc(  &(d_examples_train.data), m*n );
    magma_dsetmatrix ( m , n , examples_train.data , 
                       m , d_examples_train.data , m);
   
    m = examples_cv.row;
    n = examples_cv.col;
    d_examples_cv.row = examples_cv.row;
    d_examples_cv.col = examples_cv.col;
    magma_dmalloc(  &(d_examples_cv.data), m*n );
    magma_dsetmatrix ( m , n , examples_cv.data , 
                       m , d_examples_cv.data , m);
    
    m = examples_test.row;
    n = examples_test.col;
    d_examples_test.row = examples_test.row;
    d_examples_test.col = examples_test.col;
    magma_dmalloc(  &(d_examples_test.data), m*n );
    magma_dsetmatrix ( m , n , examples_test.data , 
                       m , d_examples_test.data , m);
 
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
    magma_dmalloc  (   &d_nn_params , nn_paramsLength );
    d_Theta   = (MATRIX*) malloc(sizeof(MATRIX) * (NUM_HIDDEN_LAYERS + 1));
    d_Theta_grad=(MATRIX*)malloc(sizeof(MATRIX) * (NUM_HIDDEN_LAYERS + 1));
    grad = (double*) malloc(sizeof(double)*nn_paramsLength);
    magma_dmalloc( &d_grad , nn_paramsLength );

    Theta[0].col = featureNum + 1;
    Theta[0].row = HIDDEN_LAYER_NODE_NUM;
    Theta[0].data= &nn_params[0];

    d_Theta[0].col = Theta[0].col;
    d_Theta[0].row = Theta[0].row;
    d_Theta[0].data= &d_nn_params[0];
    
    d_Theta_grad[0].col = Theta[0].col;
    d_Theta_grad[0].row = Theta[0].row;
    d_Theta_grad[0].data= &d_grad[0];
    
    index = (featureNum+1) * HIDDEN_LAYER_NODE_NUM ;
    
    for( i=1; i<NUM_HIDDEN_LAYERS; i++){ 
        Theta[i].col = HIDDEN_LAYER_NODE_NUM + 1;
	Theta[i].row = HIDDEN_LAYER_NODE_NUM;
        Theta[i].data= &nn_params[index];

        d_Theta[i].col = Theta[i].col;
	d_Theta[i].row = Theta[i].row;
	d_Theta[i].data= &d_nn_params[index];

	d_Theta_grad[i].col = Theta[i].col;
	d_Theta_grad[i].row = Theta[i].row;
	d_Theta_grad[i].data= &d_grad[index];

	index += (HIDDEN_LAYER_NODE_NUM + 1)*
	          HIDDEN_LAYER_NODE_NUM;  
    }
    
    Theta [NUM_HIDDEN_LAYERS].col = HIDDEN_LAYER_NODE_NUM+1;
    Theta [NUM_HIDDEN_LAYERS].row = NUM_OUTPUT_NODE;
    Theta [NUM_HIDDEN_LAYERS].data= &nn_params[index];
    
    d_Theta[NUM_HIDDEN_LAYERS].col = Theta [NUM_HIDDEN_LAYERS].col;
    d_Theta[NUM_HIDDEN_LAYERS].row = Theta [NUM_HIDDEN_LAYERS].row;
    d_Theta[NUM_HIDDEN_LAYERS].data= &d_nn_params[index];

    d_Theta_grad[NUM_HIDDEN_LAYERS].col = Theta [NUM_HIDDEN_LAYERS].col;
    d_Theta_grad[NUM_HIDDEN_LAYERS].row = Theta [NUM_HIDDEN_LAYERS].row;
    d_Theta_grad[NUM_HIDDEN_LAYERS].data= &d_grad[index];
    //Initialize
    for( i=0; i<NUM_HIDDEN_LAYERS+1; i++){
        length = Theta[i].col * Theta[i].row;
	for( j=0; j<length; j++){
	    Theta[i].data[j] = random_generate( -epsilon[i] , epsilon[i] ); 
	}
    }
    //put nn_params to GPU

    magma_dsetvector(   nn_paramsLength , 
                        nn_params , 1 ,
                      d_nn_params , 1   );

    
    // initialize d_a,better also d_Theta and d_Theta_grad 
    d_a = (MATRIX*)malloc( sizeof(MATRIX)* (NUM_HIDDEN_LAYERS + 2) );
    a   = (MATRIX*)malloc( sizeof(MATRIX)* (NUM_HIDDEN_LAYERS + 2) );
    d_delta = (MATRIX*)malloc( sizeof(MATRIX)* (NUM_HIDDEN_LAYERS + 2) ); 
    d_a[0].col = TRAINING_SIZE;
    d_a[0].row = featureNum + 1;
    magma_dmalloc( &(d_a[0].data) ,d_a[0].row*d_a[0].col);
    
    a[0].row   = d_a[0].row;
    a[0].col   = d_a[0].col;
    a[0].data  = (double*)malloc(sizeof(double)*a[0].row*a[0].col);

    d_delta[0].row = a[0].row;
    d_delta[0].col = a[0].col;

    for( i=0 ; i<a[0].col ; i++ ){
        a[0].data[ i*a[0].row ] = 1.0 ;
	cblas_dcopy( examples_train.row    ,
	             examples_train.data+i*featureNum,
		     1                     ,
	             a[0].data+i*a[0].row+1,
		     1                    );
    }
    magma_dsetvector( a[0].row*a[0].col,
                      a[0].data        ,
		      1                ,
		      d_a[0].data      ,
		      1                );


    for( i=1 ; i<NUM_HIDDEN_LAYERS+1 ; i++ ){
        a[i].col = TRAINING_SIZE;
	a[i].row = HIDDEN_LAYER_NODE_NUM+1;
	a[i].data= (double*)malloc(sizeof(double)*a[i].col*a[i].row);

        d_a[i].col=a[i].col;
	d_a[i].row=a[i].row;
	magma_dmalloc( &(d_a[i].data) , d_a[i].col*d_a[i].row );
	
	d_delta[i].col = a[i].col;
	d_delta[i].row = a[i].row;
        magma_dmalloc( &(d_delta[i].data), a[i].col*a[i].row );
	
	for( j=0 ; j<a[i].col ; j++ ){
	    a[i].data[ j*a[i].row ] = 1.0;
	}
	magma_dsetvector( a[i].row*a[i].col,
	                  a[i].data        ,
			  1                ,
			  d_a[i].data      ,
			  1                );
    }
    a[NUM_HIDDEN_LAYERS+1].col = TRAINING_SIZE;
    a[NUM_HIDDEN_LAYERS+1].row = NUM_OUTPUT_NODE;
    mallocSize = NUM_OUTPUT_NODE*TRAINING_SIZE;
    a[NUM_HIDDEN_LAYERS+1].data= (double*)malloc(sizeof(double)*mallocSize);
    
    d_a[NUM_HIDDEN_LAYERS+1].col = TRAINING_SIZE;
    d_a[NUM_HIDDEN_LAYERS+1].row = NUM_OUTPUT_NODE;
    magma_dmalloc( &(d_a[NUM_HIDDEN_LAYERS+1].data) , mallocSize) ;
    
    d_delta[NUM_HIDDEN_LAYERS+1].col = TRAINING_SIZE;
    d_delta[NUM_HIDDEN_LAYERS+1].row = NUM_OUTPUT_NODE;
    magma_dmalloc( &(d_delta[NUM_HIDDEN_LAYERS+1].data) , mallocSize );

    //printf("Regularization term before passing to fcn:%lf\n", correction); 
    J = costFunction( d_grad,
                      d_nn_params,
		      d_a,
                      d_Theta,
		      d_Theta_grad,
		      d_delta,
		      featureNum,
		      HIDDEN_LAYER_NODE_NUM,
		      NUM_OUTPUT_NODE,
		      NUM_HIDDEN_LAYERS,
		      d_coefMatrix_train,
		      lambda );
    printf("J= %lf\n", J);
    //gradient check
    ep = 0.001;
    gradientCheckIndex = 5;
    p1 = (double*) malloc(sizeof(double)*nn_paramsLength);
    p2 = (double*) malloc(sizeof(double)*nn_paramsLength);
    for( i=0; i<nn_paramsLength; i++){
        p1[i] = nn_params[i];
        p2[i] = nn_params[i];
	
    }
    p1[gradientCheckIndex] = p1[gradientCheckIndex] + ep;
    p2[gradientCheckIndex] = p2[gradientCheckIndex] - ep;
    
    magma_dmalloc (&d_p1,nn_paramsLength);
    magma_dmalloc (&d_p2,nn_paramsLength);

    magma_dsetvector( nn_paramsLength,
                      p1,
		      1,
		      d_p1,
		      1           );
    magma_dsetvector( nn_paramsLength,
                      p2,
                      1,
                      d_p2,
                      1           );

    magma_dsetvector( nn_paramsLength,
                      p1,
		      1,
		      d_nn_params,
		      1           );

    J1 = costFunction(d_grad,
                      d_p1,
		      d_a,
		      d_Theta,
		      d_Theta_grad,
		      d_delta,
		      featureNum,
		      HIDDEN_LAYER_NODE_NUM,
		      NUM_OUTPUT_NODE,
		      NUM_HIDDEN_LAYERS,
		      d_coefMatrix_train,
		      lambda);
    magma_dsetvector (nn_paramsLength,
                      p2,
		      1,
		      d_nn_params,
		      1               );

    J2 = costFunction(d_grad,
                      d_p2,
		      d_a,
		      d_Theta,
		      d_Theta_grad,
		      d_delta,
		      featureNum,
		      HIDDEN_LAYER_NODE_NUM,
		      NUM_OUTPUT_NODE,
		      NUM_HIDDEN_LAYERS,
		      d_coefMatrix_train,
		      lambda);
//    printf("J2 = %lf\n",J2);
    magma_dsetvector( nn_paramsLength,
                      nn_params,
		      1,
		      d_nn_params,
		      1              );
    J = costFunction( d_grad,
                      d_nn_params,
		      d_a,
		      d_Theta,
		      d_Theta_grad,
		      d_delta,
		      featureNum,
		      HIDDEN_LAYER_NODE_NUM,
		      NUM_OUTPUT_NODE,
		      NUM_HIDDEN_LAYERS,
		      d_coefMatrix_train,
		      lambda);

    magma_dgetvector( nn_paramsLength,d_grad,1,grad,1);
                      
    printf("Backpropogation grad:%lf\n" , grad[gradientCheckIndex]);
    printf("Real grad: %lf\n", (J1-J2)/(2*ep));
    magma_free(d_p1);
    magma_free(d_p2);
    //end of gradient check 
    double cost( double* d_p, double *d_gradient){
        double returnValue;
        returnValue = costFunction( d_gradient,
                                    d_p,
                                    d_a,
				    d_Theta,
				    d_Theta_grad,
				    d_delta,
		                    featureNum,
		                    HIDDEN_LAYER_NODE_NUM,
		                    NUM_OUTPUT_NODE,
		                    NUM_HIDDEN_LAYERS,
		                    d_coefMatrix_train,
		                    lambda);
        return returnValue ; 
    }

    struct timeval start, end;
    double diff;
    gettimeofday(&start, NULL);
    J = fmincg (d_nn_params , d_grad , cost, maxIter , nn_paramsLength);
/*    for( i=0 ; i<10000 ; i++){
         J = costFunction( d_grad,
                      d_nn_params,
                      d_a,
                      d_Theta,
                      d_Theta_grad,
                      d_delta,
                      featureNum,
                      HIDDEN_LAYER_NODE_NUM,
                      NUM_OUTPUT_NODE,
                      NUM_HIDDEN_LAYERS,
                      //d_examples_train,
                      d_coefMatrix_train,
                      lambda);
    }    
*/
    gettimeofday(&end, NULL);
    diff= ((end.tv_sec  - start.tv_sec) * 1000000u +
         end.tv_usec - start.tv_usec) / 1.e6;
    printf("Time elapsed = %f sec\n", diff);
    
    //free everything
     
    free(epsilon);
    free(Theta);
    free(grad);
    free(nn_params);
    free(d_Theta);
    free(d_Theta_grad);
    
    free(coefMatrix_train.data);
    free(coefMatrix_cv.data);
    free(coefMatrix_test.data);
    free(examples_train.data);
    free(examples_cv.data);
    free(examples_test.data);
    for( i=0 ; i<NUM_HIDDEN_LAYERS+2 ; i++ ){
        free(a[i].data);
	magma_free(d_a[i].data);
        magma_free(d_delta[i].data);
    }
     
    free(a);
    free(d_a);
    free(d_delta);
    magma_free(d_coefMatrix_train.data);
    magma_free(d_coefMatrix_cv.data);
    magma_free(d_coefMatrix_test.data);
    magma_free(d_examples_train.data);
    magma_free(d_examples_cv.data);
    magma_free(d_examples_test.data);
    magma_free(d_nn_params);
    magma_free(d_grad);
    magma_finalize();
    return 0;
}
