#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "magma.h"
#include "magma_lapack.h"

#include"NoNoiseNoPrintMagma.h"

double tanh(double x){
    return (exp(x)-exp(-x))/(exp(x)+exp(-x));
 
}

double random_generate(double min, double max){
    double returnValue;
    returnValue = (double)rand()/(double)RAND_MAX;
    returnValue = min + (max-min)* returnValue;
    return returnValue;   

}
/**************************************************
  Description: readinExamples() reads in the examples 
generated by matlab.For each set of training examles,
both the examples image and the coeffcients for the 
three basic modes are provided.
  Return value: 0 if some error, 1 if everything ok.

  NOTE: You should pass the address of the matrix
        Don't malloc space of the matrix! This is 
	done in this function!!!

	All arguments are allocated and NOT freed.
************************************************/
int readinExamples(MATRIX* ptr_coefMatrix_train,
                    MATRIX* ptr_coefMatrix_cv,
                    MATRIX* ptr_coefMatrix_test,
                    MATRIX* ptr_examples_train,
                    MATRIX* ptr_examples_cv,
                    MATRIX* ptr_examples_test){
    FILE* fileInput;
    int row,col;
    int temp_size;
    int i;
    int err;
    double temp;
    
    
 
   //readin coefMatrix_train
    fileInput = fopen("./examples100/coefMatrix_train","r");
    if ( fileInput == NULL ){
        printf( "Unable to open the file!\n" );
	return (0);
    }
    
    fscanf(fileInput, "%d %d", &row, &col);
    //printf("row:%d col:%d\n", row, col);
    ptr_coefMatrix_train->row = row;
    ptr_coefMatrix_train->col = col;
    ptr_coefMatrix_train->data= (double*)malloc(sizeof(double)*row*col);
    
    i = 0;
    temp_size = row*col;
    while(i < temp_size) {
 	fscanf( fileInput, "%lf", &temp );
        //printf("num:%lf\n", temp);
        ptr_coefMatrix_train->data[i] = temp;
        i++;
   }
    fclose( fileInput );
    
    //read coefMatrix_cv
    fileInput = fopen("./examples/coefMatrix_cv","r");
    if ( fileInput == NULL ){
        printf( "Unable to open the file!\n" );
	return (0);
    }

    fscanf(fileInput, "%d %d", &row, &col);
    ptr_coefMatrix_cv->row = row;
    ptr_coefMatrix_cv->col = col;
   // ptr_coefMatrix_cv->data= (double*)malloc(sizeof(double)*row*col);
    ptr_coefMatrix_cv->data= (double*)malloc(sizeof(double)*row*col);
    //printf("row:%d col:%d\n", row, col);    
   
    i = 0;

    temp_size = row*col;
    while(i < temp_size) {
 	fscanf( fileInput, "%lf", &temp );
    //    printf("num:%lf\n", temp);
        ptr_coefMatrix_cv->data[i] = temp;
        i++;
   }

    fclose( fileInput );
    
    //read in coefMatrix_test
    fileInput = fopen("./examples/coefMatrix_test","r");
    if ( fileInput == NULL ){
        printf( "Unable to open the file!\n" );
	return (0);
    }

    fscanf(fileInput, "%d %d", &row, &col);
    ptr_coefMatrix_test->row = row;
    ptr_coefMatrix_test->col = col;
    ptr_coefMatrix_test->data= (double*)malloc(sizeof(double)*row*col);
    //ptr_coefMatrix_test->data= (double*)malloc(sizeof(double)*row*col);
    //printf("row:%d col:%d\n", row, col); 
    i = 0;
    temp_size = row*col;
    while(i < temp_size) {
 	fscanf( fileInput, "%lf", &temp );
    //    printf("num:%lf\n", temp);
        ptr_coefMatrix_test->data[i] = temp;
        i++;
   }

    fclose( fileInput );
 
 
    fileInput = fopen("./examples100/examples_train","r");
    if ( fileInput == NULL ){
        printf( "Unable to open the file!\n" );
	return (0);
    }

    fscanf(fileInput, "%d %d", &row, &col);
    ptr_examples_train->row = row;
    ptr_examples_train->col = col;
    ptr_examples_train->data= (double*)malloc(sizeof(double)*row*col);
    //ptr_examples_train->data= (double*)malloc(sizeof(double)*tempsize);
    //printf("row:%d col:%d\n", row, col);
    i = 0;
    temp_size = row*col;
    while(i < temp_size) {
        fscanf( fileInput, "%lf", &temp );
    //    printf("num:%lf\n", temp);
        ptr_examples_train->data[i] = temp;
        i++;
   }

    fclose( fileInput );
 
    fileInput = fopen("./examples/examples_cv","r");
    if ( fileInput == NULL ){
        printf( "Unable to open the file!\n" );
	return (0);
    }

    fscanf(fileInput, "%d %d", &row, &col);
    ptr_examples_cv->row = row;
    ptr_examples_cv->col = col;
    ptr_examples_cv->data= (double*)malloc(sizeof(double)*row*col);
    //ptr_examples_cv->data= (double*)malloc(sizeof(double)*tempsize);
    //printf("row:%d col:%d\n", row, col);
    i = 0;
    temp_size = row*col;
    while(i < temp_size) {
        fscanf( fileInput, "%lf", &temp );
        //printf("num%d:%lf\n",i, temp);
        ptr_examples_cv->data[i] = temp;
        i++;
   }

    fclose( fileInput );

    fileInput = fopen("./examples/examples_test","r");
    if ( fileInput == NULL ){
        printf( "Unable to open the file!\n" );
	return (0);
    }

    fscanf(fileInput, "%d %d", &row, &col);
    ptr_examples_test->row = row;
    ptr_examples_test->col = col;
    ptr_examples_test->data= (double*)malloc(sizeof(double)*row*col);
    //ptr_examples_test->data= (double*)malloc(sizeof(double)*tempsize);
    //printf("row:%d col:%d\n", row, col);
    i = 0;
    temp_size = row*col;
    while(i < temp_size) {
        fscanf( fileInput, "%lf", &temp );
    //    printf("num:%lf\n", temp);
        ptr_examples_test->data[i] = temp;
        i++;
   }
    printf("Examples successfully read in!\n");
    fclose( fileInput );
    return (1);
}

/**************************************************
 *     Description: costFunction computes the value of the cost function,
 *   as well as the gradient. The return value will be the value and the
 *   argument grad will store the gradient, which is unrolled according to 
 *   the "d_Theta(:)" in matlab.
 *     NOTE : For the regularization not penalizing bias, d_grad and d_nn_params are NOT used in this function, Future function calls will pass these to arguments, That is only to make the code more friendly. Actually, passing d_Theta and d_Theta_grad is already enough, because they point to the d_grad and d_nn_params in the main function. However, for the other kind of regularization, d_grad and d_nn_params are used.
 *     ****************************************************/
double costFunction ( double* d_grad,
                      double* d_nn_params,
                      MATRIX* d_a,//keep the top rows of d_a at all costs!
		      MATRIX* d_Theta,
		      MATRIX* d_Theta_grad,
		      MATRIX* d_delta,
		      const int input_layer_size,
                      const int hidden_layer_size,
                      const int num_labels,
                      const int NUM_HIDDEN_LAYERS,
                      const MATRIX Y,//Y should be on GPU
                      const double lambda){
    double J = 0;       // The value of the cost function will be stored in J.
    int currentProg = 0;
    int i = 0;
    int k = 0, index;
    int j = 0;
    int jStart, jEnd;
    int m;
    int ii,jj;
    int mallocSize;
    int err;
    MATRIX temp;
int nn_paramsLength;
nn_paramsLength = (input_layer_size+1)*hidden_layer_size+
             (NUM_HIDDEN_LAYERS-1)*(hidden_layer_size+1)*hidden_layer_size+
             (hidden_layer_size+1)*num_labels;
    double tempDouble;
    double correction=0;
   
    for( i=1; i<=NUM_HIDDEN_LAYERS; i++){
        magma_dgemm( MagmaNoTrans      ,
	             MagmaNoTrans      , 
		     d_Theta[i-1].row          ,
		     d_a[i-1].col          ,
		     d_Theta[i-1].col  ,
		     1.0               ,
		     d_Theta[i-1].data ,
		     d_Theta[i-1].row  ,
		     d_a[i-1].data     ,
		     d_a[i-1].row      ,
                     0.0               ,
		     d_a[i].data+1         ,
		     d_a[i].row          );

        magmablas_dlatanh( d_a[i].row -1    ,
                           d_a[i].col       ,
			   d_a[i].data+1    , 
			   d_a[i].row         ,
			   d_a[i].data+1    ,
			   d_a[i].row   );

     }

    magma_dgemm(     MagmaNoTrans                    ,
	             MagmaNoTrans                    , 
		     d_a[NUM_HIDDEN_LAYERS+1].row    ,
		     d_a[NUM_HIDDEN_LAYERS+1].col    ,
		     d_Theta[NUM_HIDDEN_LAYERS].col  ,
		     1.0                             ,
		     d_Theta[NUM_HIDDEN_LAYERS].data ,
		     d_Theta[NUM_HIDDEN_LAYERS].row  ,
		     d_a[NUM_HIDDEN_LAYERS].data     ,
		     d_a[NUM_HIDDEN_LAYERS].row      ,
                     0.0                             ,
		     d_a[NUM_HIDDEN_LAYERS+1].data   ,
		     d_a[NUM_HIDDEN_LAYERS+1].row    );

    magmablas_dlacpy( MagmaFull ,
                      Y.row     ,
                      Y.col     ,
                      d_a[NUM_HIDDEN_LAYERS+1].data    ,
                      Y.row     ,
                      d_delta[NUM_HIDDEN_LAYERS+1].data,
                      Y.row     );

    magmablas_dgeadd ( Y.row     ,
                       Y.col     ,
                       -1.0      ,
                       Y.data    ,
                       Y.row     ,
                       d_delta[NUM_HIDDEN_LAYERS+1].data ,
                       Y.row     );
    J = magma_dnrm2 ( Y.row*Y.col ,
                      d_delta[NUM_HIDDEN_LAYERS+1].data   ,
                      1           );
    J = J*J;
    J = J/2;

    //regularization: not penalizing the bias
   /* correction = 0;
    for( i=0; i<NUM_HIDDEN_LAYERS+1; i++ ){
        //jStart = d_Theta[i].row;//7.18 modification: from col to row.
	tempDouble = magma_dnrm2( d_Theta[i].row*(d_Theta[i].col - 1),
                                  &(d_Theta[i].data[d_Theta[i].row]),
				  1 );
	tempDouble *= tempDouble;
	correction += tempDouble;
    }
    correction *= lambda/2;
    J += correction;
    */
    // regularization:penalizing the bias:
    correction = magma_dnrm2( nn_paramsLength,
                              d_nn_params    ,
                              1              );
    correction *= correction*lambda/2;    
    J += correction;
    
    magma_dgemm ( MagmaNoTrans                        ,
                  MagmaTrans                          ,
                  d_Theta_grad[NUM_HIDDEN_LAYERS].row ,
		  d_Theta_grad[NUM_HIDDEN_LAYERS].col ,
                  d_delta[NUM_HIDDEN_LAYERS+1].col    ,
                  1.0                                 ,
		  d_delta[NUM_HIDDEN_LAYERS+1].data   ,
                  d_delta[NUM_HIDDEN_LAYERS+1].row    ,
                  d_a[NUM_HIDDEN_LAYERS].data         ,
		  d_a[NUM_HIDDEN_LAYERS].row          ,
		  0.0                                 ,
		  d_Theta_grad[NUM_HIDDEN_LAYERS].data,
		  d_Theta_grad[NUM_HIDDEN_LAYERS].row );


   for( i=1; i<=NUM_HIDDEN_LAYERS; i++ ){
        k = NUM_HIDDEN_LAYERS - i;
        mallocSize = d_delta[k+1].col * d_delta[k+1].row;
	
	magma_dgemm( MagmaTrans       ,
                     MagmaNoTrans     ,
                     d_delta[k+1].row ,
                     d_delta[k+1].col ,
                     d_Theta[k+1].row ,
                     1.0              ,
                     d_Theta[k+1].data,
                     d_Theta[k+1].row ,
                     d_delta[k+2].data+(i!=1),
                     d_delta[k+2].row ,
                     0.0              ,
                     d_delta[k+1].data,
                     d_delta[k+1].row );

        magmablas_dla_pmult_dtanh( d_a[k+1].row     ,
	                           d_a[k+1].col     ,
				   d_a[k+1].data    ,
				   d_a[k+1].row     ,
				   d_delta[k+1].data,
				   d_delta[k+1].row );

        magma_dgemm( MagmaNoTrans          ,
                     MagmaTrans            ,
                     d_Theta_grad[k].row   ,
                     d_Theta_grad[k].col   ,
                     d_a[k].col            ,
                     1.0                   ,
		     d_delta[k+1].data + 1 ,
		     d_delta[k+1].row      ,
                     d_a[k].data           ,
                     d_a[k].row            ,
		     0.0                   ,
		     d_Theta_grad[k].data  ,
		     d_Theta_grad[k].row   );
        // regularization: not penalizing the bias.
        /*magmablas_dgeadd( d_Theta_grad[k].row                        ,
	                  d_Theta_grad[k].col - 1                    ,
			  lambda                                     ,
			  d_Theta[k].data + d_Theta[k].row           ,
			  d_Theta[k].row                             ,
			  d_Theta_grad[k].data + d_Theta_grad[k].row ,
			  d_Theta_grad[k].row );
	  */
                        
    }	
    //regularization : penalizing the bias
    magma_daxpy( nn_paramsLength,
                 lambda,
		 d_nn_params,
		 1,
		 d_grad,
		 1               );

         
     
   return J;
}


double my_min(double x, double y){
    if ( x>y ){ return y;}
    else { return x;}

}
double my_max(double x, double y){
    if ( x>y ){ return x; } 
    else { return y; }
}
/*******************************************************
 * The fmincg() minimizes the costFunction(),the nn_params will be finally 
 * modified. Return the ultimate value of the costFunction.
 **********************************************************/

double fmincg (       double* d_nn_params,
                      double* d_grad,
                      double (*cost)(double*,double*),
                      const int maxIter,
		      const int nn_paramsLength){
    int length = maxIter;
    
    double RHO   = 0.01,
           SIG   = 0.5,
	   INT   = 0.1,
	   EXT   = 3.0,
	   MAX   = 20.0,
           RATIO = 100.0,
           red   = 1.0;
    
    int        i = 0,
       ls_failed = 0;
    
    double *d_df0, *d_df1, *d_df2, *d_df3, f0, f1, f2, f3;
    double *d_s;
    double *d_nn_params0;
    double d1, d2, d3;
    double z1, z2, z3;
    double M , limit, A, B;
    int    ii, jj, kk, success;
    double alpha;
    double returnValue;
 
    magma_dmalloc( &d_df0,nn_paramsLength );
    magma_dmalloc( &d_df1,nn_paramsLength );
    magma_dmalloc( &d_df2,nn_paramsLength );
    magma_dmalloc( &d_df3,nn_paramsLength );
    magma_dmalloc( &d_s  ,nn_paramsLength );
    magma_dmalloc( &d_nn_params0,nn_paramsLength );
 
    f1 = (*cost) (d_nn_params, d_grad);
    magma_dcopy( nn_paramsLength , d_grad , 1, d_df1 , 1 );
    //s=-d_df1;
    magma_dcopy( nn_paramsLength,
                 d_df1          ,
		 1              ,
		 d_s            ,
		 1              );
    magma_dscal( nn_paramsLength,
                 -1.0           ,
		 d_s            ,
		 1              );
    //d1=-s'*s;
    d1 = magma_dnrm2(nn_paramsLength,
                     d_s            ,
		     1              );
    d1 *= d1;
    d1 = -d1;

    z1 = red/(1-d1);
    
    while( i < length ){
        i += ( length>0 );       
        
	//nn_params0=nn_params; 
	//f0=f1; 
	//d_df0=d_df1
	//nn_params += z1*s 
	f0 = f1;
        magma_dcopy (nn_paramsLength,
	             d_nn_params    ,
		     1              ,
		     d_nn_params0   ,
		     1              );
        magma_dcopy (nn_paramsLength,
	             d_df1          ,
		     1              ,
		     d_df0          ,
		     1              );
        magma_daxpy (nn_paramsLength,
	             z1             ,
		     d_s            ,
		     1              ,
                     d_nn_params    ,
		     1              );
         f2 = (*cost) ( d_nn_params , d_grad );
         magma_dcopy( nn_paramsLength ,d_grad , 1, d_df2 , 1 );
	 i += (length<0);
	 d2 = magma_ddot( nn_paramsLength,
	                  d_df2          ,
			  1              ,
			  d_s            , 
			  1              );
         
	 f3 = f1; d3 = d1; z3 = -z1;
         
	 if (length > 0){
	     M = MAX;
	 }else{
	     M = my_min( MAX , (double) (-length - i) );
	 }

         success = 0;
	 limit = -1.0;

	 while (1){
             while(  ( (f2>f1+z1*RHO*d1) || (d2>-SIG*d1) ) && (M>0)  ){
	         
		 limit = z1;
		 if (f2 > f1){
		     z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);
		 }else{
                     A = 6*(f2-f3)/z3 + 3*(d2+d3) ;
		     B = 3*(f3-f2) - z3*(d3+2*d2);
		     z2= (sqrt(B*B - A*d2*z3*z3)-B)/A;
		 }
	         
		 z2 = my_max( my_min( z2 , INT*z3 ) , (1-INT)*z3 );
	         z1 = z1 + z2;
	     
	         magma_daxpy( nn_paramsLength,
		              z2             ,
			      d_s            ,
			      1              ,
			      d_nn_params    ,
			      1              );
	         
		 f2 = (*cost) ( d_nn_params , d_df2 );
		 magma_dcopy( nn_paramsLength ,d_grad , 1, d_df2 , 1 );
                 M --;
		 i += ( length<0 );
		 d2 = magma_ddot( nn_paramsLength ,
		                  d_df2           ,
				  1               ,
				  d_s             ,
				  1               );
                 z3 = z3 - z2;
	     
	     }	 
             
	     if( f2 > f1+z1*RHO*d1 || d2 > -SIG*d1 ){
	         break;
	     }
	     else if( d2 > SIG*d1 ){
	         success = 1;
		 break;
	     }
	     else if( M==0 ){
	         break;
	     } 

	     A = 6*(f2-f3)/z3 + 3*(d2+d3);
	     B = 3*(f3-f2) - z3*(d3+2*d2);
	     
	     if( B*B-A*d2*z3*z3 >= 0 ){
	         z2 = -d2*z3*z3/( B + sqrt( B*B-A*d2*z3*z3) );
	     }else{
	         z2 = -d2*z3*z3/( B + sqrt(-B*B+A*d2*z3*z3) );
	     }

	     if( B*B-A*d2*z3*z3 < 0 || z2<0 ){
	         if( limit < -0.5 ){
	             z2 = z1 * (EXT-1) ;
	         }
	         else{
	             z2 = ( limit-z1 )/2 ;
	         }
	     }
             else if( limit > -0.5 && z2+z1 > limit ){
	         z2 = (limit-z1)/2;
	     }
             else if( limit < -0.5 && z2+z1 > z1*EXT){
	         z2 = z1*(EXT-1.0);
	     }
	     else if(z2 < -z3*INT){
	         z2 = -z3 * INT;
	     }
	     else if( limit > -0.5 && z2 < (limit-z1)*(1.0-INT) ){
	         z2 = (limit-z1)*(1.0-INT);
	     }

	     f3 = f2; d3 = d2 ; z3 = -z2;
	     z1 = z1 + z2; 
	     magma_daxpy( nn_paramsLength,
	                  z2             ,
			  d_s            ,
			  1              ,
			  d_nn_params    ,
			  1              );
             
	     f2 = (*cost) ( d_nn_params , d_grad );
	     magma_dcopy( nn_paramsLength ,d_grad , 1, d_df2 , 1 );
             M --;
	     i += (length<0);
	     d2 = magma_ddot (nn_paramsLength,
	                      d_df2          ,
			      1              ,
			      d_s            ,
			      1              );
			      

	 }

	 if(success){
	     f1 = f2;
             returnValue = f1;	     //fX = [fX' f1]'
             //printf("%d | Cost: %lf\r", i , f1 );
	     // s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;
	     alpha = (  magma_ddot( nn_paramsLength, d_df2,1,d_df2,1 )
	               -magma_ddot( nn_paramsLength, d_df1,1,d_df2,1 ) )
		     /
		     magma_ddot( nn_paramsLength, d_df1,1,d_df1,1);
	       // s = alpha*s -df2;
             magma_dscal( nn_paramsLength,
	                  alpha          ,
			  d_s            ,
			  1              );
	     magma_daxpy( nn_paramsLength,
	                  -1.0           ,
			  d_df2          ,
			  1              ,
			  d_s            ,
			  1              );
             
            //swap:tmp = df1;df1=df2;df2=tmp;d2=df1'*s;
            magma_dswap( nn_paramsLength,
	                  d_df1,1,
			  d_df2,1);
	 
             d2 = magma_ddot( nn_paramsLength,
	                      d_df1,1,
			      d_s  ,1        );

	     if (d2>0){
                 magma_dcopy( nn_paramsLength,
		              d_df1,1,
			      d_s  ,1        );
	         magma_dscal( nn_paramsLength,
		              -1.0,
			      d_s ,1         );
	         d2 = -magma_ddot( nn_paramsLength,
		                  d_s,1,
				  d_s,1          );
	     }
	     z1 *= my_min( RATIO , d1/d2 );
	     d1  = d2;
	     ls_failed = 0;   
	 }else{
	     magma_dcopy( nn_paramsLength,
	                  d_nn_params0,1,
			  d_nn_params ,1   ); 
	     f1 = f0;
	     magma_dcopy( nn_paramsLength,
	                  d_df0,1,
			  d_df1,1        );
             if( ls_failed || i>abs(length) ){
	         break;
	     }
	     magma_dswap( nn_paramsLength,
	                  d_df1,1,
			  d_df2,1        );
             magma_dcopy( nn_paramsLength,
                          d_df1, 1,
			  d_s  , 1       );
	     magma_dscal( nn_paramsLength,
	                  -1.0,
			  d_s,
			  1              );
             d1 = -magma_ddot( nn_paramsLength, d_s,1,d_s,1);

	     z1 = 1/(1-d1);
	     ls_failed = 1;
    
	 }
    }
    printf("J = %lf\n", returnValue);
    magma_free( d_df0);
    magma_free( d_df1);
    magma_free( d_df2);
    magma_free( d_df3);
    magma_free( d_s );
    magma_free( d_nn_params0);
    return returnValue;   
}


