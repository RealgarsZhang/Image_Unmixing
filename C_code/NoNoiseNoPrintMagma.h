#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "magma.h"
#include "magma_lapack.h"
struct matrixx{
  double* data;
  unsigned int row;
  unsigned int col;

};

typedef struct matrixx MATRIX;

double tanh(double x);

double random_generate(double min, double max);

void my_dgemm (MATRIX A, MATRIX B, MATRIX C, 
               double alpha, double beta,
               magma_trans_t A_tran,
               magma_trans_t B_tran);



/**********************************************
  Description: readinExamples() reads in the examples 
generated by matlab.For each set of training examles,
both the example image and the coeffcients for the 
three basic modes are provided.
  NOTE: Don't malloc for the six input arguments!
        Malloc will be done in this function!

************************************************/
int readinExamples( MATRIX* coefMatrix_train,
                     MATRIX* coefMatrix_cv,
		     MATRIX* coefMatrix_test,
                     MATRIX* examples_train,
		     MATRIX* examples_cv,
		     MATRIX* examples_test);


/***************************************************
 *     Description: padOne() add a row where all entries are
 *   1 to the top of a matrix A. Return a MATRIX struct
 *     NOTE : malloc() is called here. To allocate space for the 
 *   new matrix.
 *         ****************************************************/
MATRIX padOne(MATRIX A);
void apply_activation_funtion (MATRIX temp);


void pointwise_mult_and_apply_diff_activation(MATRIX d_delta,MATRIX d_a);

/**************************************************
  Description: costFunction computes the value of the cost function,
as well as the gradient. The return value will be the value and the
argument grad will store the gradient, which is unrolled according to 
the "Theta(:)" in matlab.
****************************************************/
double costFunction ( double* grad,
		      double* nn_params,
                      MATRIX* d_a,
                      MATRIX* d_Theta,
                      MATRIX* d_Theta_grad,
                      MATRIX* d_delta,
		      const int input_layer_size,
                      const int hidden_layer_size,
		      const int num_labels,
		      const int NUM_HIDDEN_LAYERS,
		      //const MATRIX X,
                      const MATRIX Y,
		      const double lambda);



double innerProduct (double* v1, double* v2, int length);
void   vectorAssignment (double * v1, double *v2, int length, double alpha);
double my_min( double x, double y);
double my_max( double x, double y);
double fmincg (       double* d_nn_params,
                      double* d_grad,
                      double (*cost)(double*,double*),
                      const int maxIter,
                      const int nn_paramsLength);