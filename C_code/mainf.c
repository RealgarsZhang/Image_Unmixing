#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> 
          
#include "magma.h"
#include "magma_lapack.h"
         
#include "cublas_v2.h"    

#include"NoNoiseNoPrint.h"
int main(){
    magma_init();
    MATRIX coefMatrix_train, coefMatrix_cv, coefMatrix_test;
    MATRIX examples_train,   examples_cv,   examples_test;

    readinExamples(&coefMatrix_train, 
                   &coefMatrix_cv,
		   &coefMatrix_test,
		   &examples_train,
		   &examples_cv,
		   &examples_test);



    free(coefMatrix_train.data);
    free(coefMatrix_cv.data);
    free(coefMatrix_test.data);
    free(examples_train.data);
    free(examples_cv.data);
    free(examples_test.data);
    magma_finalize();
    return 0;
}
