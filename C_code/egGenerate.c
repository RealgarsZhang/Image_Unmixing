#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "magma.h"
#include "magma_lapack.h"



#include"NoNoiseNoPrint.h"
void examplesGenerate(MATRIX* coefMatrix,        // in, out
                      MATRIX* result,            // in, out
                      unsigned int numExamples,
                      double min,double max){
    int N = 192;
    int N_base=768;
    double *full_data[16], *Model1, *Model2, *Model_base;
    double *A, *temp_matrix, *temp_w, *temp_result, *unit_err;
    double temp;
    magma_int_t i, j, Nx_unit, Ny_unit, Nz_unit_1, Nz_unit_2, Nx_1, Ny_1, Nz_1, Nx_2, Ny_2, Nz_2;
    magma_int_t Nx_base, Ny_base, Nz_base, unit_cell_size, temp_size;
    FILE *input;
   //reading in model_1
    input = fopen("model_one", "r");
    if(input == NULL) {
        printf("Error, file not found\n");
	return(1);
    }

    fscanf(input, "%d %d %d", &Nx_1, &Ny_1, &Nz_1);
    temp_size = Nx_1*Ny_1*Nz_1;

    //192x192x14x14
    Model1 = (double*)malloc(temp_size*sizeof(double));

    i = 0;
    fscanf(input, "%lf", &temp);
    while(i < temp_size) {
        Model1[i] = temp;
	fscanf(input, "%lf", &temp);
	i++;
    }
 
    fclose(input);


    //reading in model_2
    input = fopen("model_two", "r");
    if(input == NULL) {
        printf("Error, file not found\n");
	free(Model1);
	return(1);
    }

    fscanf(input, "%d %d %d", &Nx_2, &Ny_2, &Nz_2);
    temp_size = Nx_2*Ny_2*Nz_2;

    //192x192x14x14
    Model2 = (double*)malloc(temp_size*sizeof(double));


    i = 0;
    fscanf(input, "%lf", &temp);
    while(i < temp_size) {
        Model2[i] = temp;
	fscanf(input, "%lf", &temp);
	i++;
    }

    fclose(input);


    //reading in model_base
    input = fopen("model_base", "r");
    if(input == NULL) {
	free(Model1);
	free(Model2);
	return(1);
    }

    fscanf(input, "%d %d %d", &Nx_base, &Ny_base, &Nz_base);
    temp_size = Nx_base*Ny_base*Nz_base;

    //192x192x14x14
    Model_base = (double*)malloc(temp_size*sizeof(double));

    i = 0;
    fscanf(input, "%lf", &temp);
    while(i < temp_size) {
        Model_base[i] = temp;
	fscanf(input, "%lf", &temp);
	i++;
    }

    fclose(input);
    
    printf("All data of three modes read in.\n");

    //   
}

