/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Azzam Haidar
       @author Stan Tomov
       
       @generated from magmablas/zlatanh.cu, normal z -> d, Tue Jul 18 11:35:57 2017

*/

#include "magma.h"

// BLK_X and BLK_Y need to be equal for dlaset_q to deal with diag & offdiag
// when looping over super blocks.
// Formerly, BLK_X and BLK_Y could be different.
#define BLK_X 64
#define BLK_Y BLK_X

#define MAGMA_D_TANH(x) tanh(x)

/******************************************************************************/
/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to dlaset, dlacpy, dlag2s, clag2z, dgeadd.
*/
__global__
void dlatanh_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = MAGMA_D_TANH(dA[j*ldda]);
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = MAGMA_D_TANH(dA[j*ldda]);
            }
        }
    }
}


/******************************************************************************

    Purpose
    -------
    DLATANH initialized dB by taking tanh of the corresponding element of 
    matrix dA.
    
    Arguments
    ---------
    @param[in]
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA      DOUBLE PRECISION array, dimension (LDDA,N)
            The M-by-N matrix dA.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[out]
    dB      DOUBLE PRECISION array, dimension (LDDB,N)
            The M-by-N matrix dB.
            On exit, dB = .tanh(dA)
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    
    @ingroup magma_ml
*******************************************************************************/
extern "C" void
magmablas_dlatanh(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const dA, magma_int_t ldda,
    magmaDouble_ptr             dB, magma_int_t lddb)
{

    if ( m == 0 || n == 0 ) {
        return;
    }

    dim3 threads( BLK_X, 1, 1 );
    dim3 grid( magma_ceildiv( m, BLK_X ), magma_ceildiv( n, BLK_Y ), 1 );

    dlatanh_kernel
        <<< grid, threads, 0, NULL >>>
        ( m, n, dA, ldda, dB, lddb );
}

