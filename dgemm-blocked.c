#include <x86intrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

const char* dgemm_desc = "Yiwen's Optimized blocked dgemm.";

#if !defined(BLOCK_SIZE)
  #define BLOCK_SIZE 32
#endif

#define min(a,b) (((a) < (b))? (a) : (b))

//A takes submatrix of 8x4  B takes 4x4 C takes 8x4
static void compute_8x4(double* A, double* B, double* C, int M, int N, int K, int lda){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  __m256d ci0, ci1, ci2, ci3, ci40, ci41, ci42, ci43;
  __m256d bk0, bk1, bk2, bk3;
  __m256d aik, ai4k;
  for(int i = 0; i <= M - 8; i += 8){
    for(int j = 0; j <= N - 4; j += 4){
      double* C_ij = C + j * lda + i;
      ci0 = _mm256_load_pd(C_ij);
      ci1 = _mm256_load_pd(C_ij + lda);
      ci2 = _mm256_load_pd(C_ij + lda * 2);
      ci3 = _mm256_load_pd(C_ij + lda * 3);
      ci40 = _mm256_load_pd(C_ij + 4);
      ci41 = _mm256_load_pd(C_ij + lda + 4);
      ci42 = _mm256_load_pd(C_ij + lda * 2 + 4);
      ci43 = _mm256_load_pd(C_ij + lda * 3 + 4);

      for(int k = 0; k < K; k ++){
        aik = _mm256_load_pd(A + k * lda + i);
        ai4k = _mm256_load_pd(A + k * lda + i + 4);

        double* B_kj = B + j * lda + k;

        bk0 = _mm256_set1_pd(*B_kj);
        bk1 = _mm256_set1_pd(*(B_kj + lda));
        bk2 = _mm256_set1_pd(*(B_kj + lda * 2));
        bk3 = _mm256_set1_pd(*(B_kj + lda * 3));

        ci0 = _mm256_fmadd_pd(aik, bk0, ci0);
        ci1 = _mm256_fmadd_pd(aik, bk1, ci1);
        ci2 = _mm256_fmadd_pd(aik, bk2, ci2);
        ci3 = _mm256_fmadd_pd(aik, bk3, ci3);
        ci40 = _mm256_fmadd_pd(ai4k, bk0, ci40);
        ci41 = _mm256_fmadd_pd(ai4k, bk1, ci41);
        ci42 = _mm256_fmadd_pd(ai4k, bk2, ci42);
        ci43 = _mm256_fmadd_pd(ai4k, bk3, ci43);
      }

      _mm256_store_pd(C_ij, ci0);
      _mm256_store_pd(C_ij + lda, ci1);
      _mm256_store_pd(C_ij + lda * 2, ci2);
      _mm256_store_pd(C_ij + lda * 3, ci3);
      _mm256_store_pd(C_ij + 4, ci40);
      _mm256_store_pd(C_ij + lda + 4, ci41);
      _mm256_store_pd(C_ij + lda * 2 + 4, ci42);
      _mm256_store_pd(C_ij + lda * 3 + 4, ci43);
    }

    for(int j = (N / 4) * 4; j < N; j ++){
      ci0 = _mm256_load_pd(C + j * lda + i);
      ci40 = _mm256_load_pd(C + j * lda + i + 4);
      for(int k = 0; k < K; k ++){
        aik = _mm256_load_pd(A + k * lda + i); 
        ai4k = _mm256_load_pd(A + k * lda + i + 4);

        bk0 = _mm256_set1_pd(B[j * lda + k]);

        ci0 = _mm256_fmadd_pd(aik, bk0, ci0);
        ci40 = _mm256_fmadd_pd(ai4k, bk0, ci40);
      }
      _mm256_store_pd(C + j * lda + i, ci0);
      _mm256_store_pd(C + j * lda + i + 4, ci40);
    }
  }
}

//A takes submatrix of 4x4  B takes 4x4 C takes 4x4
static void compute_4x4(double* A, double* B, double* C, int M, int N, int K, int lda){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  __m256d ci0, ci1, ci2, ci3;
  __m256d bk0, bk1, bk2, bk3;
  __m256d aik; //local or global
  for(int i = M / 8 * 8; i <= M - 4; i += 4){
    for(int j = 0; j <= N - 4; j += 4){
      double* C_ij = C + j * lda + i;
      ci0 = _mm256_load_pd(C_ij);
      ci1 = _mm256_load_pd(C_ij + lda);
      ci2 = _mm256_load_pd(C_ij + lda * 2);
      ci3 = _mm256_load_pd(C_ij + lda * 3);

      for(int k = 0; k < K; k ++){
        aik = _mm256_load_pd(A + k * lda + i);

        double* B_kj = B + j * lda + k;

        bk0 = _mm256_set1_pd(*B_kj);
        bk1 = _mm256_set1_pd(*(B_kj + lda));
        bk2 = _mm256_set1_pd(*(B_kj + lda * 2));
        bk3 = _mm256_set1_pd(*(B_kj + lda * 3));

        ci0 = _mm256_fmadd_pd(aik, bk0, ci0);
        ci1 = _mm256_fmadd_pd(aik, bk1, ci1);
        ci2 = _mm256_fmadd_pd(aik, bk2, ci2);
        ci3 = _mm256_fmadd_pd(aik, bk3, ci3);
      }
      _mm256_store_pd(C_ij, ci0);
      _mm256_store_pd(C_ij + lda, ci1);
      _mm256_store_pd(C_ij + lda * 2, ci2);
      _mm256_store_pd(C_ij + lda * 3, ci3);
    }

    for(int j = (N / 4) * 4; j < N; j ++){
      ci0 = _mm256_load_pd(C + j * lda + i);
      for(int k = 0; k < K; k ++){
        aik = _mm256_load_pd(A + k * lda + i); 
        bk0 = _mm256_set1_pd(B[j * lda + k]);
        ci0 = _mm256_fmadd_pd(aik, bk0, ci0);
      }
      _mm256_store_pd(C + j * lda + i, ci0);
    }
  }
}

//A takes submatrix of 2x2  B takes 2x2 C takes 2x2
static void compute_2x2(double* A, double* B, double* C, int M, int N, int K, int lda){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  __m128d ci0, ci1, ci2, ci3;
  __m128d bk0, bk1, bk2, bk3;
  __m128d aik; //local or global
  for(int i = (M / 4)* 4; i <= M - 2; i += 2){
    for(int j = 0; j <= N - 4; j += 4){
      double* C_ij = C + j * lda + i;
      ci0 = _mm_load_pd(C_ij);
      ci1 = _mm_load_pd(C_ij + lda);
      ci2 = _mm_load_pd(C_ij + lda * 2);
      ci3 = _mm_load_pd(C_ij + lda * 3);

      for(int k = 0; k < K; k ++){
        aik = _mm_load_pd(A + k * lda + i);

        double* B_kj = B + j * lda + k;
        bk0 = _mm_set1_pd(*(B_kj));
        bk1 = _mm_set1_pd(*(B_kj + lda));
        bk2 = _mm_set1_pd(*(B_kj + lda * 2));
        bk3 = _mm_set1_pd(*(B_kj + lda * 3));

        ci0 = _mm_fmadd_pd(aik, bk0, ci0);
        ci1 = _mm_fmadd_pd(aik, bk1, ci1);
        ci2 = _mm_fmadd_pd(aik, bk2, ci2);
        ci3 = _mm_fmadd_pd(aik, bk3, ci3);
      }
      _mm_store_pd(C_ij, ci0);
      _mm_store_pd(C_ij + lda, ci1);
      _mm_store_pd(C_ij + lda * 2, ci2);
      _mm_store_pd(C_ij + lda * 3, ci3);
    }

    for(int j = (N / 4) * 4; j < N; j ++){
      ci0 = _mm_load_pd(C + j * lda + i);
      for(int k = 0; k < K; k ++){
        aik = _mm_load_pd(A + k * lda + i); 
        bk0 = _mm_set1_pd(B[j * lda + k]);
        ci0 = _mm_fmadd_pd(aik, bk0, ci0);
      }
      _mm_store_pd(C + j * lda + i, ci0);
    }
  }

}

//A takes submatrix of 1x*  B takes 1 x lda C takes 1 x lda
static void compute_naive(double* A, double* B, double* C, int M, int N, int K, int lda){
  for (int i = (M / 2)* 2; i < M; i++) {
    for (int j = 0; j < N; ++j) {
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k) {
        cij += A[i+k*lda] * B[k+j*lda];
      }
      C[i+j*lda] = cij;
    }
  }
}

static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  compute_8x4(A, B, C, M, N, K, lda);
  compute_4x4(A, B, C, M, N, K, lda);
  compute_2x2(A, B, C, M, N, K, lda);
  compute_naive(A, B, C, M, N, K, lda);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
    //   fp = fopen("matrix.txt","w+");
    // double* ptrA = A;
    // fprintf(fp, "A = \n");
    // for(int mm = 0; mm < lda; mm ++){
    //   for(int nn = 0; nn < lda; nn ++){
    //     fprintf(fp, "%.2f\t", ptrA[nn * lda + mm]);
    //   }
    //   fprintf(fp, "\n");
    // } 
  // double* weird_A = weird_transformation(A, lda, STRIDE);

  //   ptrA = weird_A;
  //   fprintf(fp, "\n weird_A = \n");
  //   for(int mm = 0; mm < ceil( (double)lda / STRIDE) * lda; mm ++){
  //     for(int nn = 0; nn < STRIDE; nn ++){
  //       fprintf(fp, "%.2f\t", ptrA[mm * STRIDE + nn]);
  //     }
  //     fprintf(fp, "\n");
  //   }
    // ptrA = B;
    // fprintf(fp, "\nB = \n");
    // for(int mm = 0; mm < lda; mm ++){
    //   for(int nn = 0; nn < lda; nn ++){
    //     fprintf(fp, "%.2f\t", ptrA[nn * lda + mm]);
    //   }
    //   fprintf(fp, "\n");
    // } 

    // ptrA = C;
    // fprintf(fp, "\nC = \n");
    // for(int mm = 0; mm < lda; mm ++){
    //   for(int nn = 0; nn < lda; nn ++){
    //     fprintf(fp, "%.2f\t", ptrA[nn * lda + mm]);
    //   }
    //   fprintf(fp, "\n");
    // } 

  // For each block-row of A
  for (int i = 0; i < lda; i += BLOCK_SIZE) {
    // For each block-column of B
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
      // Accumulate block dgemms into block of C
      for (int k = 0; k < lda; k += BLOCK_SIZE) {
        // Correct block dimensions if block "goes off edge of" the matrix
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);
        // Perform individual block dgemm
        do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }

    // ptrA = C;
    // fprintf(fp, "\nResult C = \n");
    // for(int mm = 0; mm < lda; mm ++){
    //   for(int nn = 0; nn < lda; nn ++){
    //     fprintf(fp, "%.2f\t", ptrA[nn * lda + mm]);
    //   }
    //   fprintf(fp, "\n");
    // } 

}
