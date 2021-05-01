#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define inf 9999
#define N 5
#define tolerance 0.001

__global__ void floyd(int n, int k, float* x, int* qx) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int j = ix & (n - 1);
    float temp2 = x[ix - j + k] + x[k * n + j];
    if (x[ix] > temp2) {
        x[ix] = temp2;
        qx[ix] = k;
    }
    if (x[ix] == inf) {
        qx[ix] = -2;
    }
}

__global__ void floyd2(int n, int k, float* x, int* qx) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int j = ix & (n - 1);
    float temp2 = x[ix - j + k] + x[k * n + j];
    if (x[ix] > temp2) {
        x[ix] = temp2;
        qx[ix] = k;
    }
}

void cpu_floyd(int n, float* D, int* Q) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((D[i * n + k] + D[k * n + j]) < D[i * n + j]) {
                    D[i * n + j] = D[i * n + k] + D[k * n + j];
                    Q[i * n + j] = k;
                } 
                if (D[i * n + j] == inf) {
                    Q[i*n+j]=-2;
                }
            }
        }
    }
}

void valid(int n, float* D, float* host_D) {
    printf("VALIDATING THAT D array from CPU and host_D array from GPU match... \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // if (abs(D[i * n + j] - host_D[i * n + j]) > tolerance) {
            //     printf("ERROR MISMATCH in array D i %d j %d CPU SAYS %f and GPU SAYS %f \n", i, j, D[i * n + j], host_D[i * n + j]);
            // }
            if (D[i * n + j] != host_D[i * n + j]) {
                printf("ERROR MISMATCH in array D i %d j %d CPU SAYS %f and GPU SAYS %f \n", i, j, D[i * n + j], host_D[i * n + j]);
            }
        }
    }
    printf("OK \n");
}

int main(int argc, char **argv) {
    float *host_A, *host_D;
    int *host_Q;
    float *dev_x;
    int *dev_qx;
    float *A, *D;
    int *Q;
    
    int i, j, bk;
    int k = 0;
    int n = N;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\n");
    printf("RUNNING WITH %d VERTICES \n", n);
    printf("\n");

    cudaMalloc(&dev_x, n * n * sizeof (float));
    cudaMalloc(&dev_qx, n * n * sizeof (float));

    //CPU arrays
    A = (float *) malloc(n * n * sizeof (float)); 
    D = (float *) malloc(n * n * sizeof (float)); 
    Q = (int *) malloc(n * n * sizeof (int)); 

    //GPU arrays
    host_A = (float *) malloc(n * n * sizeof (float));
    host_D = (float *) malloc(n * n * sizeof (float));
    host_Q = (int *) malloc(n * n * sizeof (int));

    // Randomize distances in between each node
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = 0;
            } else {
                A[i * n + j] = 1200 * (float) rand() / RAND_MAX + 1;
                if (A[i * n + j] > 1000) {
                    A[i * n + j] = inf;
                }
            }
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            Q[i * n + j] = -1;
            D[i * n + j] = A[i * n + j];
            host_A[i * n + j] = A[i * n + j];
            host_Q[i * n + j] = Q[i * n + j];
        }
    }

    // First Mem Copy
    cudaMemcpy(dev_x, host_A, n * n * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_qx, host_Q, n * n * sizeof (int), cudaMemcpyHostToDevice);

    // GPU Calculation
    bk = (int) (n * n / 512);
    int gputhreads = 512;
    if (bk > 0) {
        gputhreads = 512;
    } else {
        bk = 1;
        gputhreads = n*n;
    }
    printf(" \n");
    printf("BLOCKS :   %d      GPU THREADS:     %d \n", bk, gputhreads);
    printf(" \n");

    cudaEventRecord(start); 

    floyd<<<bk, gputhreads>>>(n, k, dev_x, dev_qx);
    for (k = 1; k < n; k++) 
        floyd2<<<bk, gputhreads>>>(n, k, dev_x, dev_qx);

    cudaEventRecord(stop);

    // Second Mem Copy
    cudaMemcpy(host_D, dev_x, n * n * sizeof (float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Q, dev_qx, n * n * sizeof (int), cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Calculation Time elapsed: %.20f\n", milliseconds * .0001);
    printf("\n");
    
    // CPU calculation
    cpu_floyd(n, D, Q);

    // Check validation of D array from CPU calc and host_D array from GPU calc
    // See if the two arrays match
    valid(n, D, host_D);

    cudaFree(dev_x);
    cudaFree(dev_qx);

    free(A);
    free(D);
    free(Q);
    free(host_A);
    free(host_D);
    free(host_Q);

    printf("ALL OK WE ARE DONE \n");
    return 0;
}