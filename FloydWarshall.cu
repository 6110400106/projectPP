#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define NV 5                                // number of vertices

void createGraph(float *arr, int N) {
    time_t t;                               // used for randomizing values
    int col; 
    int row;
    int maxWeight = 100;                    // limit the weight an edge can have

    srand((unsigned) time(&t));             // generate random

    for (col = 0; col < sqrt(N); col++) { 
        for(row = 0; row < sqrt(N); row++) {
            if( col != row){
                arr[(int)(row*sqrt(N)) + col] = rand() % maxWeight; // assign random

                // have a symmetric graph
                arr[(int)(col*sqrt(N)) + row] = arr[(int)(row*sqrt(N)) + col];
            }
            else
                arr[(int)(row*sqrt(N)) + col] = 0; // NO LOOPS
        }
    }
}

void printGraph(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f      ", arr[i * n + j]);
        }
        printf("\n");
    }
}

__global__ void gpuFloyd(int n, float* arr, int k) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= n) {
        return;
    }
 
    int idx = n * blockIdx.y + gid;
    __shared__  int shortest_distance;

    if(tid == 0) {
        shortest_distance = arr[n * blockIdx.y + k];
    }

    __syncthreads();

    int node_distance = arr[k * n + gid];
    int total_distance = shortest_distance + node_distance;
    if (arr[idx] > total_distance){
       arr[idx] = total_distance;
    }

    __syncthreads();

}

void cpuFloyd(int n, float* cpuGraph) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (cpuGraph[i * n + j] > (cpuGraph[i * n + k] + cpuGraph[k * n + j])) {
                    cpuGraph[i * n + j] = cpuGraph[i * n + k] + cpuGraph[k * n + j];
                } 
            }
        }
    }
}

void valid(int n, float* cpuGraph, float* gpuGraph) {
    printf("VALIDATING THAT cpuGraph array from CPU and gpuGraph array from GPU match... \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (cpuGraph[i * n + j] != gpuGraph[i * n + j]) {
                printf("ERROR MISMATCH in array cpuGraph i %d j %d CPU SAYS %f and GPU SAYS %f \n", i, j, cpuGraph[i * n + j], gpuGraph[i * n + j]);
            }
        }
    }
    printf("OK \n\n");
}

int main(int argc, char **argv) {
    clock_t t;
    float *hostArr, *gpuGraph;
    float *devArr;
    float *graph, *cpuGraph;
    
    int i, j, bk;
    int n = NV;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\n");
    printf("RUNNING WITH %d VERTICES \n", n);
    printf("\n");

    cudaMalloc(&devArr, n * n * sizeof (float));

    //CPU arrays
    graph = (float *) malloc(n * n * sizeof (float)); 
    cpuGraph = (float *) malloc(n * n * sizeof (float)); 

    //GPU arrays
    hostArr = (float *) malloc(n * n * sizeof (float));
    gpuGraph = (float *) malloc(n * n * sizeof (float));

    // Randomize distances in between each node
    createGraph(graph, (n*n)); 

    // Printing graph
    //printGraph(graph, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            cpuGraph[i * n + j] = graph[i * n + j];
            hostArr[i * n + j] = graph[i * n + j];
        }
    }

    // First Mem Copy
    cudaMemcpy(devArr, hostArr, n * n * sizeof (float), cudaMemcpyHostToDevice);

    // For GPU Calculation
    int gputhreads = 512;

    // Kernel call
    // dim3 dimGrid(n, n, 1);
    dim3 dimGrid((n + gputhreads - 1) / gputhreads, n);  
    cudaEventRecord(start); 
    for(int k = 0; k < n; k++) {
        gpuFloyd<<<dimGrid, gputhreads>>>(n, devArr, k);
    }
    cudaEventRecord(stop);

    // Second Mem Copy
    cudaMemcpy(gpuGraph, devArr, n * n * sizeof (float), cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Calculation Time elapsed: %.20f milliseconds\n", milliseconds);
    
    // CPU calculation
    t = clock();
    cpuFloyd(n, cpuGraph);

    t = clock() - t;
    printf("CPU Calculation Time elapsed: %.20f milliseconds\n\n", (((float)t)/CLOCKS_PER_SEC)*1000);

    // Check validation of cpuGraph array from CPU calc and gpuGraph array from GPU calc
    // See if the two arrays match
    valid(n, cpuGraph, gpuGraph);

    /*
    printf("Graph from GPU:\n");
    printGraph(gpuGraph, n);
    printf("\n");

    printf("Graph from CPU:\n");
    printGraph(cpuGraph, n);
    printf("\n");
    */

    cudaFree(devArr);

    free(graph);
    free(cpuGraph);
    free(hostArr);
    free(gpuGraph);

    printf("FINISHED!!! \n");
    return 0;
}