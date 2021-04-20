#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#define NV 6                                // number of vertices

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
void printGraph(float *arr, int size) {
    int index;
    printf("\nGraph:\n");
    for(index = 0; index < size; index++) {
        if(((index + 1) % (int)sqrt(size)) == 0) {
            printf("%5.1f\n", arr[index]);
        }
        else {
            printf("%5.1f ", arr[index]);
        }
    }
    printf("\n");
}
__global__ void dijkstraAlgo(float *graph, float *result, bool* visited, int V) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    visited[index] = false;
    
    if(index == ((blockDim.x * blockIdx.x) + blockIdx.x))
        result[index] = 0;                                  // distance to itself is always 0
    else
        result[index] = INT_MAX;
    for (int count = 0; count < V-1; count++) {
        // Pick the minimum distance vertex from the set of vertices not
        // yet processed.
        int min = INT_MAX, u;
        for (int v = 0; v < V; v++)
            if (visited[(V * blockIdx.x) + v] == false && result[(V *blockIdx.x) +  v] <= min)
                min = result[(V * blockIdx.x) + v], u = v;
    
        // Mark the picked vertex as processed
        visited[(V * blockIdx.x) + u] = true;
    
        // Update the wieght value 
        for (int v = 0; v < V; v++) {
    
            // Update only if is not in visited, there is an edge from 
            // u to v, and total weight of path from src to  v through u is 
            // smaller than current value
            if (!visited[(V * blockIdx.x) + v] && graph[(u*V) + v] && result[(V * blockIdx.x) + u] != INT_MAX
                && result[(V * blockIdx.x) + u] + graph[(u*V) + v] < result[(V * blockIdx.x) + v])
                result[(V * blockIdx.x) + v] = result[(V*blockIdx.x) + u] + graph[(u*V) + v];
        }
    }
}
int main() {
    float* graph = (float *) malloc((NV*NV) * sizeof(float));
    float* result = (float *) malloc((NV*NV) * sizeof(float));

    createGraph(graph, (NV*NV));                    // Generate the graph & store in array
    printGraph(graph, (NV*NV));                     // Print the array
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_graph, *d_result;
    bool *d_visited;

    cudaMalloc((void **) &d_graph, ((NV*NV) * sizeof(float)));
    cudaMalloc((void **) &d_result, ((NV*NV) * sizeof(float)));
    cudaMalloc((void **) &d_visited, ((NV*NV) * sizeof(bool)));

    cudaMemcpy(d_graph, graph, ((NV*NV) * sizeof(float)), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    dijkstraAlgo<<<NV, 1>>>(d_graph,d_result, d_visited, NV);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(result, d_result, ((NV*NV) * sizeof(float)), cudaMemcpyDeviceToHost);
    
    printGraph(result, (NV*NV));
    
    cudaFree(d_graph);
    cudaFree(d_result);
    cudaFree(d_visited);

    free(graph);
    free(result);

    printf("Time used: %f milliseconds\n", milliseconds);
    return 0;
}