#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#define V 6

// __host__ __device__ void printPath(int *parent, int j) { 
//     // Base Case : If j is source 
//     if (parent[j] == - 1) 
//         return; 
  
//     printPath(parent, parent[j]); 
  
//     printf("%d ", j); 
// } 

__global__ void dijkstraAlgo(int graph[V][V]) {
    int src = 0;
    int dist[V];
    bool sSet[V];
    int parent[V];

    // parent[0] = -1;
    // for(int i = 0; i < V; i++) {
    //     dist[i] = INT_MAX; 
    //     sSet[i] = false; 
    // }
    // dist[src] = 0;

    // int min = INT_MAX, min_index; 
    // for(int i = 0; i < V-1; i++) {
    //     // find min index
    //     for (int v = 0; v < V; v++) 
    //         if (sSet[v] == false && dist[v] <= min) 
    //             min = dist[v], min_index = v; 
    //     int u = min_index;

    //     sSet[u] = true;

    //     for(int j = 0; j < V; j++) {
    //         if(!sSet[j] && graph[i][j] && dist[i] + graph[i][j] < dist[j]) {
    //             parent[j] = i; 
    //             dist[j] = dist[i] + graph[i][j]; 
    //         }
    //     }
    // }

    // printf("Vertex\t Distance\tPath"); 
    // for (int i = 1; i < V; i++) { 
    //     printf("\n%d -> %d \t\t %d\t%d ", src, i, dist[i], src); 
    //     printPath(parent, i); 
    // }
}

int main() {
    int graph[V][V] = { {0, 4, 4, 0, 0, 0},
                        {4, 0, 2, 0, 0, 0},
                        {4, 2, 0, 3, 1, 6},
                        {0, 0, 3, 0, 0, 2},
                        {0, 0, 1, 0, 0, 3},
                        {0, 0, 6, 2, 3, 0} };
	int h_out[V][V];
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *d_in;
    cudaMalloc((void**) &d_in, V*sizeof(int));
	cudaMemcpy(d_in, &graph, V*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    dijkstraAlgo<<<1, V>>>(graph);
    cudaEventRecord(stop);

    cudaMemcpy(&h_out, d_in, V*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaFree(d_in);

	printf("Time used: %f milliseconds\n", milliseconds);

    return 0;
}