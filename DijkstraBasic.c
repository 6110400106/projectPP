#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>

int minDistance(int V, int dist[], bool sSet[]) {
    int min = INT_MAX, min_index; 
  
    for (int v = 0; v < V; v++) 
        if (sSet[v] == false && dist[v] <= min) 
            min = dist[v], min_index = v; 
  
    return min_index;
}

void printPath(int parent[], int j) { 
    // Base Case : If j is source 
    if (parent[j] == - 1) 
        return; 
  
    printPath(parent, parent[j]); 
  
    printf("%d ", j); 
} 

void printSolution(int dist[], int V, int parent[]) {
    int src = 0; 
    printf("Vertex\t Distance\tPath"); 
    for (int i = 1; i < V; i++) { 
        printf("\n%d -> %d \t\t %d\t%d ", src, i, dist[i], src); 
        printPath(parent, i); 
    } 
}

int dijkstraAlgo(int V, int graph[V][V], int src) {
    int dist[V];
    bool sSet[V];
    int parent[V];

    parent[0] = -1;
    for(int i = 0; i < V; i++) {
        dist[i] = INT_MAX; 
        sSet[i] = false; 
    }
    dist[src] = 0;

    for(int i = 0; i < V-1; i++) {
        int u = minDistance(V, dist, sSet);
        sSet[u] = true;

        for(int j = 0; j < V; j++) {
            if(!sSet[j] && graph[i][j] && dist[i] + graph[i][j] < dist[j]) {
                parent[j] = i; 
                dist[j] = dist[i] + graph[i][j]; 
            }
        }
    }

    printSolution(dist, V, parent);
}

int main() {
    // To measure runtime
    clock_t t;

    int V = 6;                 // number of vertices, edge, and weight of each edge
    int graph[6][6] = { {0, 3, 60, 3, 16, 17},
                        {3, 0, 88, 88, 86, 26},
                        {60, 88, 0, 11, 59, 11},
                        {3, 88, 11, 0, 22, 51},
                        {16, 86, 59, 22, 0, 45},
                        {17, 26, 11, 51, 45, 0} };

    int start, end;
    t = clock();
    dijkstraAlgo(V, graph, 0);


    // for printing graph
    //printf("\n");
    //for(int i = 0; i < V; i++) {
    //    for(int j = 0; j < V; j++) 
    //        printf("%d ", graph[i][j]);
    //    printf("\n");
    //}

    t = clock() - t;
    printf ("\nIt took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
    printf("Time used: %f milliseconds\n", (((float)t)/CLOCKS_PER_SEC)*1000);
    return 0;
}