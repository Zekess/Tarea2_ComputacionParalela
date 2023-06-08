#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cooperative_groups.h>


using namespace std;
using namespace cooperative_groups;

__device__ int reduce_max(thread_group g, int *temp, int val) {
    int tid = g.thread_rank();
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        temp[tid] = val;
	g.sync();
        if(tid<i) val = max(temp[tid], temp[tid + i]);
	g.sync();
    }
    return val; // note: only thread 0 will return full sum
}

__global__ void maximo(int *max, int *input, int k, int *parcial)
{
    extern __shared__ int temp[];
    grid_group grid = this_grid();
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    thread_group g = this_thread_block();

    int block_max = reduce_max(g,temp, input[blockIdx.y*k + id]);

    if (threadIdx.x == 0) parcial[blockIdx.y*gridDim.x + blockIdx.x] = block_max;
    grid.sync();
    int max_total;
    thread_group tile32 = tiled_partition(g,32);
    if(blockIdx.x == 0 && threadIdx.x<32){
	max_total = reduce_max(tile32, temp, parcial[blockIdx.y*gridDim.x + id]);
    }
    if(blockIdx.x == 0 && threadIdx.x == 0){
	max[blockIdx.y] = max_total;
    }

}

void print(int *in, int N){
        for(int i=0; i<N; i++)
                printf("%d ", in[i]);
        printf("\n");
}


int main(int argc, char *argv[]){
	if(argc != 4){
		cout<<"USO: "<< argv[0] <<" N_bits k_bits blockSize\n";
		cout<<"N_bits: Log2 de cantidad de listas.\n";
		cout<<"k_bits: Log2 de tamaño de las listas.\n";
		cout<<"blockSize: Threads por bloque.\n";
		return 1;
	}
	int N = 1 << atoi(argv[1]);
	int k = 1 << atoi(argv[2]);
	int blockSize = atoi(argv[3]);
	int nBlocks = (k + blockSize - 1)/blockSize;

	//int N = 1<<24;
	//int k = 1<<6;
	//int nBlocks_x_lista = ;
	//int blockSize = 256;

	printf("%i listas de tamaño %i cada una. Utilizando %i hebras por bloque.\n", N, k, blockSize);
	int *host_Nlistas, *device_Nlistas;
	int *host_max, *device_max;

	printf("CHECK1");
	host_Nlistas = (int*)malloc(N * k * sizeof(int));
     	cudaMalloc(&device_Nlistas, N * k * sizeof(int));
	host_max = (int*)malloc(N * sizeof(int));
	cudaMalloc(&device_max, N * sizeof(int));

	printf("CHECK2");
	int *parcial;
     	cudaMalloc(&parcial, nBlocks * sizeof(int));

     	//cudaMemset(device_max, 0, N*sizeof(int));

	// Rellenar N listas con k elementos (Usar rand(?))
	for (int i = 0; i < N; i++){
		for (int j = 0; j < k; j++){
			// i*k + j --> elemento j de la lista i.
			host_Nlistas[i*k+j] = i+j;
		}
	}

	
	for (int i=0; i<N; i++){
		host_max[i] = host_Nlistas[i*k];
		//printf("%i\n", host_max[i]);
	}
	
	int sharedMemory = blockSize * sizeof(int);


	int dim_x = (k + blockSize - 1) / blockSize; //nBlocks x lista
	int dim_y = N; //nListas

	dim3 gridDim_x_stream(dim_x, dim_y);

	
	void *args[] = { &device_max, &device_Nlistas, &k, &parcial };
	dim3 gd(dim_x, dim_y);
	dim3 bd = dim3(blockSize,1,1);

        auto st_time = std::chrono::high_resolution_clock::now();
	//cudaError_t res = maximo<<<gd, bd, sharedMemory>>>(device_max, device_Nlistas, k, parcial);
        cudaError_t res = cudaLaunchKernel((void *)maximo, gd, bd, args, sharedMemory);
	if (res != cudaSuccess) {
        	printf ("error en kernel launch: %s \n", cudaGetErrorString(res));
        	return -1;
    	}
	//reduce<<<nBlocks, blockSize, sharedBytes>>>(sum, data,parcial);
	cudaDeviceSynchronize();
        auto e_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ttime = e_time - st_time;
        std::cout<<" tiempo kernel "<<ttime.count()<<"s"<<std::endl;

	cudaMemcpy(&host_max, &device_max, N*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0;i<10;i++){
		std::cout << host_max[N-i-1] << std::endl;
	}
	//cout<<" sum "<<*sum<<endl;	
	cudaFree(device_Nlistas);
	cudaFree(device_max);
	cudaFree(parcial);


}
