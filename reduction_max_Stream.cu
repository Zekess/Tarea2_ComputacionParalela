#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<cuda_runtime.h>

using namespace std;

__device__ int reduce_max(int *temp, int val, int k) {
	int tid = threadIdx.x;
	int tam = min(k, blockDim.x);
	for (int i = tam / 2; i > 0; i /= 2)
	{
		temp[tid] = val;
		__syncthreads();
		if(tid<i) val = max(temp[tid],temp[tid+i]); //+= temp[tid + i];
		__syncthreads();
	}
	return val; // note: only thread 0 will return full sum
}

__global__ void maximo(int *max, int *input, int k, int offset)
{
	extern __shared__ int temp[];
	int id_lista = blockIdx.y + offset;
	int id_hebra = blockIdx.x * blockDim.x + threadIdx.x;
	//input[id_lista] corresponde a la lista id_lista.
	//temp[id_lista] es la matriz temporal auxiliar que 
	// almacena en temp[i][j] el máximo de la lista i bloque j.
	if (id_hebra < k){
		int block_max = reduce_max(temp, input[id_lista*k + id_hebra], k);
		if (threadIdx.x == 0){
			atomicMax(&max[id_lista], block_max);
		}
	}
}


int main(int argc, char *argv[]){

	if(argc != 4){
		cout<<" USO: "<< argv[0] <<" N_bits k_bits blockSize\n";
		cout<<"N_bits: Log2 de cantidad de listas.\n";
		cout<<"k_bits: Log2 de tamaño de las listas.\n";
		cout<<"blockSize: Threads por bloque.\n";
		return 1;
	}
	int N = 1 << atoi(argv[1]);
	int k = 1 << atoi(argv[2]);
	int blockSize = atoi(argv[3]);

	//int N = 1<<24;
	//int k = 1<<6;
	//int nBlocks_x_lista = ;
	//int blockSize = 256;

	printf("%i listas de tamaño %i cada una. Utilizando %i hebras por bloque.\n", N, k, blockSize);
	int *host_Nlistas, *device_Nlistas;
	int *host_max, *device_max;

	host_Nlistas = (int*)malloc(N * k * sizeof(int));
     	cudaMalloc(&device_Nlistas, N * k * sizeof(int));
	host_max = (int*)malloc(N * sizeof(int));
	cudaMalloc(&device_max, N * sizeof(int));

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
	

	int nStreams = 16;
	int streamSize = N * k / nStreams;
	int streamSizeBytes = streamSize * sizeof(int);
	int sharedMemory = blockSize * sizeof(int);


	int dim_x = (k + blockSize - 1) / blockSize; //nBlocks x lista
	int dim_y = (N / nStreams); //nListas por Stream
	dim3 gridDim_x_stream(dim_x, dim_y);

	cudaStream_t stream[nStreams];
	for (int i = 0; i < nStreams; i ++) {
		cudaStreamCreate(&stream[i]);
	}
	//printf("BlockSize: %d, streamSize: %d, GridDim: (%d, %d)\n", BlockSize, streamSize, dim_x, dim_y);

	for (int i = 0; i < nStreams; i ++) {
		//offset en la dimensión Y de la grilla
		int offset = i * streamSize;
		int offset_max = i * dim_y;
		cudaMemcpyAsync(&device_Nlistas[offset], &host_Nlistas[offset], streamSizeBytes, cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(&device_max[offset_max], &host_max[offset_max], dim_y*sizeof(int), cudaMemcpyHostToDevice, stream[i]);
		maximo<<<gridDim_x_stream, blockSize, sharedMemory, stream[i]>>>(device_max, device_Nlistas, k, offset_max);
		cudaMemcpyAsync(&host_max[offset_max], &device_max[offset_max], dim_y*sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
	}
	cudaDeviceSynchronize();

	for (int i = 0; i < 10; i++) {
		printf("Lista %i: %i\n", i, host_max[N-i-1]);
	}

	cudaFree(device_Nlistas);
	cudaFreeHost(host_Nlistas);
	cudaFreeHost(host_max);
	cudaFree(device_max);
	for (int i = 0; i < nStreams; i ++) {
		cudaStreamDestroy(stream[i]);
	}

	return 0;
}
