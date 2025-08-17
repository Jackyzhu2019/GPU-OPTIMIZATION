#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "./common.h"
#include <stdint.h>

void Mimo64_alloc_host_mem(void** host_ptr_addr, size_t size)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    err = cudaMallocHost((void **)host_ptr_addr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host memory (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	return;
}

void Mimo64_free_host_mem(void* host_ptr)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    if (host_ptr != NULL)
    {
        err = cudaFreeHost(host_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to free host memory (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
	
	return;
}

void Mimo64_alloc_device_mem(void** dev_ptr_addr, size_t size)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
  
	err = cudaMalloc((void**)dev_ptr_addr, size);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate cuda device mem (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return;
}

void Mimo64_free_device_mem(void* dev_ptr)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    if (dev_ptr != NULL){
		err = cudaFree(dev_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to free cuda device mem (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	}

	return;
}


void initialData_f32(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF);
		//printf("val[%d]: %d \n", i, ip[i]);
    }
	
}


void Mimo64_init_device_const_mem()
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

 

	return;
}



cudaStream_t *streams;

void Mimo64_createStreams(int numOfStreams){

	streams = (cudaStream_t *)malloc(numOfStreams * sizeof(cudaStream_t));

	for (int i = 0; i < numOfStreams; i++)
		cudaStreamCreate(&streams[i]);

	return;
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %f gpu %f at %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}




void mimo64_naive_kernel(float *G, float *Y, float *X, int nElem, int nElemPerMatrix)
{
	int numOfloop = nElem / nElemPerMatrix;
	int loopIdx;
	int iRow, iCol, iK;
	float *G0, *Y0, *X0;
	
	for (loopIdx = 0; loopIdx < numOfloop; loopIdx++)
	{
		G0 = G + 64 * 64 * loopIdx;
		Y0 = Y + 64 * nElemPerMatrix * loopIdx;
		X0 = X + 64 * nElemPerMatrix * loopIdx;
		
		// G matrix (64 x 64) x Y matrix (64 x nElemPerMatrix(4))
		for (iRow = 0; iRow < 64; iRow++)
		{
			for (iCol = 0; iCol < nElemPerMatrix; iCol++)
			{
				float accuVal = 0.0f;
				
				for (iK = 0; iK < 64; iK++)
				{
					
					accuVal += G0[iRow * 64 + iK] * Y0[iK * nElemPerMatrix + iCol];
#if 0	
if (iRow < 1 && iCol < 1)
{
	printf("iRow: %d iCol: %d iK: %d accu: %f G0: %f Y0: %f \n", 
	iRow, iCol, iK, accuVal, G0[iRow * 64 + iK], Y0[iK * nElemPerMatrix + iCol]);
}
#endif	
				}
				
				X0[iRow * nElemPerMatrix + iCol] = accuVal;
			}
		}
	}


	return;
}

__global__ void mimo64_naive_gpu_kernel(float *G, float *Y, float *X, int nElem)
{
	int loopIdx;
	int iRow, iCol, iK;
	float *G0, *Y0, *X0;
	
	int M = 64;
	int K = 64;
	int N = 4;
	int nElemPerMatrix = 4;
	
	// inside a block
	int xIdx = threadIdx.x; 
	int yIdx = threadIdx.y;
	// block shape
	int xLen = blockDim.x;
	int yLen = blockDim.y;
	// block index
	int Block_X_Idx = blockIdx.x;
	int Block_Y_Idx = blockIdx.y; // 0
	
	int tid_x = Block_X_Idx * xLen + xIdx;
	int tid_y = Block_Y_Idx * yLen + yIdx;
	
	// printf("xIdx: %d yIdx: %d \n", xIdx, yIdx);
	
	G0 = G + 64 * 64 * Block_X_Idx;
	Y0 = Y + 64 * nElemPerMatrix * Block_X_Idx;
	X0 = X + 64 * nElemPerMatrix * Block_X_Idx;
	
	iRow = yIdx;
	iCol = xIdx;
	//for (iRow = 0; iRow < 64; iRow++)
	{
		//for (iCol = 0; iCol < nElemPerMatrix; iCol++)
		{
			float accuVal = 0.0f;
			
			for (iK = 0; iK < 64; iK++)
			{
				
				accuVal += G0[iRow * 64 + iK] * Y0[iK * nElemPerMatrix + iCol];
			
			}
			
			X0[iRow * nElemPerMatrix + iCol] = accuVal;
		}
	}
	
	// printf("tid: %d %d iRow: %d iCol: %d val: %f \n", tid_x, tid_y, iRow, iCol, X0[iRow * nElemPerMatrix + iCol]);

	return;
}

template <
    const int BLOCK_SIZE_M,  // height of block of X that each thread calculate, i.e. bm
    const int BLOCK_SIZE_K,  // width of block of G that each thread load into shared memory, i.e. bk
    const int BLOCK_SIZE_N  // width of block of X that each thread calculate i.e. bn
    > 
__global__ void mimo64_block_naive_gpu_kernel(float *G, float *Y, float *X, int nElem)
{	
	int i, j, k;
	
	// inside a block
	int xIdx = threadIdx.x; 
	int yIdx = threadIdx.y;
	// block shape
	int xLen = blockDim.x;
	int yLen = blockDim.y;
	// block index
	int Block_X_Idx = blockIdx.x;
	int Block_Y_Idx = blockIdx.y; // 0
	
	int tid_x = Block_X_Idx * xLen + xIdx;
	int tid_y = Block_Y_Idx * yLen + yIdx;

	// shared memory
	__shared__ float Gs[64][4]; // shared memory of G: 1 Kbyte
	__shared__ float Ys[64][4]; // shared memory of Y: 1 Kbyte
	
	float accu[BLOCK_SIZE_M][BLOCK_SIZE_N];

	// load data from global memory to shared memeory
	// load the whole Y 
	int sLoadIdx;
	// each thread load BLOCK_SIZE_K * BLOCK_SIZE_N
	#pragma unroll
	for (i = 0; i < BLOCK_SIZE_K; i++)
	{
	    #pragma unroll
		for (j = 0; j < BLOCK_SIZE_N; j++)
		{
			sLoadIdx = nElem * (tid_y * BLOCK_SIZE_K + i)
						+ tid_x * BLOCK_SIZE_N + j;

			Ys[yIdx * BLOCK_SIZE_K + i][j] = Y[sLoadIdx];
// printf("tid: %d %d ldIdx: %d row: %d col: %d Y: %f \n", tid_x, tid_y, 
//sLoadIdx, i, j, Ys[yIdx * BLOCK_SIZE_K + i][j]);

		}
	}
	
	int G_LEN_ONE_ROW = 64 * (nElem / 4);
	
	#pragma unroll
	for (i = 0; i < BLOCK_SIZE_M; i++)
	{
	    #pragma unroll
		for (j = 0; j < BLOCK_SIZE_K; j++)
		{
			sLoadIdx = G_LEN_ONE_ROW * (tid_y * BLOCK_SIZE_M + i)
						+ tid_x * BLOCK_SIZE_K + j;
			
			Gs[yIdx * BLOCK_SIZE_M + i][j] = G[sLoadIdx];
			
			//if (tid_x == 0 && tid_y == 0)
//printf("tid: %d %d ldIdx: %d row: %d col: %d G: %f Y: %f \n", tid_x, tid_y, sLoadIdx, i, j, Gs[i][j], Ys[i][j]);
		}
	}
		
	// reset the accumulation
	#pragma unroll
	for (i = 0; i < BLOCK_SIZE_M; i++)
	{
		#pragma unroll
		for (j = 0; j < BLOCK_SIZE_N; j++)
		{
				accu[i][j] = 0.0f;
		}
	}
		
	__syncthreads();
#if 0		
 	if ((tid_x == 0) && (tid_y == 0))
	{
		for (i = 0; i < 64; i++)
		{
			for (j = 0; j < 1; j++)
			{
				printf("i: %d j:%d  Y: %f \n", i, j,  Ys[i][j]);
			}
		}
	}
#endif	
	// load from shared memory to register
	float val_G[BLOCK_SIZE_M][BLOCK_SIZE_K];
	float val_Y[BLOCK_SIZE_K][BLOCK_SIZE_N];
	
	const int loopIdx = 64 / BLOCK_SIZE_K;
	int iTile = 1;
	while (iTile <= loopIdx)
	{
		#pragma unroll
		for (i = 0; i < BLOCK_SIZE_M; i++)
		{
			#pragma unroll
			for (j = 0; j < BLOCK_SIZE_K; j++)
			{
				val_G[i][j] = Gs[yIdx * BLOCK_SIZE_M + i][j];
			}
		}
		
		#pragma unroll
		for (i = 0; i < BLOCK_SIZE_K; i++)
		{
			#pragma unroll
			for (j = 0; j < BLOCK_SIZE_N; j++)
			{
				val_Y[i][j] = Ys[(iTile - 1) * BLOCK_SIZE_K + i][j];
#if 0
if (tid_x == 0 && tid_y == 0)
{
				printf("val: %f i: %d j: %d, ldIdx: %d \n",
					val_Y[i][j], i, j, yIdx * BLOCK_SIZE_K + i);
}	
#endif			
			}
		}
		


		// matrix multiply
		#pragma unroll
		for (i = 0; i < BLOCK_SIZE_M; i++)
		{
			#pragma unroll
			for (j = 0; j < BLOCK_SIZE_N; j++)
			{
				#pragma unroll
				for (k = 0; k < BLOCK_SIZE_K; k++)
				{
					accu[i][j] += val_G[i][k] * val_Y[k][j];
#if 0					
if (tid_x == 0 && tid_y == 0 && i == 0 && j == 0)
{
	printf("row: %d col: %d k: %d G: %f Y: %f, accu: %f \n", 
					i, j, k, val_G[i][k], val_Y[k][j], accu[i][j]);
}
#endif
				}
			}
		}
		
		int storeIdx;
		#pragma unroll
		for (i = 0; i < BLOCK_SIZE_M; i++)
		{
			#pragma unroll
			for (j = 0; j < BLOCK_SIZE_N; j++)
			{
				storeIdx = nElem * (tid_y * BLOCK_SIZE_M + i)
							+ tid_x * BLOCK_SIZE_N + j;
				X[storeIdx] = accu[i][j];
			}
		}
		
		if (iTile != loopIdx)
		{
			// next G data: BLOCK_SIZE_M * BLOCK_SIZE_K 
			#pragma unroll
			for (i = 0; i < BLOCK_SIZE_M; i++)
			{
				#pragma unroll
				for (j = 0; j < BLOCK_SIZE_K; j++)
				{
					sLoadIdx = G_LEN_ONE_ROW * (tid_y * BLOCK_SIZE_M + i)
								+ tid_x * BLOCK_SIZE_K + BLOCK_SIZE_K * iTile + j;
					
					Gs[yIdx * BLOCK_SIZE_M + i][j] = G[sLoadIdx];
				}
			}
				
			__syncthreads();
		}
		
		iTile++;
	}

	return;
}



#define NSTREAM 4
#define BDIM 128

int main(int argc, char **argv)
{
    printf("> %s Starting...\n", argv[0]);
	
    // set up data size of vectors
    int nElem = 273 * 12 * 14;
	const int nElemPerMatrix = 4;
    printf("> vector size = %d\n", nElem);

    float *G;
	float *Y;
	float *N0;
	float *X_base;
	float *X;
	
	//G = (float *)malloc((nElem / nElemPerMatrix) * 64 * 64 * sizeof(float));
	//Y = (float *)malloc(nElem * 64 * sizeof(float));
//	N0 = (float *)malloc(nElem * 64 * sizeof(float));
	//X = (float *)malloc(nElem * 64 * sizeof(float));
	//X_base = (float *)malloc(nElem * 64 * sizeof(float));

	//int16_t *h_ScaleLUT;
	Mimo64_alloc_host_mem((void **)&G, (nElem / nElemPerMatrix) * 64 * 64 * sizeof(float));
	Mimo64_alloc_host_mem((void **)&Y, nElem * 64 * sizeof(float));
	Mimo64_alloc_host_mem((void **)&X_base, nElem * 64 * sizeof(float)); 
	Mimo64_alloc_host_mem((void **)&X, nElem * 64 * sizeof(float));

	float *d_G;
	float *d_Y;
	float *d_X;

	Mimo64_alloc_device_mem((void **)&d_G, (nElem / nElemPerMatrix) * 64 * 64 * sizeof(float));
	Mimo64_alloc_device_mem((void **)&d_Y, nElem * 64 * sizeof(float));
	Mimo64_alloc_device_mem((void **)&d_X, nElem * 64 * sizeof(float));

    memset(X, 0, nElem * 64 * sizeof(float));
    memset(X_base,  0, nElem * 64 * sizeof(float));

 	initialData_f32(G, (nElem / nElemPerMatrix) * 64 * 64);
	initialData_f32(Y, nElem * 64);

	//mimo64_naive_kernel(G, Y, X_base, nElem, nElemPerMatrix);
	long t_start = useconds();

	mimo64_naive_kernel(G, Y, X_base, nElem, nElemPerMatrix);
	
	long t_end = useconds();
	printf("mimo64_naive_kernel() costs %ld us \n", (t_end - t_start) );

    float kernel_time;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    dim3 block (4, 64);
    dim3 grid  ((nElem + block.x - 1) / block.x);
	
	CHECK(cudaMemcpy(d_G, G, (nElem / nElemPerMatrix) * 64 * 64 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Y, Y, nElem * 64 * sizeof(float), cudaMemcpyHostToDevice));

  	CHECK(cudaEventRecord(start, 0));

    mimo64_naive_gpu_kernel<<<grid, block>>>(d_G, d_Y, d_X, nElem);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaMemcpy(X, d_X, nElem * 64 * sizeof(float), cudaMemcpyDeviceToHost));

	printf("mimo64_naive_gpu_kernel() costs %ld us \n", (long)(kernel_time * 1000.0f));

	CHECK(cudaMemcpy(d_G, G, (nElem / nElemPerMatrix) * 64 * 64 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Y, Y, nElem * 64 * sizeof(float), cudaMemcpyHostToDevice));

  	CHECK(cudaEventRecord(start, 0));

    mimo64_naive_gpu_kernel<<<grid, block>>>(d_G, d_Y, d_X, nElem);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaMemcpy(X, d_X, nElem * 64 * sizeof(float), cudaMemcpyDeviceToHost));

	printf("mimo64_naive_gpu_kernel() costs %ld us \n", (long)(kernel_time * 1000.0f));

	checkResult(X_base, X, nElem * 64);

#if 0
	const int BLOCK_SIZE_M = 2;
	const int BLOCK_SIZE_N = 4;
	const int BLOCK_SIZE_K = 2;
	
	dim3 block1 (1, 32);
    dim3 grid1  ((nElem + block1.y - 1) / block1.y);

  	CHECK(cudaEventRecord(start, 0));

    mimo64_block_naive_gpu_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<grid1, block1>>>(d_G, d_Y, d_X, nElem);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaMemcpy(X, d_X, nElem * 64 * sizeof(float), cudaMemcpyDeviceToHost));

	printf("mimo64_block_naive_gpu_kernel() costs %ld us \n", (long)(kernel_time * 1000.0f));

	checkResult(X, X_base, nElem * 64);
#endif

	Mimo64_free_host_mem(G);
	Mimo64_free_host_mem(Y);
	Mimo64_free_host_mem(X_base);
	Mimo64_free_host_mem(X);

	Mimo64_free_device_mem(d_G);
	Mimo64_free_device_mem(d_Y);
	Mimo64_free_device_mem(d_X);
	


	
	return 0;
}