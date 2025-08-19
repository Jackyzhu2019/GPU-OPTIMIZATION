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
if (loopIdx == 0)
{
	for (iRow = 0; iRow < 64; iRow++)
	{
		printf("cpu iRow: %d Y0: %f %f %f %f \n", iRow, Y0[iRow * 4], Y0[iRow * 4 + 1], Y0[iRow * 4 + 2], Y0[iRow * 4 + 3]);
	}		
	// printf("iRow: %d iCol: %d iK: %d accu: %f G0: %f Y0: %f \n", 
	// iRow, iCol, iK, accuVal, G0[iRow * 64 + iK], Y0[iK * nElemPerMatrix + iCol]);
}
#endif	
				}
				
				X0[iRow * nElemPerMatrix + iCol] = accuVal;
			}
		}
		
#if 0
		if (loopIdx == 0)
		{
			for (iRow = 0; iRow < 64; iRow++)
			{
				printf("cpu iRow: %d G0: %f %f %f %f \n", iRow, G0[iRow * 64], G0[iRow * 64 + 1], G0[iRow * 64 + 2], G0[iRow * 64 + 3]);
			}		
			// printf("iRow: %d iCol: %d iK: %d accu: %f G0: %f Y0: %f \n", 
			// iRow, iCol, iK, accuVal, G0[iRow * 64 + iK], Y0[iK * nElemPerMatrix + iCol]);
		}
#endif	
		
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

	int xUnit = BLOCK_SIZE_K / xLen; // BLOCK_SIZE_K is multiple of xLen
	
	// shared memory
	__shared__ float Gs[64][BLOCK_SIZE_K]; // shared memory of G: 256 bytes x BLOCK_SIZE_K
	__shared__ float Ys[64][4]; // shared memory of Y: 1 Kbyte
	
	float accu[BLOCK_SIZE_M][BLOCK_SIZE_N];
	float *G0, *Y0, *X0;

	G0 = G + 64 * 64 * Block_X_Idx;
	Y0 = Y + 64 * 4 * Block_X_Idx;
	X0 = X + 64 * 4 * Block_X_Idx;

	// load data from global memory to shared memeory
	// load the whole Y 
	int sLoadIdx;
	float *Gs_0, *Ys_0;
	Gs_0 = &Gs[0][0];
	Ys_0 = &Ys[0][0];
	
	int nElemPerThread = 64 * 4 / (xLen * yLen);

	int currentThreadIdx = yIdx * xLen + xIdx;
	
	for (i = 0; i < nElemPerThread; i++)
	{
		sLoadIdx = currentThreadIdx * nElemPerThread + i;

		Ys_0[sLoadIdx] = Y0[sLoadIdx];
	}
	
	#pragma unroll
	for (i = 0; i < BLOCK_SIZE_M; i++)
	{
	    #pragma unroll
		for (j = 0; j < xUnit; j++)
		{
			int mIdx = yIdx * BLOCK_SIZE_M + i;
			// note: int kIdx = /*xIdx * BLOCK_SIZE_K +*/ j;
			int kIdx = xIdx * xUnit + j; // BLOCK_SIZE_K: 1, 2, 4
			
			sLoadIdx = 64 * mIdx + kIdx;
			
			Gs[mIdx][kIdx] = G0[sLoadIdx];
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
				val_Y[i][j] = Ys[(iTile - 1) * BLOCK_SIZE_K + i][xIdx * BLOCK_SIZE_N + j];		
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
				}
			}
		}
		

		
		if (iTile != loopIdx)
		{
			// next G data: BLOCK_SIZE_M * BLOCK_SIZE_K 
			#pragma unroll
			for (i = 0; i < BLOCK_SIZE_M; i++)
			{
				#pragma unroll
				for (j = 0; j < xUnit; j++)
				{
				
					int mIdx = yIdx * BLOCK_SIZE_M + i;
					// note: int kIdx = /*xIdx * BLOCK_SIZE_K +*/ j;
					int kIdx = xIdx * xUnit + j; // BLOCK_SIZE_K: 1, 2, 4
					
					sLoadIdx = 64 * mIdx + BLOCK_SIZE_K * iTile + kIdx;
					
					Gs[mIdx][kIdx] = G0[sLoadIdx];
				}
			}
				
			__syncthreads();
			
		}
		
		iTile++;
	}
	
	int storeIdx;
	#pragma unroll
	for (i = 0; i < BLOCK_SIZE_M; i++)
	{
		#pragma unroll
		for (j = 0; j < BLOCK_SIZE_N; j++)
		{
			storeIdx = (yIdx * BLOCK_SIZE_M + i) * 4
						+ xIdx * BLOCK_SIZE_N + j;
			X0[storeIdx] = accu[i][j];
		}
	}
	
	

	return;
}

template <
    const int BLOCK_SIZE_M,  // height of block of X that each thread calculate, i.e. bm
    const int BLOCK_SIZE_K,  // width of block of G that each thread load into shared memory, i.e. bk
    const int BLOCK_SIZE_N  // width of block of X that each thread calculate i.e. bn
    > 
__global__ void mimo64_block_revised_gpu_kernel(float *G, float *Y, float *X, int nElem)
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

	int xUnit = BLOCK_SIZE_K / xLen; // BLOCK_SIZE_K is multiple of xLen

	// shared memory
	__shared__ float Gs[64][BLOCK_SIZE_K]; // shared memory of G: 256 bytes x BLOCK_SIZE_K
	__shared__ float Ys[64][4]; // shared memory of Y: 1 Kbyte
	
	float accu[BLOCK_SIZE_M][BLOCK_SIZE_N];
	float *G0, *Y0, *X0;

	G0 = G + 64 * 64 * Block_X_Idx;
	Y0 = Y + 64 * 4 * Block_X_Idx;
	X0 = X + 64 * 4 * Block_X_Idx;

	// load data from global memory to shared memeory
	// load the whole Y 
	int sLoadIdx;
	float *Gs_0, *Ys_0;
	Gs_0 = &Gs[0][0];
	Ys_0 = &Ys[0][0];
	
	// printf("nElemPerThread: %d x/y Len: %d %d. currentThreadIdx: %d \n", nElemPerThread, xIdx, yIdx, currentThreadIdx);
	int loadGap = xLen * yLen; // multple of 32s
	int nG_ElemPerThread = (64 * BLOCK_SIZE_K) / loadGap;
	int currentThreadIdx = yIdx * xLen + xIdx;

	int yIdxK = currentThreadIdx / BLOCK_SIZE_K;
	int xIdxK = currentThreadIdx  -  yIdxK * BLOCK_SIZE_K;

	int sGLoadIdx = yIdxK * 64 + xIdxK;
	sLoadIdx = currentThreadIdx;
	int G_ElemGap = 64 / nG_ElemPerThread;
	//yIdxK += G_ElemGap;
	for (i = 0; i < nG_ElemPerThread; i++)
	{
		Gs_0[sLoadIdx] = G0[sGLoadIdx];
		sLoadIdx += loadGap;
		//sGLoadIdx = yIdxK * 64 + xIdxK;
		sGLoadIdx += G_ElemGap * 64;
	}

	//printf("hello world \n");
#if 0
	#pragma unroll
	for (i = 0; i < BLOCK_SIZE_M; i++)
	{
	    #pragma unroll
		for (j = 0; j < xUnit; j++)
		{
			int mIdx = yIdx * BLOCK_SIZE_M + i;
			// note: int kIdx = /*xIdx * BLOCK_SIZE_K +*/ j;
			int kIdx = xIdx * xUnit + j; // BLOCK_SIZE_K: 1, 2, 4
			
			sLoadIdx = 64 * mIdx + kIdx;
			
			Gs[mIdx][kIdx] = G0[sLoadIdx];
		}
	}
#endif

	int nElemPerThread = 64 * 4 / loadGap;

	sLoadIdx = currentThreadIdx;
	for (i = 0; i < nElemPerThread; i++)
	{
		Ys_0[sLoadIdx] = Y0[sLoadIdx];
		sLoadIdx += loadGap;
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
	
	// load from shared memory to register
	//float val_G[BLOCK_SIZE_M][BLOCK_SIZE_K];
	//float val_Y[BLOCK_SIZE_K][BLOCK_SIZE_N];
	float val_G[BLOCK_SIZE_M];
	float val_Y[BLOCK_SIZE_N];
	
	const int loopIdx = 64 / BLOCK_SIZE_K;
	int iTile = 1;
	int G_offset = yIdx * BLOCK_SIZE_M;
	int y_offset = xIdx * BLOCK_SIZE_N;
	while (iTile <= loopIdx)
	{
		// matrix multiply	
		#pragma unroll
		for (k = 0; k < BLOCK_SIZE_K; k++)
		{
			#pragma unroll
			for (i = 0; i < BLOCK_SIZE_M; i++)
			{
				val_G[i] = Gs[G_offset + i][k];
			}
			
			#pragma unroll
			for (j = 0; j < BLOCK_SIZE_N; j++)
			{
				val_Y[j] = Ys[(iTile - 1) * BLOCK_SIZE_K + k][y_offset + j];			
			}
			
			
			#pragma unroll
			for (i = 0; i < BLOCK_SIZE_M; i++)
			{
				#pragma unroll
				for (j = 0; j < BLOCK_SIZE_N; j++)
				{
					accu[i][j] += val_G[i] * val_Y[j];

				}
			}
		}
		

		
		if (iTile != loopIdx)
		{
			// next G data: BLOCK_SIZE_M * BLOCK_SIZE_K 
if (iTile >= 12)//(iTile != 13 && iTile != 2)
{
			#pragma unroll
			for (i = 0; i < BLOCK_SIZE_M; i++)
			{
				#pragma unroll
				for (j = 0; j < xUnit; j++)
				{
				
					int mIdx = yIdx * BLOCK_SIZE_M + i;
					// note: int kIdx = /*xIdx * BLOCK_SIZE_K +*/ j;
					int kIdx = xIdx * xUnit + j; // BLOCK_SIZE_K: 1, 2, 4
					
					sLoadIdx = 64 * mIdx + BLOCK_SIZE_K * iTile + kIdx;
					
					Gs[mIdx][kIdx] = G0[sLoadIdx];
				}
			}
} else {
			sLoadIdx = currentThreadIdx;

			sGLoadIdx = yIdxK * 64 + xIdxK + BLOCK_SIZE_K * iTile;

			//if ((blockIdx.x == 0) && (blockIdx.y == 0))
			//	printf("iTile: %d yIdxK: %d xIdxK: %d \n", iTile, yIdxK, xIdxK);
			
			for (i = 0; i < nG_ElemPerThread; i++)
			{
				Gs_0[sLoadIdx] = G0[sGLoadIdx];
				sLoadIdx += loadGap;
				//sGLoadIdx = yIdxK * 64 + xIdxK;
				sGLoadIdx += G_ElemGap * 64;
			}
}			

			__syncthreads();
		}
		
		iTile++;
	}
	
	int storeIdx;
	#pragma unroll
	for (i = 0; i < BLOCK_SIZE_M; i++)
	{
		#pragma unroll
		for (j = 0; j < BLOCK_SIZE_N; j++)
		{
			storeIdx = (yIdx * BLOCK_SIZE_M + i) * 4
						+ xIdx * BLOCK_SIZE_N + j;
			X0[storeIdx] = accu[i][j];
		}
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

#if 1
	// memset d_X, X
    memset(X, 0, nElem * 64 * sizeof(float));
    CHECK(cudaMemcpy(d_X, X, nElem * 64 * sizeof(float), cudaMemcpyHostToDevice));

	// (M, N, K) = (1, 1, 4), (1, 1, 2), (2, 1, 4), (2, 2, 4), (4, 1, 2), (4, 1, 4), (1, 2, 4), (2, 1, 8)
	const int BLOCK_SIZE_M = 2;
	const int BLOCK_SIZE_N = 1;
	const int BLOCK_SIZE_K = 4;
	
	dim3 block1 (4 / BLOCK_SIZE_N, 64 / BLOCK_SIZE_M);
    dim3 grid1  ((nElem + 3) / 4); // 4 elements in one block

  	CHECK(cudaEventRecord(start, 0));

    mimo64_block_naive_gpu_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<grid1, block1>>>(d_G, d_Y, d_X, nElem);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaMemcpy(X, d_X, nElem * 64 * sizeof(float), cudaMemcpyDeviceToHost));

	printf("mimo64_block_naive_gpu_kernel() costs %ld us \n", (long)(kernel_time * 1000.0f));

	checkResult(X_base, X, nElem * 64);
#endif

#if 1
	// memset d_X, X
    memset(X, 0, nElem * 64 * sizeof(float));
    CHECK(cudaMemcpy(d_X, X, nElem * 64 * sizeof(float), cudaMemcpyHostToDevice));

	// (M, N, K) = (1, 1, 4), (1, 4, 4), (1, 1, 2), (2, 1, 4), (2, 2, 4), (4, 1, 2), (4, 1, 4), (1, 2, 4), (2, 1, 8) 
	const int BLOCK_SIZE_M_2 = 2;
	const int BLOCK_SIZE_N_2 = 1;
	const int BLOCK_SIZE_K_2 = 4;
	
	dim3 block2 (4 / BLOCK_SIZE_N_2, 64 / BLOCK_SIZE_M_2);
    dim3 grid2  ((nElem + 3) / 4); // 4 elements in one block

	printf("blockIdx.x: %d blockIdx.y: %d \n", block2.x, block2.y);

  	CHECK(cudaEventRecord(start, 0));

    mimo64_block_revised_gpu_kernel<BLOCK_SIZE_M_2, BLOCK_SIZE_K_2, BLOCK_SIZE_N_2><<<grid2, block2>>>(d_G, d_Y, d_X, nElem);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaMemcpy(X, d_X, nElem * 64 * sizeof(float), cudaMemcpyDeviceToHost));

	printf("mimo64_block_revised_gpu_kernel() costs %ld us \n", (long)(kernel_time * 1000.0f));

	checkResult(X_base, X, nElem * 64);
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