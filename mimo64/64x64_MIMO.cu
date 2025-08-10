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

void initialData(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}


void initialData_u32(uint32_t *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (uint32_t)(rand() & 0xFFFFFFFF);
    }
}

void initialData_u8(uint8_t *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (uint8_t)(rand() & 0xFF);
    }
}


void initialData_s16(int16_t *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (int16_t)(rand() & 0xFFFF);
		//printf("val[%d]: %d \n", i, ip[i]);
    }
}

void initialData_f32(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFFFF);
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


void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
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

__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        for (int i = 0; i < N; ++i)
        {
            C[idx] = A[idx] + B[idx];
        }
    }
}

__global__ void warmingup(uint8_t *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = (uint8_t)(ia + ib);
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
				
				}
				
				X0[iRow * nElemPerMatrix + iCol] = accuVal;
			}
		}
	}


	return;
}

__global__ void mimo64_naive_gpu_kernel(float *G, float *Y, float *X, int nElem)
{
	int numOfloop = nElem / 4;
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
	

	return;
}


__global__ void mimo64_revised_gpu_kernel(float *G, float *Y, float *X, int nElem)
{
	int numOfloop = nElem / 4;
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
	

	return;
}



#define NSTREAM 4
#define BDIM 128

int main(int argc, char **argv)
{
    printf("> %s Starting...\n", argv[0]);
	
    // set up data size of vectors
    int nElem = 273 * 12 * 14;
	int nElemPerMatrix = 4;
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

	mimo64_naive_kernel(G, Y, X_base, nElem, nElemPerMatrix);
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


	CHECK(cudaMemcpy(d_G, G, (nElem / nElemPerMatrix) * 64 * 64 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Y, Y, nElem * 64 * sizeof(float), cudaMemcpyHostToDevice));

  	CHECK(cudaEventRecord(start, 0));

    mimo64_naive_gpu_kernel<<<grid, block>>>(d_G, d_Y, d_X, nElem);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaMemcpy(X, d_X, nElem * 64 * sizeof(float), cudaMemcpyDeviceToHost));

	printf("mimo64_naive_gpu_kernel() costs %ld us \n", (long)(kernel_time * 1000.0f));

	checkResult(X, X_base, nElem * 64);

	Mimo64_free_host_mem(G);
	Mimo64_free_host_mem(Y);
	Mimo64_free_host_mem(X_base);
	Mimo64_free_host_mem(X);

	Mimo64_free_device_mem(d_G);
	Mimo64_free_device_mem(d_Y);
	Mimo64_free_device_mem(d_X);
	


	
	return 0;
}