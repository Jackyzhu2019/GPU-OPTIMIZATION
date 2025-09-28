#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "./common.h"
#include <stdint.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

#define DOUBLE(pointer) (reinterpret_cast<double*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define float16_t half


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

	return;
}


void initialData_Y_f32(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
		if (i < size)
			ip[i] = (float) 1.0f; //(rand() & 0xFF);
		//printf("val[%d]: %d \n", i, ip[i]);
		else
			ip[i] = (float)0.0; //(rand() & 0xFF);
		//printf("val[%d]: %d \n", i, ip[i]);
    }

	return;
}

void Mimo64_init_device_const_mem()
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

 

	return;
}


void float2half(float16_t *dst, float *src, int nElem){
	for (int i = 0; i < nElem; i++){
		dst[i] = __float2half(src[i]);
	}
	
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
	
//	printf("hello world. \n");
//	printf("host: %f gpu: %f \n", hostRef[0], gpuRef[0]);
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %f gpu %f at %d (block: %d, thread: %d)\n", hostRef[i], gpuRef[i], i, i / (64 * 4), i % (64 * 4));
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}


void flash_att_naive_kernel(float* Q,
							float* K,
							float* V,
							float* O,
							float* S,
							int nBatch,
							int nHead,
							int N,
							int d) 
{
	int iBatch, iHead, iN, id;
	int iRow, jCol;
	
	for (iBatch = 0; iBatch < nBatch; iBatch++){
		for (iHead = 0; iHead < nHead; iHead++){
			// for each batch, length = nHead * N * d
			// for each head, length = N * d
			int Offset = iBatch * nHead * N * d + iHead * N * d;
			float* Q_1 = Q + Offset;
			float* K_1 = K + Offset;
			float* V_1 = V + Offset;
			float* O_1 = O + Offset;
			float* S_1 = S; // + Offset_QK_T;
			
			// Step 1: Q * K_T
			for (iRow = 0; iRow < N; iRow++){
				for (jCol = 0; jCol < N; jCol++){
					float* Q_2 = Q_1 + iRow * d;
					float* K_2 = K_1 + jCol * d;
					float* S_2 = S_1 + iRow * N + jCol;

					float accu = 0.0f;
					for (id = 0; id < d; id++)
					{
						float Q_val = *(Q_2 + id);
						float K_val = *(K_2 + id);
						accu += Q_val * K_val;
					}

					*S_2 = accu;
				}
			}
			
			
			// Step 2: Softmax
			for (iRow = 0; iRow < N; iRow++){
				float* S_3 = S_1 + iRow * N;

				// softmax 1) find max
				float max = *S_3;
				float val;
				for (jCol = 0; jCol < N; jCol++){
					val = *(S_3 + jCol);
					
					max = (max > val) ? max : val;
				}

				// softmax 2) sumExp among all elements
				float sumExp = 0.0f;
				for (jCol = 0; jCol < N; jCol++){
					val = *(S_3 + jCol);
					
					sumExp += expf(val - max);
				}
				
				// softmax 3) for each element, divide by sumExp
				for (jCol = 0; jCol < N; jCol++){
					val = *(S_3 + jCol);
					
					*(S_3 + jCol) = expf(val - max) / sumExp;
				}
			}
			
			// Step 3: softmax output * V
			for (iRow = 0; iRow < N; iRow++){
				for (id = 0; id < d; id++){
					float* S_4 = S_1 + iRow * N;
					float* V_4 = V_1 + id;
					float* O_4 = O_1 + iRow * N + id;

					float accu1 = 0.0f;
					for (iN = 0; iN < N; iN++)
					{
						float S_val = *(S_4 + iN);
						float V_val = *(V_4 + iN * d);
						accu1 += S_val * V_val;
					}

					*O_4 = accu1;
				}
			}
		}
	}

	return;
}


int main(int argc, char **argv)
{
    printf("> %s Starting...\n", argv[0]);
	
    // GPT-2 parameters
    int nBatch = 32; // number of batchs
	int nHead = 12; // number of heads
	int HeadDim = 64; // head dimension: 768 / 12 heads
	int nTokens = 1024; // number of tokens in a batch
	
    float *Q;
	float *K;
	float *V;
	float *O_base; // output for reference
	float *O; // output
	float *S; // temporary result to storce Q * K_T
	
	int QKV_Size = nBatch * nHead * HeadDim * nTokens; // 24M
	int QK_T_Size = nTokens * nTokens; // 1MB
	Mimo64_alloc_host_mem((void **)&Q, QKV_Size * sizeof(float)); // 96MB
	Mimo64_alloc_host_mem((void **)&K, QKV_Size * sizeof(float));
	Mimo64_alloc_host_mem((void **)&V, QKV_Size * sizeof(float)); 
	Mimo64_alloc_host_mem((void **)&O_base, QKV_Size * sizeof(float));
	Mimo64_alloc_host_mem((void **)&O, QKV_Size * sizeof(float));
	Mimo64_alloc_host_mem((void **)&S, QK_T_Size * sizeof(float)); // 1,536MB 


	float *d_Q;
	float *d_K;
	float *d_V;
	float *d_O;

	Mimo64_alloc_device_mem((void **)&d_Q, QKV_Size * sizeof(float));
	Mimo64_alloc_device_mem((void **)&d_K, QKV_Size * sizeof(float));
	Mimo64_alloc_device_mem((void **)&d_V, QKV_Size * sizeof(float));

    memset(O, 0, QKV_Size * sizeof(float));
    memset(O_base,  0, QKV_Size * sizeof(float));

 	initialData_f32(Q, QKV_Size);
 	initialData_f32(K, QKV_Size);
 	initialData_f32(V, QKV_Size);

	//mimo64_naive_kernel(G, Y, X_base, nElem, nElemPerMatrix);
	long t_start = useconds();

	flash_att_naive_kernel(Q, K, V, O_base, S, nBatch, nHead, nTokens, HeadDim);
	
	long t_end = useconds();
	printf("flash_att_naive_kernel() costs %ld us \n", (t_end - t_start) );

	Mimo64_free_host_mem(Q);
	Mimo64_free_host_mem(K);
	Mimo64_free_host_mem(V);
	Mimo64_free_host_mem(O);
	Mimo64_free_host_mem(O_base);
	
	Mimo64_free_device_mem(d_Q);
	Mimo64_free_device_mem(d_K);
	Mimo64_free_device_mem(d_V);
	
	return 0;
}