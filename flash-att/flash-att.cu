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
//#define printf(...);

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
        ip[i] = (float)rand() / (float)RAND_MAX;
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
    double epsilon = 1.0E-2;
    bool match = 1;
	
//	printf("hello world. \n");
//	printf("host: %f gpu: %f \n", hostRef[0], gpuRef[0]);
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

					// printf("iRow: %d, jCol: %d, S: %f \n", iRow, jCol, accu);
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
				
				// printf("iRow: %d max: %f sumExp: %f \n", iRow, max, sumExp);
				// softmax 3) for each element, divide by sumExp
				for (jCol = 0; jCol < N; jCol++){
					val = *(S_3 + jCol);
					
					*(S_3 + jCol) = expf(val - max) / sumExp;
					// printf("iRow: %d, jCol: %d, Softmax(S): %f \n", iRow, jCol, *(S_3 + jCol));
				}
			}
			
			// Step 3: softmax output * V
			for (iRow = 0; iRow < N; iRow++){
				for (id = 0; id < d; id++){
					float* S_4 = S_1 + iRow * N;
					float* V_4 = V_1 + id;
					float* O_4 = O_1 + iRow * d + id;

					float accu1 = 0.0f;
					for (iN = 0; iN < N; iN++)
					{
						float S_val = *(S_4 + iN);
						float V_val = *(V_4 + iN * d);
						accu1 += S_val * V_val;
					}

					// printf("iRow: %d, id: %d, result idx: %d, Softmax(S) * V: %f \n", iRow, id, iRow * N + id, accu1);

					*O_4 = accu1;
				}
			}
		}
	}

	return;
}


void flash_att_merge_kernel(float* Q,
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
			float* S_1 = S;
			
			for (iRow = 0; iRow < N; iRow++){
				float max_iter = -999999.0f; //-FLT_MAX;
				float max_prev = 0.0f;
				float sumExp_iter = 0.0f;
				float sumExp_prev = 0.0f;
				float Output = 0.0f;
	
				for (jCol = 0; jCol < N; jCol++){
					float* Q_2 = Q_1 + iRow * d;
					float* K_2 = K_1 + jCol * d;
					float* S_2 = S_1 + iRow * N + jCol;

					// Step 1: Q * K_T
					float accu = 0.0f;
					for (id = 0; id < d; id++)
					{
						float Q_val = *(Q_2 + id);
						float K_val = *(K_2 + id);
						accu += Q_val * K_val;
					}
					
					// Step 2: Softmax
					max_iter = (max_iter > accu) ? max_iter : accu;
					float exp_diff = expf(max_prev - max_iter);
					float exp_curr = expf(accu - max_iter);

					sumExp_iter = exp_diff * sumExp_prev + exp_curr;

					// Step 3: Softmax output * V
					float k_i = exp_curr / sumExp_iter;
					
					for (id = 0; id < d; id++)
					{
						float output = *(O_1 + iRow * d + id);
						float V_val = *(V_1 + jCol * d + id);
						
						output = (output * exp_diff * sumExp_prev) / sumExp_iter;

						V_val = output + k_i * V_val;
						*(O_1 + iRow * d + id) = V_val;
					}
					
					// update max_prev, sumExp_prev
					max_prev = max_iter;
					sumExp_prev = sumExp_iter;
				}
			}
		}
	}

	return;
}

void flash_att_merge_block_kernel(float* Q,
							float* K,
							float* V,
							float* O,
							float* S,
							int nBatch,
							int nHead,
							int N,
							int d,
							int Br,
							int Bc)
{
	int iBatch, iHead, iN, id;
	int iRow, jCol;
	const int Tc = N / Bc;
	const int Tr = N / Br;
	int i, j;
	float S_ij[Br][Bc]; // used to store block tile Q * K_T
	float max_S[Br]; // used to store max value of each row in block tile
	float l_expsum[Br]; // used to store exp sum value of each row in block tile
	float O_ij[Br][d]; // used to store the output per block


	for (iBatch = 0; iBatch < nBatch; iBatch++){
		for (iHead = 0; iHead < nHead; iHead++){
			// for each batch, length = nHead * N * d
			// for each head, length = N * d
			int Offset = iBatch * nHead * N * d + iHead * N * d;
			float* Q_0 = Q + Offset;
			float* K_0 = K + Offset;
			float* V_0 = V + Offset;
			float* O_0 = O + Offset;
			float* S_0 = S;
			
			// Initialization
			memset(&S_ij[0][0], 0.0, Br * Bc * sizeof(float));
	
			
			for (i = 0; i < Tr; i++){
				float* Q_1 = Q_0 + i * Br * d;
				float* O_1 = O_0 + i * Br * d;

				float* S_1 = S;
				
				memset(&l_expsum[0], 0.0, Br * sizeof(float));

				for (int iMax = 0; iMax < Br; iMax++){
					max_S[iMax] = -999999.0f;
				}

				memset(&O_ij[0][0], 0.0, Br * d * sizeof(float));

				
				for (j = 0; j < Tc; j++){
					float* K_1 = K_0 + j * Bc * d;
					float* V_1 = V_0 + j * Bc * d;

					for (iRow = 0; iRow < Br; iRow++){
						float max_iter = -999999.0f; //-FLT_MAX;
						float max_prev = max_S[iRow];
						float sumExp_iter = 0.0f;
						float sumExp_prev = l_expsum[iRow];
						float Output = 0.0f;
						
						for (jCol = 0; jCol < Bc; jCol++){
							float* Q_2 = Q_1 + iRow * d;
							float* K_2 = K_1 + jCol * d;
							float* S_2 = S_1 + iRow * N + jCol;

							// Step 1: Q * K_T
							float accu = 0.0f;
							for (id = 0; id < d; id++)
							{
								float Q_val = *(Q_2 + id);
								float K_val = *(K_2 + id);
								accu += Q_val * K_val;
							}
							
							max_iter = (max_iter > accu) ? max_iter : accu;
							S_ij[iRow][jCol] = accu;
						}
						
						float exp_curr = 0.0f;
						max_iter = (max_iter > max_prev) ? max_iter : max_prev;
						for (jCol = 0; jCol < Bc; jCol++){
							// Step 2: sum exp per block
							S_ij[iRow][jCol] = expf(S_ij[iRow][jCol] - max_iter);
							exp_curr += S_ij[iRow][jCol];
						}
							
						float exp_diff = expf(max_prev - max_iter);
						
						sumExp_iter = exp_diff * sumExp_prev + exp_curr;
							
						for (id = 0; id < d; id++)
						{
							float output = O_ij[iRow][id];							
							float temp = (output * exp_diff * sumExp_prev) / sumExp_iter;
							float sum0 = 0.0f;
							
							for (jCol = 0; jCol < Bc; jCol++){
							// Step 3: Softmax output * V for the current block
								float k_i = S_ij[iRow][jCol] / sumExp_iter;
								float V_val = *(V_1 + jCol * d + id);
								
								sum0 += k_i * V_val;
								
							}
							
							O_ij[iRow][id] = temp + sum0;;						
						}
						
						// update max_prev, sumExp_prev
						max_S[iRow] = max_iter;
						l_expsum[iRow] = sumExp_iter;
					}
				}
				
				
				for (iRow = 0; iRow < Br; iRow++){
					for (id = 0; id < d; id++)
					{						 
						*(O_1 + iRow * d + id) = O_ij[iRow][id];
					}	
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
    int nBatch = 1; // number of batchs
	int nHead = 1; // number of heads
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
	printf("QKV size: %d \n", QKV_Size);
		
	Mimo64_alloc_host_mem((void **)&Q, QKV_Size * sizeof(float)); // 96MB
	Mimo64_alloc_host_mem((void **)&K, QKV_Size * sizeof(float));
	Mimo64_alloc_host_mem((void **)&V, QKV_Size * sizeof(float)); 
	Mimo64_alloc_host_mem((void **)&O_base, QKV_Size * sizeof(float));
	Mimo64_alloc_host_mem((void **)&O, QKV_Size * sizeof(float));
	Mimo64_alloc_host_mem((void **)&S, QK_T_Size * sizeof(float)); // 4MB 


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

	long t_start = useconds();
	//printf("V: %f %f \n", V[0], V[1]);
	flash_att_naive_kernel(Q, K, V, O_base, S, nBatch, nHead, nTokens, HeadDim);
	//printf("O_base: %f %f \n", O_base[0], O_base[1]);
	
	long t_end = useconds();
	printf("flash_att_naive_kernel() costs %ld us \n", (t_end - t_start) );

	long t_start1 = useconds();
	//printf("V: %f %f \n", V[0], V[1]);
	flash_att_merge_kernel(Q, K, V, O, S, nBatch, nHead, nTokens, HeadDim);
	//printf("O: %f %f \n", O[0], O[1]);
	
	long t_end1 = useconds();
	printf("flash_att_merge_kernel() costs %ld us \n", (t_end1 - t_start1) );

	checkResult(O_base, O, QKV_Size);	


    memset(O, 0, QKV_Size * sizeof(float));
	long t_start2 = useconds();
	//printf("V: %f %f \n", V[0], V[1]);
	flash_att_merge_block_kernel(Q, K, V, O, S, nBatch, nHead, nTokens, HeadDim, 64, 128);
	//printf("O: %f %f \n", O[0], O[1]);
	
	long t_end2 = useconds();
	printf("flash_att_merge_block_kernel() costs %ld us \n", (t_end2 - t_start2) );

	checkResult(O_base, O, QKV_Size);


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