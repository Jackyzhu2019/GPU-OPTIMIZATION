#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "./common.h"
#include <stdint.h>

#define coe_scale_K_q8   0x013A //r16s11, 2/sqrt(170)
#define coe_scale_2K_q8   0x0274 //r16s11, 4/sqrt(170)
#define coe_scale_3K_q8   0x03AE //r16s11, 6/sqrt(170)
#define coe_scale_4K_q8   0x04E8 //r16s11, 8/sqrt(170)
#define coe_scale_5K_q8   0x0622 //r16s11, 10/sqrt(170)
#define coe_scale_6K_q8   0x075C //r16s11, 12/sqrt(170)
#define coe_scale_7K_q8   0x0897 //r16s11, 14/sqrt(170)
#define coe_scale_9K_q8   0x0585 //r16s10, 18/sqrt(170)
#define coe_scale_10K_q8   0x0311 //r16s9, 20/sqrt(170)
#define coe_scale_11K_q8   0x06BF //r16s10, 22/sqrt(170)
#define coe_scale_13K_q8   0x07F9 //r16s10, 26/sqrt(170)
#define coe_scale_15K_q8   0x0934 //r16s10, 30/sqrt(170)
#define coe_scale_21K_q8   0x0338 //r16s8, 42/sqrt(170)
#define coe_scale_22K_q8   0x06BF //r16s9, 44/sqrt(170)
#define coe_scale_28K_q8   0x044B //r16s8, 56/sqrt(170)

#define coe_scale_15K_0   coe_scale_15K_q8 >> 1
#define coe_scale_7K_0   coe_scale_7K_q8 >> 1
#define coe_scale_6K_0   coe_scale_6K_q8 >> 2
#define coe_scale_5K_0   coe_scale_5K_q8 >> 1
#define coe_scale_3K_0   coe_scale_3K_q8 >> 1
#define coe_scale_K_0   coe_scale_K_q8 >> 1

__constant__ int16_t scale_table_b0b1_const[8];
__constant__ int16_t scale_table_1_b0b1_const[8];
__constant__ int16_t scale_table_b2b3_const[8]; 
__constant__ int16_t scale_table_1_b2b3_const[8];

__constant__ int16_t scale_table_b4b5_const[8];
__constant__ int16_t scale_table_1_b4b5_const[8];

__constant__ int16_t scale_table_b6b7_const[8];
__constant__ int16_t scale_table_1_b6b7_const[8];

__constant__ int16_t d_In_ScaleLUT_const[256];


void QAM_alloc_host_mem(void** host_ptr_addr, size_t size)
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

void QAM_free_host_mem(void* host_ptr)
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

void QAM_alloc_device_mem(void** dev_ptr_addr, size_t size)
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

void QAM_free_device_mem(void* dev_ptr)
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

void QAM_init_device_const_mem()
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int16_t scale_table_b0b1[8]= { coe_scale_7K_q8, coe_scale_6K_q8, coe_scale_5K_q8, coe_scale_4K_q8, coe_scale_3K_q8, coe_scale_2K_q8, coe_scale_K_q8, 0 };
    int16_t scale_table_1_b0b1[8] = { coe_scale_28K_q8, coe_scale_21K_q8, coe_scale_15K_0, coe_scale_10K_q8, coe_scale_6K_0, coe_scale_3K_0, coe_scale_K_0, 0 };

    int16_t scale_table_b2b3[8]= { coe_scale_7K_q8, coe_scale_6K_q8, coe_scale_5K_q8, coe_scale_3K_q8, coe_scale_2K_q8, coe_scale_K_q8, 0, 0 };
    int16_t scale_table_1_b2b3[8]= { coe_scale_22K_q8, coe_scale_15K_q8, coe_scale_9K_q8, coe_scale_4K_q8, coe_scale_7K_0, coe_scale_9K_q8, coe_scale_10K_q8, 0 };

    int16_t scale_table_b4b5[8]= { coe_scale_7K_q8, coe_scale_5K_q8, coe_scale_4K_q8, coe_scale_3K_q8, coe_scale_K_q8, 0, 0, 0 };
    int16_t scale_table_1_b4b5[8]= { coe_scale_13K_q8, coe_scale_6K_q8, coe_scale_11K_q8, coe_scale_5K_0, coe_scale_2K_q8, coe_scale_3K_0, 0, 0 };

    int16_t scale_table_b6b7[8]= { coe_scale_6K_q8, coe_scale_4K_q8, coe_scale_2K_q8, 0, 0, 0, 0, 0 };
    int16_t scale_table_1_b6b7[8]= { coe_scale_7K_q8, coe_scale_5K_q8, coe_scale_3K_q8, coe_scale_K_q8, 0, 0, 0, 0 };

	int16_t h_ScaleLUT[256];
	
	initialData_s16(h_ScaleLUT, 256);


	err = cudaMemcpyToSymbol(scale_table_b0b1_const, scale_table_b0b1, 8 * sizeof(int16_t), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "%s:%d, cuda error (error code %s)!\n", __func__, __LINE__, cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(scale_table_1_b0b1_const, scale_table_1_b0b1, 8 * sizeof(int16_t), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "%s:%d, cuda error (error code %s)!\n", __func__, __LINE__, cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(scale_table_b2b3_const, scale_table_b2b3, 8 * sizeof(int16_t), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "%s:%d, cuda error (error code %s)!\n", __func__, __LINE__, cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(scale_table_1_b2b3_const, scale_table_1_b2b3, 8 * sizeof(int16_t), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "%s:%d, cuda error (error code %s)!\n", __func__, __LINE__, cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(scale_table_b4b5_const, scale_table_b4b5, 8 * sizeof(int16_t), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "%s:%d, cuda error (error code %s)!\n", __func__, __LINE__, cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(scale_table_1_b4b5_const, scale_table_1_b4b5, 8 * sizeof(int16_t), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "%s:%d, cuda error (error code %s)!\n", __func__, __LINE__, cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(scale_table_b6b7_const, scale_table_b6b7, 8 * sizeof(int16_t), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "%s:%d, cuda error (error code %s)!\n", __func__, __LINE__, cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(scale_table_1_b6b7_const, scale_table_1_b6b7, 8 * sizeof(int16_t), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "%s:%d, cuda error (error code %s)!\n", __func__, __LINE__, cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}


	err = cudaMemcpyToSymbol(d_In_ScaleLUT_const, h_ScaleLUT, 256 * sizeof(int16_t), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "%s:%d, cuda error (error code %s)!\n", __func__, __LINE__, cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	return;
}



cudaStream_t *streams;

void QAM_createStreams(int numOfStreams){

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

void checkResult(uint8_t *hostRef, uint8_t *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %x gpu %x at %d\n", hostRef[i], gpuRef[i], i);
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

static __device__ int16_t X_sat_16(int32_t var1)
{
#if QAM256_OPT == 0
	int16_t ret_r16;

	if(var1 > 32767)
	{
		return (int16_t)32767;
	}
	else if(var1 < -32768)
	{
		return (int16_t)-32768;
	}
	else
	{
		ret_r16 = var1;
		return ret_r16;
	}
#else
    return (int16_t)max(-32768, min(32767, var1));
#endif
}

static __device__ int8_t X_sat_8(int16_t var1)
{
#if QAM256_OPT == 0
	int8_t ret_r8;

	if(var1 > 127)
	{
		return 127;
	}
	else if(var1 < -128)
	{
		return -128;
	}
	else
	{
		ret_r8 = var1;
		return ret_r8;
	}
#else
    return (int8_t)max(-128, min(127, var1));
#endif
}

static __device__ int16_t vqsubq_s16(int16_t a, int16_t b)
{
	return X_sat_16((int32_t)a - (int32_t)b);
}

static __device__ int16_t vqaddq_s16(int16_t a, int16_t b)
{
	return X_sat_16((int32_t)a + (int32_t)b);
}


#if QAM256_OPT == 0
static __device__ int8_t QAM256_Scaling_soft_bit0_bit1_cuda(int16_t r_x, uint16_t r_x_abs, int Scaling)
{
	//QAM256
	const int16_t scale_K_q8 = 0x013A; //r16s11, 2/sqrt(170)
	const int16_t scale_2K_q8 = 0x0274; //r16s11, 4/sqrt(170)
	const int16_t scale_3K_q8 = 0x03AE; //r16s11, 6/sqrt(170)
	const int16_t scale_4K_q8 = 0x04E8; //r16s11, 8/sqrt(170)
	const int16_t scale_5K_q8 = 0x0622; //r16s11, 10/sqrt(170)
	const int16_t scale_6K_q8 = 0x075C; //r16s11, 12/sqrt(170)
	const int16_t scale_7K_q8 = 0x0897; //r16s11, 14/sqrt(170)
	const int16_t scale_9K_q8 = 0x0585; //r16s10, 18/sqrt(170)
	const int16_t scale_10K_q8 = 0x0311; //r16s9, 20/sqrt(170)
	const int16_t scale_11K_q8 = 0x06BF; //r16s10, 22/sqrt(170)
	const int16_t scale_13K_q8 = 0x07F9; //r16s10, 26/sqrt(170)
	const int16_t scale_15K_q8 = 0x0934; //r16s10, 30/sqrt(170)
	const int16_t scale_21K_q8 = 0x0338; //r16s8, 42/sqrt(170)
	const int16_t scale_22K_q8 = 0x06BF; //r16s9, 44/sqrt(170)
	const int16_t scale_28K_q8 = 0x044B; //r16s8, 56/sqrt(170)

	const int16_t scale_15K_0 = scale_15K_q8 >> 1;
	const int16_t scale_7K_0 = scale_7K_q8 >> 1;
	const int16_t scale_6K_0 = scale_6K_q8 >> 2;
	const int16_t scale_5K_0 = scale_5K_q8 >> 1;
	const int16_t scale_3K_0 = scale_3K_q8 >> 1;
	const int16_t scale_K_0 = scale_K_q8 >> 1;

	int16_t d0, d1;
	int16_t scale_K;
	int32_t r_w32;

	if(scale_7K_q8 < r_x_abs)
	{
		//r16s8
		if(r_x > 0)
		{
			scale_K = scale_28K_q8;
		}
		else if(r_x < 0)
		{
			scale_K = -scale_28K_q8;
		}
		else
		{
			scale_K = 0;
		}

		//r16s11 * 8 -> r16s8, change format
		d0 = r_x;
		//r16s8 - r16s8
		d0 = vqsubq_s16(d0, scale_K);

		//r16s8*r16s2 -> r32s10 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 10;
		d0 = X_sat_16(r_w32);

	}
	else if(scale_6K_q8 < r_x_abs)
	{
		//r16s8
		if(r_x > 0)
		{
			scale_K = scale_21K_q8;
		}
		else if(r_x < 0)
		{
			scale_K = -scale_21K_q8;
		}
		else
		{
			scale_K = 0;
		}

		//r16s11 * 8 -> r16s8, change format
		//r16s8 - r16s8
		d0 = vqsubq_s16(r_x, (r_x >> 3));
		//r16s8 - r16s8
		d0 = vqsubq_s16(d0, scale_K);

		//r16s8*r16s2 -> r32s10 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 10;
		d0 = X_sat_16(r_w32);

	}
	else if(scale_5K_q8 < r_x_abs)
	{
		//r16s9
		if(r_x > 0)
		{
			scale_K = scale_15K_0;
		}
		else if(r_x < 0)
		{
			scale_K = -scale_15K_0;
		}
		else
		{
			scale_K = 0;
		}

		//r16s11 * 4 -> r16s9, change format
		d0 = r_x;
		//r16s9 + (r16s11 * 2 (change format) >> 1 -> r16s9)
		d0 = vqaddq_s16(d0, (r_x >> 1));
		//r16s9 - r16s9
		d0 = vqsubq_s16(d0, scale_K);

		//r16s9*r16s2 -> r32s11 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 11;
		d0 = X_sat_16(r_w32);

	}
	else if(scale_4K_q8 < r_x_abs)
	{
		//r16s9
		if(r_x > 0)
		{
			scale_K = scale_10K_q8;
		}
		else if(r_x < 0)
		{
			scale_K = -scale_10K_q8;
		}
		else
		{
			scale_K = 0;
		}
	
		//r16s11 * 4 -> r16s9, change format
		//r16s9 + (r16s11 >> 2 -> r16s9)
		d0 = vqaddq_s16(r_x, (r_x >> 2));
		
		//r16s9 - r16s9
		d0 = vqsubq_s16(d0, scale_K);
		
		//r16s9*r16s2 -> r32s11 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		
		r_w32 >>= 11;
		
		d0 = X_sat_16(r_w32);
	}
	else if(scale_3K_q8 < r_x_abs)
	{
		//r16s9
		if(r_x > 0)
		{
			scale_K = scale_6K_0;
		}
		else if(r_x < 0)
		{
			scale_K = -scale_6K_0;
		}
		else
		{
			scale_K = 0;
		}

		//r16s11 * 4 -> r16s9, change format
		//r16s9 - r16s9
		d0 = vqsubq_s16(r_x, scale_K);

		//r16s9*r16s2 -> r32s11 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 11;
		d0 = X_sat_16(r_w32);

	}
	else if(scale_2K_q8 < r_x_abs)
	{
		//r16s10
		if(r_x > 0)
		{
			scale_K = scale_3K_0;
		}
		else if(r_x < 0)
		{
			scale_K = -scale_3K_0;
		}
		else
		{
			scale_K = 0;
		}

		//r16s11 * 2 -> r16s10, change format
		//r16s10 + (r16s11 >> 1 -> r16s10)
		d0 = vqaddq_s16(r_x, (r_x >> 1));
		//r16s10 - r16s10
		d0 = vqsubq_s16(d0, scale_K);

		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);

	}
	else if(scale_K_q8 < r_x_abs)
	{
		//r16s10
		if(r_x > 0)
		{
			scale_K = scale_K_0;
		}
		else if(r_x < 0)
		{
			scale_K = -scale_K_0;
		}
		else
		{
			scale_K = 0;
		}

		//r16s11 * 2 -> r16s10, change format
		//r16s10 - r16s10
		d0 = vqsubq_s16(r_x, scale_K);

		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);
	}
	else
	{
		//r16s11*r16s2 -> r32s13 -> r32s0 -> sat -> r16s0
		r_w32 = r_x*Scaling;
		r_w32 >>= 13;
		d0 = X_sat_16(r_w32);
	}

	//r16s0 -> sat -> r8s0
	return X_sat_8(d0);
}
#else
static __device__ int8_t QAM256_Scaling_soft_bit0_bit1_cuda(int16_t r_x, uint16_t r_x_abs, int Scaling)
{
	int16_t d0;
	int32_t r_w32;

	int32_t index = 7;  // default

#if 1
	for (int i = 6; i >= 0; i--) {
		index = (r_x_abs > scale_table_b0b1_const[i]) ? i : index;
	}
	
	int16_t scale_K = scale_table_1_b0b1_const[index];
#else
	
	index = (r_x_abs > coe_scale_K_q8) ? 6 : index;
	index = (r_x_abs > coe_scale_2K_q8) ? 5 : index;
	index = (r_x_abs > coe_scale_3K_q8) ? 4 : index;
	index = (r_x_abs > coe_scale_4K_q8) ? 3 : index;
	index = (r_x_abs > coe_scale_5K_q8) ? 2 : index;
	index = (r_x_abs > coe_scale_6K_q8) ? 1 : index;
	index = (r_x_abs > coe_scale_7K_q8) ? 0 : index;

	scale_K = (index == 0) ? coe_scale_28K_q8 : scale_K; 
	scale_K = (index == 1) ? coe_scale_21K_q8 : scale_K; 
	scale_K = (index == 2) ? coe_scale_15K_0 : scale_K; 
	scale_K = (index == 3) ? coe_scale_10K_q8 : scale_K; 
	scale_K = (index == 4) ? coe_scale_6K_0 : scale_K; 
	scale_K = (index == 5) ? coe_scale_3K_0 : scale_K; 
	scale_K = (index == 6) ? coe_scale_K_0 : scale_K; 
#endif

	int sign = (r_x > 0) - (r_x < 0);  // sign will be 1, -1, or 0
	scale_K = (index == 7) ? 0 : sign * scale_K;

	int16_t temp0 = (index == 1) ? r_x >> 3 : 0;
	d0 = vqsubq_s16(r_x, temp0);

    int16_t shf = (index == 3) ? 2 : ((index == 2 || index == 5) ? 1 : 0);
    int16_t shf_1 = (index <= 1) ? 10 : (index <= 4) ? 11 : (index <= 6) ? 12 : 13;

	int16_t temp1 = (shf == 0) ? 0 : r_x >> shf;
	d0 = vqaddq_s16(r_x, temp1);

	d0 = vqsubq_s16(d0, scale_K);

	r_w32 = d0*Scaling;

	r_w32 >>= shf_1;

	d0 = X_sat_16(r_w32);

	return X_sat_8(d0);
}
#endif


#if QAM256_OPT == 0
static __device__ int8_t QAM256_Scaling_soft_bit2_bit3_cuda(int16_t r_x, uint16_t r_x_abs, int Scaling)
{
	//QAM256
	const int16_t scale_K_q8 = 0x013A; //r16s11, 2/sqrt(170)
	const int16_t scale_2K_q8 = 0x0274; //r16s11, 4/sqrt(170)
	const int16_t scale_3K_q8 = 0x03AE; //r16s11, 6/sqrt(170)
	const int16_t scale_4K_q8 = 0x04E8; //r16s11, 8/sqrt(170)
	const int16_t scale_5K_q8 = 0x0622; //r16s11, 10/sqrt(170)
	const int16_t scale_6K_q8 = 0x075C; //r16s11, 12/sqrt(170)
	const int16_t scale_7K_q8 = 0x0897; //r16s11, 14/sqrt(170)
	const int16_t scale_9K_q8 = 0x0585; //r16s10, 18/sqrt(170)
	const int16_t scale_10K_q8 = 0x0311; //r16s9, 20/sqrt(170)
	const int16_t scale_11K_q8 = 0x06BF; //r16s10, 22/sqrt(170)
	const int16_t scale_13K_q8 = 0x07F9; //r16s10, 26/sqrt(170)
	const int16_t scale_15K_q8 = 0x0934; //r16s10, 30/sqrt(170)
	const int16_t scale_21K_q8 = 0x0338; //r16s8, 42/sqrt(170)
	const int16_t scale_22K_q8 = 0x06BF; //r16s9, 44/sqrt(170)
	const int16_t scale_28K_q8 = 0x044B; //r16s8, 56/sqrt(170)

	const int16_t scale_15K_0 = scale_15K_q8 >> 1;
	const int16_t scale_7K_0 = scale_7K_q8 >> 1;
	const int16_t scale_6K_0 = scale_6K_q8 >> 2;
	const int16_t scale_5K_0 = scale_5K_q8 >> 1;
	const int16_t scale_3K_0 = scale_3K_q8 >> 1;
	const int16_t scale_K_0 = scale_K_q8 >> 1;

	int16_t d0, d1;
	int16_t scale_K;
	int32_t r_w32;

	if(scale_7K_q8 < r_x_abs)
	{
		//r16s11 * 4 -> r16s9, change format
		//r16s9 - r16s9
		d0 = scale_22K_q8 - r_x_abs;

		//r16s9*r16s2 -> r32s11 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 11;
		d0 = X_sat_16(r_w32);

	}
	else if(scale_6K_q8 < r_x_abs)
	{
		//r16s10 * 2 -> r16s10, change format
		//r16s10 - r16s10 - (r16s11 >> 1 -> r16s10)
		d1 = vqaddq_s16(r_x_abs, (r_x_abs >> 1));
		d0 = vqsubq_s16(scale_15K_q8, d1);

		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);
	}
	else if(scale_5K_q8 < r_x_abs)
	{
		//r16s10 * 2 -> r16s10, change format
		//r16s10 - r16s10
		d0 = vqsubq_s16(scale_9K_q8, r_x_abs);

		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);

	}
	else if(scale_3K_q8 < r_x_abs)
	{
		//r16s11 - r16s11
		d0 = vqsubq_s16(scale_4K_q8, r_x_abs);

		//r16s11*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 13;
		d0 = X_sat_16(r_w32);

	}
	else if(scale_2K_q8 < r_x_abs)
	{
		//r16s11 * 2 -> r16s10, change format
		//r16s10 - r16s10
		d0 = vqsubq_s16(scale_7K_0, r_x_abs);

		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);

	}
	else if(scale_K_q8 < r_x_abs)
	{
		//r16s10 * 2 -> r16s10, change format
		//r16s10 - r16s10 - (r16s11 >> 1 -> r16s10)
		d1 = vqaddq_s16(r_x_abs, (r_x_abs >> 1));
		d0 = vqsubq_s16(scale_9K_q8, d1);

		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);

	}
	else
	{
		//r16s11 * 4 -> r16s9, change format
		//r16s9 - r16s9
		d0 = vqsubq_s16(scale_10K_q8, r_x_abs);

		//r16s9*r16s2 -> r32s11 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 11;
		d0 = X_sat_16(r_w32);

	}

	//r16s0 -> sat -> r8s0
	return X_sat_8(d0);
}
#else
static __device__ int8_t QAM256_Scaling_soft_bit2_bit3_cuda(int16_t r_x, uint16_t r_x_abs, int Scaling)
{	
	int16_t d0;
	//int16_t scale_K;
	int32_t r_w32;
		
	int index = 6;  // default
	
#if 1
	for (int i = 5; i >= 0; i--) {
	    index = (r_x_abs > scale_table_b2b3_const[i]) ? i : index;
	}
#else

	index = (r_x_abs > coe_scale_K_q8) ? 5 : index;
	index = (r_x_abs > coe_scale_2K_q8) ? 4 : index;
	index = (r_x_abs > coe_scale_3K_q8) ? 3 : index;
	index = (r_x_abs > coe_scale_5K_q8) ? 2 : index;
	index = (r_x_abs > coe_scale_6K_q8) ? 1 : index;
	index = (r_x_abs > coe_scale_7K_q8) ? 0 : index;
#endif	
	
	int16_t shf_1 = 12;
	shf_1 += ((index == 3) - (index == 0 || index == 6));

	int16_t temp1 = (index == 1 || index == 5) ? (r_x_abs >> 1) : 0;
	d0 = vqaddq_s16(r_x_abs, temp1);
		
	d0 = vqsubq_s16(scale_table_1_b2b3_const[index], d0);

	r_w32 = d0*Scaling;
	r_w32 >>= shf_1;
	d0 = X_sat_16(r_w32);

	//r16s0 -> sat -> r8s0
	return X_sat_8(d0);
}
#endif

#if QAM256_OPT == 0
static __device__ int8_t QAM256_Scaling_soft_bit4_bit5_cuda(int16_t r_x, uint16_t r_x_abs, int Scaling)
{
	//QAM256
	const int16_t scale_K_q8 = 0x013A; //r16s11, 2/sqrt(170)
	const int16_t scale_2K_q8 = 0x0274; //r16s11, 4/sqrt(170)
	const int16_t scale_3K_q8 = 0x03AE; //r16s11, 6/sqrt(170)
	const int16_t scale_4K_q8 = 0x04E8; //r16s11, 8/sqrt(170)
	const int16_t scale_5K_q8 = 0x0622; //r16s11, 10/sqrt(170)
	const int16_t scale_6K_q8 = 0x075C; //r16s11, 12/sqrt(170)
	const int16_t scale_7K_q8 = 0x0897; //r16s11, 14/sqrt(170)
	const int16_t scale_9K_q8 = 0x0585; //r16s10, 18/sqrt(170)
	const int16_t scale_10K_q8 = 0x0311; //r16s9, 20/sqrt(170)
	const int16_t scale_11K_q8 = 0x06BF; //r16s10, 22/sqrt(170)
	const int16_t scale_13K_q8 = 0x07F9; //r16s10, 26/sqrt(170)
	const int16_t scale_15K_q8 = 0x0934; //r16s10, 30/sqrt(170)
	const int16_t scale_21K_q8 = 0x0338; //r16s8, 42/sqrt(170)
	const int16_t scale_22K_q8 = 0x06BF; //r16s9, 44/sqrt(170)
	const int16_t scale_28K_q8 = 0x044B; //r16s8, 56/sqrt(170)

	const int16_t scale_15K_0 = scale_15K_q8 >> 1;
	const int16_t scale_7K_0 = scale_7K_q8 >> 1;
	const int16_t scale_6K_0 = scale_6K_q8 >> 2;
	const int16_t scale_5K_0 = scale_5K_q8 >> 1;
	const int16_t scale_3K_0 = scale_3K_q8 >> 1;
	const int16_t scale_K_0 = scale_K_q8 >> 1;

	int16_t d0, d1;
	int16_t scale_K;
	int32_t r_w32;

	if(scale_7K_q8 < r_x_abs)
	{
		//r16s11 * 2 -> r16s10, change format
		//r16s10 - r16s10
		d0 = vqsubq_s16(scale_13K_q8, r_x_abs);
		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);
	}
	else if(scale_5K_q8 < r_x_abs)
	{
		//r16s11 - r16s11
		d0 = vqsubq_s16(scale_6K_q8, r_x_abs);
		//r16s11*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 13;
		d0 = X_sat_16(r_w32);
	}
	else if(scale_4K_q8 < r_x_abs)
	{
		//r16s11 * 2 -> r16s10, change format
		//r16s10 - r16s10
		d0 = vqsubq_s16(scale_11K_q8, r_x_abs);
		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);
	}
	else if(scale_3K_q8 < r_x_abs)
	{
		//r16s11 * 2 -> r16s10, change format
		//r16s10 - r16s10
		d0 = vqsubq_s16(r_x_abs, scale_5K_0);
		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);
	}
	else if(scale_K_q8 < r_x_abs)
	{
		//r16s11 - r16s11
		d0 = vqsubq_s16(r_x_abs, scale_2K_q8);
		//r16s11*r16s2 -> r32s13 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 13;
		d0 = X_sat_16(r_w32);
	}
	else
	{
		//r16s11 * 2 -> r16s10, change format
		//r16s10 - r16s10
		d0 = vqsubq_s16(r_x_abs, scale_3K_0);
		//r16s10*r16s2 -> r32s12 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 12;
		d0 = X_sat_16(r_w32);
	}

	//r16s0 -> sat -> r8s0
	return X_sat_8(d0);
}

#else
static __device__ int8_t QAM256_Scaling_soft_bit4_bit5_cuda(int16_t r_x, uint16_t r_x_abs, int Scaling)
{
//	const int16_t shf_table_1[6] = { 12, 13, 12, 12, 13, 12 };
	
	int16_t d0;
	//int16_t scale_K;
	int32_t r_w32;

	int index = 5;  // default
#if 1	
	for (int i = 4; i >= 0; i--) {
	    index = (r_x_abs > scale_table_b4b5_const[i]) ? i : index;
	}
	
	int16_t scale = scale_table_1_b0b1_const[index];

#else
	
	index = (r_x_abs > coe_scale_K_q8) ? 4 : index;
	index = (r_x_abs > coe_scale_3K_q8) ? 3 : index;
	index = (r_x_abs > coe_scale_4K_q8) ? 2 : index;
	index = (r_x_abs > coe_scale_5K_q8) ? 1 : index;
	index = (r_x_abs > coe_scale_7K_q8) ? 0 : index;

	int16_t scale = 0; // = scale_table_1_b0b1_const[index];
	scale = (index == 0) ? coe_scale_13K_q8 : scale; 
	scale = (index == 1) ? coe_scale_6K_q8 : scale; 
	scale = (index == 2) ? coe_scale_11K_q8 : scale; 
	scale = (index == 3) ? coe_scale_5K_0 : scale; 
	scale = (index == 4) ? coe_scale_2K_q8 : scale; 
	scale = (index == 5) ? coe_scale_3K_0 : scale; 
#endif

	//int16_t scale = scale_table_1_b4b5_const[index];
	int16_t temp0 = vqsubq_s16(r_x_abs, scale);
	int16_t temp1 = vqsubq_s16(scale, r_x_abs);
	
	d0 = (index > 2)? temp0 : temp1;
	
	int16_t shf_1;
	shf_1 = (index == 1 || index == 4) ? 13 : 12;

	r_w32 = d0*Scaling;
	r_w32 >>= shf_1;
	d0 = X_sat_16(r_w32);

	//r16s0 -> sat -> r8s0
	return X_sat_8(d0);
}
#endif

#if QAM256_OPT == 0
static __device__ int8_t QAM256_Scaling_soft_bit6_bit7_cuda(int16_t r_x, uint16_t r_x_abs, int Scaling)
{
	//QAM256
	const int16_t scale_K_q8 = 0x013A; //r16s11, 2/sqrt(170)
	const int16_t scale_2K_q8 = 0x0274; //r16s11, 4/sqrt(170)
	const int16_t scale_3K_q8 = 0x03AE; //r16s11, 6/sqrt(170)
	const int16_t scale_4K_q8 = 0x04E8; //r16s11, 8/sqrt(170)
	const int16_t scale_5K_q8 = 0x0622; //r16s11, 10/sqrt(170)
	const int16_t scale_6K_q8 = 0x075C; //r16s11, 12/sqrt(170)
	const int16_t scale_7K_q8 = 0x0897; //r16s11, 14/sqrt(170)
	const int16_t scale_9K_q8 = 0x0585; //r16s10, 18/sqrt(170)
	const int16_t scale_10K_q8 = 0x0311; //r16s9, 20/sqrt(170)
	const int16_t scale_11K_q8 = 0x06BF; //r16s10, 22/sqrt(170)
	const int16_t scale_13K_q8 = 0x07F9; //r16s10, 26/sqrt(170)
	const int16_t scale_15K_q8 = 0x0934; //r16s10, 30/sqrt(170)
	const int16_t scale_21K_q8 = 0x0338; //r16s8, 42/sqrt(170)
	const int16_t scale_22K_q8 = 0x06BF; //r16s9, 44/sqrt(170)
	const int16_t scale_28K_q8 = 0x044B; //r16s8, 56/sqrt(170)

	const int16_t scale_15K_0 = scale_15K_q8 >> 1;
	const int16_t scale_7K_0 = scale_7K_q8 >> 1;
	const int16_t scale_6K_0 = scale_6K_q8 >> 2;
	const int16_t scale_5K_0 = scale_5K_q8 >> 1;
	const int16_t scale_3K_0 = scale_3K_q8 >> 1;
	const int16_t scale_K_0 = scale_K_q8 >> 1;
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//int index = tid << 1;
	int index_output = tid << 3;
	
	//if (tid == 0)
	//		printf("with div \n");
	
	int16_t d0, d1;
	int16_t scale_K;
	int32_t r_w32;

	if (scale_6K_q8 < r_x_abs)
	{
		//r16s11 - r16s11
		d0 = vqsubq_s16(scale_7K_q8, r_x_abs);
		//r16s11*r16s2 -> r32s13 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 13;
		d0 = X_sat_16(r_w32);
	}
	else if(scale_4K_q8 < r_x_abs)
	{
		//r16s11 - r16s11
		d0 = vqsubq_s16(r_x_abs, scale_5K_q8);
		//r16s11*r16s2 -> r32s13 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 13;
		d0 = X_sat_16(r_w32);
	}
	else if(scale_2K_q8 < r_x_abs)
	{
		//r16s11 - r16s11
		d0 = vqsubq_s16(scale_3K_q8, r_x_abs);
		//r16s11*r16s2 -> r32s13 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 13;
		d0 = X_sat_16(r_w32);
	}
	else
	{
		//r16s11 - r16s11
		d0 = vqsubq_s16(r_x_abs, scale_K_q8);
		//r16s11*r16s2 -> r32s13 -> r32s0 -> sat -> r16s0
		r_w32 = d0*Scaling;
		r_w32 >>= 13;
		d0 = X_sat_16(r_w32);
	}

	//r16s0 -> sat -> r8s0

	return X_sat_8(d0);
}
#else
static __device__ int8_t QAM256_Scaling_soft_bit6_bit7_cuda(int16_t r_x, uint16_t r_x_abs, int Scaling)
{

	int16_t d0;
	//int16_t scale_K;
	int32_t r_w32;

	int index = 3;  // default

#if 1	
	for (int i = 2; i >= 0; i--) {
	    index = (r_x_abs > scale_table_b6b7_const[i]) ? i : index;
	}

	int16_t scale = scale_table_1_b0b1_const[index];
	
#else	
 
	index = (r_x_abs > coe_scale_2K_q8) ? 2 : index;
	index = (r_x_abs > coe_scale_4K_q8) ? 1 : index;
	index = (r_x_abs > coe_scale_6K_q8) ? 0 : index;

	int16_t scale = 0; // = scale_table_1_b0b1_const[index];
	scale = (index == 0) ? coe_scale_7K_q8 : scale; 
	scale = (index == 1) ? coe_scale_5K_q8 : scale; 
	scale = (index == 2) ? coe_scale_3K_q8 : scale; 
	scale = (index == 3) ? coe_scale_K_q8 : scale; 
#endif

	//int16_t scale = scale_table_1_b6b7_const[index];
	int16_t temp0 = vqsubq_s16(r_x_abs, scale);
	int16_t temp1 = vqsubq_s16(scale, r_x_abs);

	d0 = (index & 0x1)? temp0 : temp1;

	r_w32 = d0*Scaling;
	r_w32 >>= 13;
	d0 = X_sat_16(r_w32);

	//r16s0 -> sat -> r8s0

	return X_sat_8(d0);
}
#endif

#if QAM256_OPT == 0
__global__ void QAM256_Kernel(uint8_t* pSb_deMod, uint32_t* pModSymb, uint8_t* pScalingIdxMap, uint32_t NumOfModSymb)
{
	int Scaling;
	int i;
	int16_t* pModSymb_w16;
	uint8_t* pSb_deMod_w8;
	int16_t r_x;
	int16_t r_y;
	uint16_t r_x_abs;
	uint16_t r_y_abs;

	pModSymb_w16 = (int16_t*)pModSymb;
	pSb_deMod_w8 = (uint8_t*)pSb_deMod;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int index = tid << 1;
	int index_output = tid << 3;

	//for(i = 0; i < NumOfModSymb; i++)
	if (tid < NumOfModSymb)
	{

		//r16s2
		uint8_t scaleIdx = pScalingIdxMap[tid];
		Scaling = d_In_ScaleLUT_const[scaleIdx];

		//r16s12
		r_x = pModSymb_w16[index];
		r_y = pModSymb_w16[index + 1];

		//r16s11
		r_x >>= 1;
		r_y >>= 1;

		//r16s11
		r_x_abs = (r_x > 0) ? r_x : (-r_x);
		r_y_abs = (r_y > 0) ? r_y : (-r_y);

		//soft bit 0 1
		pSb_deMod_w8[index_output] 		= QAM256_Scaling_soft_bit0_bit1_cuda(r_x, r_x_abs, Scaling);
		pSb_deMod_w8[index_output + 1] 	= QAM256_Scaling_soft_bit0_bit1_cuda(r_y, r_y_abs, Scaling);
		//soft bit 2 3
		pSb_deMod_w8[index_output + 2] 	= QAM256_Scaling_soft_bit2_bit3_cuda(r_x, r_x_abs, Scaling);
		pSb_deMod_w8[index_output + 3] 	= QAM256_Scaling_soft_bit2_bit3_cuda(r_y, r_y_abs, Scaling);
		//soft bit 4 5
		pSb_deMod_w8[index_output + 4] 	= QAM256_Scaling_soft_bit4_bit5_cuda(r_x, r_x_abs, Scaling);
		pSb_deMod_w8[index_output + 5] 	= QAM256_Scaling_soft_bit4_bit5_cuda(r_y, r_y_abs, Scaling);
		//soft bit 6 7
		pSb_deMod_w8[index_output + 6] 	= QAM256_Scaling_soft_bit6_bit7_cuda(r_x, r_x_abs, Scaling);
		pSb_deMod_w8[index_output + 7] 	= QAM256_Scaling_soft_bit6_bit7_cuda(r_y, r_y_abs, Scaling);
	}

	return;
}
#else
__global__ void QAM256_Kernel(uint8_t* pSb_deMod, uint32_t* pModSymb, uint8_t* pScalingIdxMap, uint32_t NumOfModSymb)
{
	int Scaling;
	//int i;
	//int16_t* pModSymb_w16;
	uint8_t* pSb_deMod_w8;
	//int16_t r_x;
	//int16_t r_y;
	uint16_t r_x_abs;
	uint16_t r_y_abs;
	uint32_t r;
	
	//pModSymb_w16 = (int16_t*)pModSymb;
	pSb_deMod_w8 = (uint8_t*)pSb_deMod;

	//uint32_t* pSb_deMod_u32 = (uint32_t*)pSb_deMod;

	uint64_t* pSb_deMod_u64 = (uint64_t*)pSb_deMod;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//int index = tid << 1;
	int index_output = tid << 3;
	
	//if (tid % 32 == 0)
	//	printf("%p %d \n", pSb_deMod_w8 + index_output, ((uintptr_t)(pSb_deMod_w8 + index_output)) % 256);
	

	//for(i = 0; i < NumOfModSymb; i++)
	if (tid < NumOfModSymb)
	{
		//r16s2
		uint8_t scaleIdx = pScalingIdxMap[tid];
		Scaling = d_In_ScaleLUT_const[scaleIdx];

		//r16s12
		//r_x = pModSymb_w16[index];
		//r_y = pModSymb_w16[index + 1];

		r = pModSymb[tid];
		int16_t r_x = (int16_t)(r & 0xffff);
		int16_t r_y = (int16_t)(r >> 16);

		//printf("r_x:%x, r_x1:%x, r_y:%x, r_y1: %x \n", r_x, r_x1, r_y, r_y1);

		//r16s11
		r_x >>= 1;
		r_y >>= 1;

		//r16s11
		r_x_abs = (r_x > 0) ? r_x : (-r_x);
		r_y_abs = (r_y > 0) ? r_y : (-r_y);

		//soft bit 0 1
		uint8_t a0	= QAM256_Scaling_soft_bit0_bit1_cuda(r_x, r_x_abs, Scaling);
		uint8_t a1 	= QAM256_Scaling_soft_bit0_bit1_cuda(r_y, r_y_abs, Scaling);
		//soft bit 2 3
		uint8_t a2 	= QAM256_Scaling_soft_bit2_bit3_cuda(r_x, r_x_abs, Scaling);
		uint8_t a3 	= QAM256_Scaling_soft_bit2_bit3_cuda(r_y, r_y_abs, Scaling);
		//soft bit 4 5
		uint8_t a4 	= QAM256_Scaling_soft_bit4_bit5_cuda(r_x, r_x_abs, Scaling);
		uint8_t a5 	= QAM256_Scaling_soft_bit4_bit5_cuda(r_y, r_y_abs, Scaling);
		//soft bit 6 7
		uint8_t a6 	= QAM256_Scaling_soft_bit6_bit7_cuda(r_x, r_x_abs, Scaling);
		uint8_t a7 	= QAM256_Scaling_soft_bit6_bit7_cuda(r_y, r_y_abs, Scaling);

#if 0
		pSb_deMod_w8[index_output] = a0;
		pSb_deMod_w8[index_output + 1] = a1;
		pSb_deMod_w8[index_output + 2] = a2;
		pSb_deMod_w8[index_output + 3] = a3;
		pSb_deMod_w8[index_output + 4] = a4;
		pSb_deMod_w8[index_output + 5] = a5;
		pSb_deMod_w8[index_output + 6] = a6;
		pSb_deMod_w8[index_output + 7] = a7;
#else		

		uint32_t upper = ((uint32_t) a3) << 24;
		upper = upper | (((uint32_t) a2) << 16);
		upper = upper | (((uint32_t) a1) << 8);
		upper = upper | (((uint32_t) a0));
		
		uint32_t lower = (((uint32_t) a7) << 24);
		lower = lower | (((uint32_t) a6) << 16);
		lower = lower | (((uint32_t) a5) << 8);
		lower = lower | (((uint32_t) a4));
		
		pSb_deMod_u64[tid] = (((uint64_t)lower) << 32) | ((uint64_t)upper);
		//pSb_deMod_u64[index + 1] = lower;
#endif
	}

	return;
}
#endif

#define NSTREAM 4
#define BDIM 128

int main(int argc, char **argv)
{
    printf("> %s Starting...\n", argv[0]);
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (prop.deviceOverlap) // compute and data transmission overlap
		printf("Dev overlap： %d \n", prop.deviceOverlap);
	if (prop.concurrentKernels) // concurrent kernels enable
		printf("Concurrent kernels: %d \n", prop.concurrentKernels);
	
#if QAM256_OPT == 0
	printf("QAM256 OPT disabled \n");
#else
	printf("QAM256 OPT enabled \n");
#endif

	
    // set up data size of vectors
    int nElem = 1 << 18;
    printf("> vector size = %d\n", nElem);

    uint32_t *h_A;
	uint8_t *h_B; 
	uint8_t *hostRef, *gpuRef;
	//int16_t *h_ScaleLUT;
	QAM_alloc_host_mem((void **)&h_A, nElem * 4); // uint32_t
	QAM_alloc_host_mem((void **)&h_B, nElem);
	QAM_alloc_host_mem((void **)&hostRef, nElem * 8); // 1 uint32_t will have 8 soft bytes
	QAM_alloc_host_mem((void **)&gpuRef, nElem * 8);

	uint32_t *d_A;
	uint8_t *d_B;
	uint8_t *d_C;

	QAM_alloc_device_mem((void **)&d_A, nElem * 4);
	QAM_alloc_device_mem((void **)&d_B, nElem);
	QAM_alloc_device_mem((void **)&d_C, nElem * 8);

    // initialize data at host side
    initialData_u32(h_A, nElem);
    initialData_u8(h_B, nElem);
	
    memset(hostRef, 0, nElem * 8);
    memset(gpuRef,  0, nElem * 8);

    // add vector at host side for result checks
    //sumArraysOnHost(h_A, h_B, hostRef, nElem);


    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

#if QAM256_OPT == 0
    dim3 block (1024);
    dim3 grid  ((nElem + block.x - 1) / block.x);
    printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x,
            block.y);
#else
    dim3 block (128);
    dim3 grid  ((nElem + block.x - 1) / block.x);
    printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x,
            block.y);
#endif
			
			
    CHECK(cudaDeviceSynchronize());
	
    float memcpy_h2d_time;
    float kernel_time;
    float memcpy_d2h_time;
	float itotal;
	// copy constant buffer
	QAM_init_device_const_mem();
   	
    CHECK(cudaEventRecord(start, 0));

    CHECK(cudaMemcpy(d_A, h_A, nElem * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nElem, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

	CHECK(cudaEventRecord(start, 0));
    QAM256_Kernel<<<grid, block>>>(d_C, d_A, d_B, nElem);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(hostRef, d_C, nElem * 8, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
    itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;

	printf("First run: \n");
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device\t: %f ms (%f GB/s)\n",
           memcpy_h2d_time, (nElem * 1e-6) / memcpy_h2d_time);
    printf(" Memcpy device to host\t: %f ms (%f GB/s)\n",
           memcpy_d2h_time, (nElem * 1e-6) / memcpy_d2h_time);
    printf(" Kernel\t\t\t: %f ms (%f GB/s)\n",
           kernel_time, (nElem * 2e-6) / kernel_time);
    printf(" Total\t\t\t: %f ms (%f GB/s)\n",
           itotal, (nElem * 2e-6) / itotal);

	// 2nd run
    CHECK(cudaEventRecord(start, 0));

    CHECK(cudaMemcpy(d_A, h_A, nElem * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nElem, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

	CHECK(cudaEventRecord(start, 0));
    QAM256_Kernel<<<grid, block>>>(d_C, d_A, d_B, nElem);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(hostRef, d_C, nElem * 8, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
    itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;

    printf("second run \n");
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device\t: %f ms (%f GB/s)\n",
           memcpy_h2d_time, (nElem * 1e-6) / memcpy_h2d_time);
    printf(" Memcpy device to host\t: %f ms (%f GB/s)\n",
           memcpy_d2h_time, (nElem * 1e-6) / memcpy_d2h_time);
    printf(" Kernel\t\t\t: %f ms (%f GB/s)\n",
           kernel_time, (nElem * 2e-6) / kernel_time);
    printf(" Total\t\t\t: %f ms (%f GB/s)\n",
           itotal, (nElem * 2e-6) / itotal);

   // grid parallel operation
    int iElem = nElem / NSTREAM;
    size_t iBytes = iElem;
    grid.x = (iElem + block.x - 1) / block.x;	

    cudaStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    CHECK(cudaEventRecord(start, 0));
	

    // initiate all asynchronous transfers to the device
    for (int i = 0; i < NSTREAM; ++i)
    {
        int ioffset = i * iElem;
        CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes * 4, cudaMemcpyHostToDevice, stream[i]));
        CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
    }

    // launch a kernel in each stream
    for (int i = 0; i < NSTREAM; ++i)
    {
        int ioffset = i * iElem;
        QAM256_Kernel<<<grid, block, 0, stream[i]>>>(&d_C[ioffset * 8], &d_A[ioffset], &d_B[ioffset], iElem);
    }

    // enqueue asynchronous transfers from the device
    for (int i = 0; i < NSTREAM; ++i)
    {
        int ioffset = i * iElem * 8;
        CHECK(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iElem * 8,
                              cudaMemcpyDeviceToHost, stream[i]));
    }

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float execution_time;
    CHECK(cudaEventElapsedTime(&execution_time, start, stop));

    printf("parallel \n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM,
           execution_time, (nElem * 2e-6) / execution_time );
    printf(" speedup                : %f \n",
           ((itotal - execution_time) * 100.0f) / itotal);

    // check kernel error
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(start, 0));

    // initiate all asynchronous transfers to the device
    for (int i = 0; i < NSTREAM; ++i)
    {
        int ioffset = i * iElem;
        CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes * 4, cudaMemcpyHostToDevice, stream[i]));
        CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
   
        ioffset = i * iElem;
        QAM256_Kernel<<<grid, block, 0, stream[i]>>>(&d_C[ioffset * 8], &d_A[ioffset], &d_B[ioffset], iElem);
    
        ioffset = i * iElem * 8;
        CHECK(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iElem * 8,
                              cudaMemcpyDeviceToHost, stream[i]));
    }

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    //float execution_time;
    CHECK(cudaEventElapsedTime(&execution_time, start, stop));

    printf("sequential: \n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM,
           execution_time, (nElem * 2e-6) / execution_time );
    printf(" speedup                : %f \n",
           ((itotal - execution_time) * 100.0f) / itotal);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nElem * 8);
	
	QAM_free_host_mem(h_A);
	QAM_free_host_mem(h_B);
	QAM_free_host_mem(hostRef);
	QAM_free_host_mem(gpuRef);

	QAM_free_device_mem(d_A);
	QAM_free_device_mem(d_B);
	QAM_free_device_mem(d_C);

	// destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

	
	return 0;
}