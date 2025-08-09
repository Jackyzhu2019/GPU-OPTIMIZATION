
void mimo64_neon_kernel(float *G, float *Y, float *X, int nElem, int nElemPerMatrix)
{
	int numOfloop = nElem / nElemPerMatrix;
	int loopIdx;
	int iRow, iCol, iK;
	float *G0, *Y0, *X0;
	int i, j, k, t;
	float G_cache[64][64];
	float Y_cache[4][64];
	
	for (loopIdx = 0; loopIdx < numOfloop; loopIdx++)
	{
		G0 = G + 64 * 64 * loopIdx;
		Y0 = Y + 64 * nElemPerMatrix * loopIdx;
		X0 = X + 64 * nElemPerMatrix * loopIdx;
		
		float *dst = &G_cache[0][0];

		// prefetch the data to L1 cahce
		for (int i = 0; i < 64; i++)
		{
			for (int j = 0; j < 64; j+=16)
			{
				float32x4_t tempG0_f32x4 = vld1q_f32(G0);
				float32x4_t tempG1_f32x4 = vld1q_f32(G0 + 4);
				float32x4_t tempG2_f32x4 = vld1q_f32(G0 + 8);
				float32x4_t tempG3_f32x4 = vld1q_f32(G0 + 12);

				vst1q_f32(dst, tempG0_f32x4);
				vst1q_f32(dst + 4, tempG1_f32x4);
				vst1q_f32(dst + 8, tempG2_f32x4);
				vst1q_f32(dst + 12, tempG3_f32x4);

				dst += 16;
				G0 += 16;
			}			
		}
		

		dst = &Y_cache[0][0];
		for (int i = 0; i < 64; i += 4) {
			float32x4_t row0 = vld1q_f32(Y0 + i * 4);
			float32x4_t row1 = vld1q_f32(Y0 + (i + 1) * 4);
			float32x4_t row2 = vld1q_f32(Y0 + (i + 2) * 4);
			float32x4_t row3 = vld1q_f32(Y0 + (i + 3) * 4);

			float32x4x2_t tmp0 = vtrnq_f32(row0, row1);
			float32x4x2_t tmp1 = vtrnq_f32(row2, row3);
			float32x4_t col0 = vcombine_f32(vget_low_f32(tmp0.val[0]), vget_low_f32(tmp1.val[0]));
			float32x4_t col1 = vcombine_f32(vget_low_f32(tmp0.val[1]), vget_low_f32(tmp1.val[1]));
			float32x4_t col2 = vcombine_f32(vget_high_f32(tmp0.val[0]), vget_high_f32(tmp1.val[0]));
			float32x4_t col3 = vcombine_f32(vget_high_f32(tmp0.val[1]), vget_high_f32(tmp1.val[1]));

			vst1q_f32(dst + i, col0);
			vst1q_f32(dst + 64 + i, col1);
			vst1q_f32(dst + 128 + i, col2);
			vst1q_f32(dst + 192 + i, col3);
		}
		
		
		// G matrix (64 x 64) x Y matrix (64 x nElemPerMatrix(4))
		for (iRow = 0; iRow < 64; iRow += BLOCK_SIZE_ROW)
		{
			for (iCol = 0; iCol < nElemPerMatrix; iCol += BLOCK_SIZE_COL)
			{
			    float32x4_t acc[BLOCK_SIZE_ROW][BLOCK_SIZE_COL];
    
				// init
				for (int i = 0; i < BLOCK_SIZE_ROW; i++) {
					for (int j = 0; j < BLOCK_SIZE_COL; j++) {
						acc[i][j] = vmovq_n_f32(0.0f);
					}
				}
				
				for (iK = 0; iK < 64; iK += BLOCK_SIZE_K)
				{
					
					for (i = 0; i < BLOCK_SIZE_ROW; i++)
					{	
						for (j = 0; j < BLOCK_SIZE_COL; j++)
						{	
							for (k = 0; k < BLOCK_SIZE_K; k+=4){
								// accuVal[i][j] += G_cache[iRow + i][iK + k] * Y_cache[iCol + j][iK + k];	
								
								float32x4_t y_val = vld1q_f32(&Y_cache[iCol + j][iK + k]);
								float32x4_t g_val = vld1q_f32(&G_cache[iRow + i][iK + k]);

								acc[i][j] = vmlaq_f32(acc[i][j], g_val, y_val);

							}
						}
					}
				}
				
				// store
				for (i = 0; i < BLOCK_SIZE_ROW; i++)
				{
					for (j = 0; j < BLOCK_SIZE_ROW; j++)
					{
						float32x2_t sum = vadd_f32(vget_low_f32(acc[i][j]), vget_high_f32(acc[i][j]));					
						X0[(iRow + i) * nElemPerMatrix + iCol + j] = vget_lane_f32(vpadd_f32(sum, sum), 0);
					}
				}
	
			}
		}
	}


	return;
}
