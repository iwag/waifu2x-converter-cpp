/* -*- mode: c++ -*- */

#define UNROLL9(F)				\
	F(0);					\
	F(1);					\
	F(2);					\
	F(3);					\
	F(4);					\
	F(5);					\
	F(6);					\
	F(7);					\
	F(8);					\


#define UNROLL8x3x3(F)				\
	F(0,0,0);				\
	F(0,0,1);				\
	F(0,0,2);				\
	F(0,1,0);				\
	F(0,1,1);				\
	F(0,1,2);				\
	F(0,2,0);				\
	F(0,2,1);				\
	F(0,2,2);				\
						\
	F(1,0,0);				\
	F(1,0,1);				\
	F(1,0,2);				\
	F(1,1,0);				\
	F(1,1,1);				\
	F(1,1,2);				\
	F(1,2,0);				\
	F(1,2,1);				\
	F(1,2,2);				\
						\
	F(2,0,0);				\
	F(2,0,1);				\
	F(2,0,2);				\
	F(2,1,0);				\
	F(2,1,1);				\
	F(2,1,2);				\
	F(2,2,0);				\
	F(2,2,1);				\
	F(2,2,2);				\
						\
	F(3,0,0);				\
	F(3,0,1);				\
	F(3,0,2);				\
	F(3,1,0);				\
	F(3,1,1);				\
	F(3,1,2);				\
	F(3,2,0);				\
	F(3,2,1);				\
	F(3,2,2);				\
						\
	F(4,0,0);				\
	F(4,0,1);				\
	F(4,0,2);				\
	F(4,1,0);				\
	F(4,1,1);				\
	F(4,1,2);				\
	F(4,2,0);				\
	F(4,2,1);				\
	F(4,2,2);				\
						\
	F(5,0,0);				\
	F(5,0,1);				\
	F(5,0,2);				\
	F(5,1,0);				\
	F(5,1,1);				\
	F(5,1,2);				\
	F(5,2,0);				\
	F(5,2,1);				\
	F(5,2,2);				\
						\
	F(6,0,0);				\
	F(6,0,1);				\
	F(6,0,2);				\
	F(6,1,0);				\
	F(6,1,1);				\
	F(6,1,2);				\
	F(6,2,0);				\
	F(6,2,1);				\
	F(6,2,2);				\
						\
	F(7,0,0);				\
	F(7,0,1);				\
	F(7,0,2);				\
	F(7,1,0);				\
	F(7,1,1);				\
	F(7,1,2);				\
	F(7,2,0);				\
	F(7,2,1);				\
	F(7,2,2);				\

#define UNROLL8(F)				\
	F(0);					\
	F(1);					\
	F(2);					\
	F(3);					\
	F(4);					\
	F(5);					\
	F(6);					\
	F(7);					\


#define UNROLL8x3(F)				\
	F(0,0);					\
	F(0,1);					\
	F(0,2);					\
	F(0,3);					\
	F(0,4);					\
	F(0,5);					\
	F(0,6);					\
	F(0,7);					\
						\
	F(1,0);					\
	F(1,1);					\
	F(1,2);					\
	F(1,3);					\
	F(1,4);					\
	F(1,5);					\
	F(1,6);					\
	F(1,7);					\
						\
	F(2,0);					\
	F(2,1);					\
	F(2,2);					\
	F(2,3);					\
	F(2,4);					\
	F(2,5);					\
	F(2,6);					\
	F(2,7);					\


#define UNROLL10x3(F)				\
	F(0,0);					\
	F(0,1);					\
	F(0,2);					\
	F(0,3);					\
	F(0,4);					\
	F(0,5);					\
	F(0,6);					\
	F(0,7);					\
	F(0,8);					\
	F(0,9);					\
						\
	F(1,0);					\
	F(1,1);					\
	F(1,2);					\
	F(1,3);					\
	F(1,4);					\
	F(1,5);					\
	F(1,6);					\
	F(1,7);					\
	F(1,8);					\
	F(1,9);					\
						\
	F(2,0);					\
	F(2,1);					\
	F(2,2);					\
	F(2,3);					\
	F(2,4);					\
	F(2,5);					\
	F(2,6);					\
	F(2,7);					\
	F(2,8);					\
	F(2,9);					\


#define BLOCK_SIZE 8

extern "C" __global__ void
filter(const float * __restrict__ packed_input,
       int nInputPlanes,
       float * __restrict__ packed_output,
       int nOutputPlanes,
       const float * __restrict__ biases,
       unsigned int hsz,
       unsigned int wsz,
       const float * __restrict__ weight)
{
	extern __shared__ float shared_buf[];

	unsigned int yi = blockIdx.x;

	size_t in_step = wsz * nInputPlanes;
	const float *inp = packed_input;
	inp += yi * in_step;

	const float *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}
	const float *in1p = inp;

	const float *in2p = inp + in_step;
	if (yi == hsz-1) {
		in2p = in1p;
	}

	const float *in01 = in0p;
	const float *in11 = in1p;
	const float *in21 = in2p;

	float *shared_ptr = shared_buf;
	float *in_block0_base = shared_ptr;
	shared_ptr += nInputPlanes*(BLOCK_SIZE+2);
	float *in_block1_base = shared_ptr;
	shared_ptr += nInputPlanes*(BLOCK_SIZE+2);
	float *in_block2_base = shared_ptr;
	shared_ptr += nInputPlanes*(BLOCK_SIZE+2);

	float *in_block0 = in_block0_base + nInputPlanes;
	float *in_block1 = in_block1_base + nInputPlanes;
	float *in_block2 = in_block2_base + nInputPlanes;
	int lid = threadIdx.x;
	float bv0 = biases[lid*2+0];
	float bv1 = biases[lid*2+1];

	for (int xi0=0; xi0<wsz; xi0+=BLOCK_SIZE) {
		/*for (unsigned int op=0; op<nOutputPlanes; op++) thread */
		{
			int op = lid*2;
			int rem = wsz - xi0;
			__syncthreads();
			if (lid < nInputPlanes/2) {
				int bi;
				int lid2 = lid*2;
				for (bi=0; bi<BLOCK_SIZE; bi++) {
					int xi = xi0 + bi;
					if (xi == wsz) {
						break;
					}

					/* load to shared */
					*(float2*)&in_block0[bi*nInputPlanes + lid2] = *(float2*)&in01[xi*nInputPlanes + lid2];
					*(float2*)&in_block1[bi*nInputPlanes + lid2] = *(float2*)&in11[xi*nInputPlanes + lid2];
					*(float2*)&in_block2[bi*nInputPlanes + lid2] = *(float2*)&in21[xi*nInputPlanes + lid2];
				}

				{
					int xi = xi0 + bi;
					if (xi == wsz) {
						*(float2*)&in_block0[bi*(int)nInputPlanes + lid2] = *(float2*)&in01[(xi-1)*(int)nInputPlanes + lid2];
						*(float2*)&in_block1[bi*(int)nInputPlanes + lid2] = *(float2*)&in11[(xi-1)*(int)nInputPlanes + lid2];
						*(float2*)&in_block2[bi*(int)nInputPlanes + lid2] = *(float2*)&in21[(xi-1)*(int)nInputPlanes + lid2];
					} else {
						*(float2*)&in_block0[bi*(int)nInputPlanes + lid2] = *(float2*)&in01[xi*(int)nInputPlanes + lid2];
						*(float2*)&in_block1[bi*(int)nInputPlanes + lid2] = *(float2*)&in11[xi*(int)nInputPlanes + lid2];
						*(float2*)&in_block2[bi*(int)nInputPlanes + lid2] = *(float2*)&in21[xi*(int)nInputPlanes + lid2];
					}
				}

				{
					int xi = xi0-1;
					if (xi == -1) {
						*(float2*)&in_block0[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in01[lid2];
						*(float2*)&in_block1[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in11[lid2];
						*(float2*)&in_block2[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in21[lid2];
					} else {
						*(float2*)&in_block0[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in01[xi*(int)nInputPlanes + lid2];
						*(float2*)&in_block1[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in11[xi*(int)nInputPlanes + lid2];
						*(float2*)&in_block2[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in21[xi*(int)nInputPlanes + lid2];
					}
				}
			}
			__syncthreads();

			if (0 && rem >= BLOCK_SIZE) {
#if 0

#define DECL_PTR(y,x)		float *p##y##x = &in_block##y[nInputPlanes * (x-1)];

				UNROLL10x3(DECL_PTR);

				float sum00 = 0;
				float sum01 = 0;
				float sum02 = 0;
				float sum03 = 0;
				float sum04 = 0;
				float sum05 = 0;
				float sum06 = 0;
				float sum07 = 0;

				float sum10 = 0;
				float sum11 = 0;
				float sum12 = 0;
				float sum13 = 0;
				float sum14 = 0;
				float sum15 = 0;
				float sum16 = 0;
				float sum17 = 0;

				{
					const float *w0 = weight + lid;

					for (int ip = 0; ip < nInputPlanes; ip++) {
#define LOAD_INPUT2(y,x)			float2 i##y##x##_2 = *(float2*)&p##y##x[ip];

						UNROLL10x3(LOAD_INPUT2);

#define LOAD_COEF(X)							\
						float w0_##X = w[X * 128]; \
						float w1_##X = w[X * 128];

#define CALC(IDX,Y,I0,I1,I2,I3,I4,I5,I6,I7)				\
						sum0 += w_##IDX * i##Y##I0; \
						sum1 += w_##IDX * i##Y##I1; \
						sum2 += w_##IDX * i##Y##I2; \
						sum3 += w_##IDX * i##Y##I3; \
						sum4 += w_##IDX * i##Y##I4; \
						sum5 += w_##IDX * i##Y##I5; \
						sum6 += w_##IDX * i##Y##I6; \
						sum7 += w_##IDX * i##Y##I7;


						{
#define LOAD_INPUT1X(Y,X)				float i##Y##X = i##Y##X##_2.x;

							UNROLL10x3(LOAD_INPUT1X);

							const float *w = (w0 + (ip * 128) * 9);
							UNROLL9(LOAD_COEF);

							{
								CALC(0,0,0,1,2,3,4,5,6,7);
								CALC(1,0,1,2,3,4,5,6,7,8);
								CALC(2,0,2,3,4,5,6,7,8,9);

								CALC(3,1,0,1,2,3,4,5,6,7);
								CALC(4,1,1,2,3,4,5,6,7,8);
								CALC(5,1,2,3,4,5,6,7,8,9);

								CALC(6,2,0,1,2,3,4,5,6,7);
								CALC(7,2,1,2,3,4,5,6,7,8);
								CALC(8,2,2,3,4,5,6,7,8,9);
							}
						}

						ip++;
						{
#define LOAD_INPUT1Y(Y,X)				float i##Y##X = i##Y##X##_2.y;

							UNROLL10x3(LOAD_INPUT1Y);

							const float *w = (w0 + (ip * 128) * 9);
							UNROLL9(LOAD_COEF);

							{
								CALC(0,0,0,1,2,3,4,5,6,7);
								CALC(1,0,1,2,3,4,5,6,7,8);
								CALC(2,0,2,3,4,5,6,7,8,9);

								CALC(3,1,0,1,2,3,4,5,6,7);
								CALC(4,1,1,2,3,4,5,6,7,8);
								CALC(5,1,2,3,4,5,6,7,8,9);

								CALC(6,2,0,1,2,3,4,5,6,7);
								CALC(7,2,1,2,3,4,5,6,7,8);
								CALC(8,2,2,3,4,5,6,7,8,9);
							}
						}

					}

#define RELU(BI)							\
					{				\
						float *out = packed_output + (yi*wsz + (xi0+BI))*nOutputPlanes; \
									\
						{			\
							int opIndex = lid; \
							float v = sum##BI; \
							v += bv;	\
									\
							float mtz = max(v, 0.0f); \
							float ltz = min(v, 0.0f); \
									\
							v = ltz * 0.1f + mtz; \
									\
							out[opIndex] = v; \
						}			\
					}

					UNROLL8(RELU);
				}
#endif
			} else {
				for (int bi=0; bi<BLOCK_SIZE; bi++) {
					int xi = xi0+bi;
					if (xi == wsz) {
						break;
					}

					const float *w0 = weight + lid*2;

					float sum0 = 0;
					float sum1 = 0;

					for (int ip=0; ip<nInputPlanes; ip++) {
						float i00, i01, i02;
						float i10, i11, i12;
						float i20, i21, i22;

						i00 = in_block0[(bi-1)*nInputPlanes+ip];
						i10 = in_block1[(bi-1)*nInputPlanes+ip];
						i20 = in_block2[(bi-1)*nInputPlanes+ip];

						i01 = in_block0[bi*nInputPlanes+ip];
						i11 = in_block1[bi*nInputPlanes+ip];
						i21 = in_block2[bi*nInputPlanes+ip];

						i02 = in_block0[(bi+1)*nInputPlanes+ip];
						i12 = in_block1[(bi+1)*nInputPlanes+ip];
						i22 = in_block2[(bi+1)*nInputPlanes+ip];

						const float *w = w0;

						float2 w0 = *(float2*)&w[(9*ip+0) * 128];
						float2 w1 = *(float2*)&w[(9*ip+1) * 128];
						float2 w2 = *(float2*)&w[(9*ip+2) * 128];
						float2 w3 = *(float2*)&w[(9*ip+3) * 128];
						float2 w4 = *(float2*)&w[(9*ip+4) * 128];
						float2 w5 = *(float2*)&w[(9*ip+5) * 128];
						float2 w6 = *(float2*)&w[(9*ip+6) * 128];
						float2 w7 = *(float2*)&w[(9*ip+7) * 128];
						float2 w8 = *(float2*)&w[(9*ip+8) * 128];

						sum0 += w0.x*i00;
						sum0 += w1.x*i01;
						sum0 += w2.x*i02;

						sum0 += w3.x*i10;
						sum0 += w4.x*i11;
						sum0 += w5.x*i12;

						sum0 += w6.x*i20;
						sum0 += w7.x*i21;
						sum0 += w8.x*i22;


						sum1 += w0.y*i00;
						sum1 += w1.y*i01;
						sum1 += w2.y*i02;

						sum1 += w3.y*i10;
						sum1 += w4.y*i11;
						sum1 += w5.y*i12;

						sum1 += w6.y*i20;
						sum1 += w7.y*i21;
						sum1 += w8.y*i22;
					}

					float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;

					{
						float v = sum0;
						v += bv0;

						float mtz = max(v, 0.0f);
						float ltz = min(v, 0.0f);

						v = ltz * 0.1f + mtz;
						out[op] = v;
					}

					{
						float v = sum1;
						v += bv1;

						float mtz = max(v, 0.0f);
						float ltz = min(v, 0.0f);

						v = ltz * 0.1f + mtz;
						out[op+1] = v;
					}
				}
			}
		}
	}
}

