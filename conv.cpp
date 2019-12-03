#include "conv.h"
#include <ap_int.h>
#include <stdio.h>
#ifdef __SYNTHESIS__
#define BLOCKTYPE ap_fixed<16, 4>
#define BtZTYPE ap_fixed<18, 7>
#define UTYPE ap_fixed<18, 11>
#define WTYPE ap_fixed<16, -3>
#define UVTYPE ap_fixed<34, 8>
// #define WTYPE ap_fixed<16, 0>
// #define OUTTYPE ap_fixed<32, 3>
#define OUTTYPE ap_fixed<32, 6>
#else
// #define BLOCKTYPE float
// #define BtZTYPE float
// #define UTYPE float
// #define WTYPE float
// #define UVTYPE float
// #define OUTTYPE float
#define BLOCKTYPE ap_fixed<16, 4>
#define BtZTYPE ap_fixed<18, 7>
#define UTYPE ap_fixed<18, 11>
#define WTYPE ap_fixed<16, 0>
#define UVTYPE ap_fixed<34, 11>
#define OUTTYPE ap_fixed<32, 11>
#endif
const int Bt[6][6] = {
	{4, 0, -5, 0, 1, 0},
	{0, -4, -4, 1, 1, 0},
	{0, 4, -4, -1, 1, 0},
	{0, -2, -1, 2, 1, 0},
	{0, 2, -1, -2, 1, 0},
	{0, 4, 0, -5, 0, 1}
};
const int At[4][6] = {
	{1, 1, 1, 1, 1, 0},
	{0, 1, -1, 2, -2, 0},
	{0, 1, 1, 4, 4, 0},
	{0, 1, -1, 8, -8, 1}
};
const unsigned int bCHout = 64;
const unsigned int bCHin = 16;
const unsigned int bR_in = 6;
const unsigned int bC_in = 6;
const unsigned int KMax = 6;
const unsigned int SMin = 1;
const float qut = 1.;  //52.
const float qutw = 1.; //3592.
const int qdiv = 0;	//3
const float quto = (qut * qutw / (float)(1 << qdiv));
const float invquto = 1. / quto;
// 为方便起见, 在block计算时默认增加Padding, 即bR_out=bR_in, 不过Out_1中只有部分会被使用, 算完后抛弃多余的.
const unsigned int bR_out = 4;
const unsigned int bC_out = 4;
const unsigned int vbR_out = 4;
const unsigned int vbC_out = 4;
const unsigned int vbR_in = 4;
const unsigned int vbC_in = 4;
unsigned CHin, CHout, R_in, C_in;
const unsigned int K = 3;
const unsigned int Kg = 6;
const unsigned int Ki = 6;
const unsigned int Ko = 4;
const unsigned int S = 1;

inline WTYPE w_to_int8(d_type x)
{
	WTYPE y = 0;
	y = x * qutw;
	return y;
}

inline BLOCKTYPE in_to_int8(d_type x)
{
	BLOCKTYPE y = 0;
	y = x * qut;
	return y;
}

inline OUTTYPE out_to_int16(d_type x)
{
	OUTTYPE y = 0;
	y = x * (quto);
	return y;
}

inline float out_to_float32(OUTTYPE x)
{
	float y = float(x) * invquto;
	return y;
}

void load_w(d_type *W, WTYPE W_1[bCHout][bCHin][KMax][KMax], unsigned CHout_batch, unsigned CHin_batch, unsigned offset)
{
loop_W:
	for (unsigned i = 0; i < bCHout && i + CHout_batch < CHout; i++)
	{
		for (unsigned j = 0; j < bCHin && j + CHin_batch < CHin; j++)
		{
			for (unsigned k = 0; k < Kg; k++)
			{
				for (unsigned l = 0; l < Kg; l++)
				{
#pragma HLS PIPELINE
					W_1[i][j][k][l] = W[offset + i * (CHin * Kg * Kg) + j * Kg * Kg + k * Kg + l];
				}
			}
		}
	}
}
void load_in(d_type *In, BLOCKTYPE In_1[bCHin][bR_in][bC_in], unsigned R_in_batch, unsigned C_in_batch, unsigned CHin_batch, unsigned offset)
{
loop_In:
	for (unsigned j = 0; j < bR_in && j + R_in_batch < R_in; j++)
	{
		for (unsigned k = 0; k < bC_in && k + C_in_batch < C_in; k++)
		{
			for (unsigned i = 0; i < bCHin && i + CHin_batch < CHin; i++)
			{
#pragma HLS PIPELINE
				In_1[i][j][k] = In[offset + i * (R_in * C_in) + j * C_in + k];
			}
		}
	}
}

inline void ZtoU(BLOCKTYPE Z[6][6], UTYPE U[6][6])
{
	#pragma HLS inline
	BtZTYPE BtZ[6][6]={};

	for(int i = 0; i < 6; i++)
	{
	#pragma HLS unroll
		for(int j = 0; j < 6; j++)
		{
		#pragma HLS unroll
			U[i][j] = 0;
		}
	}

	for(int i = 0; i < 6; i++)
	{
		#pragma HLS unroll
		for(int j = 0; j < 6; j++)
		{
			#pragma HLS unroll
			for(int k = 0; k < 6; k++)
			{
				#pragma HLS unroll
				BtZ[i][j] += Bt[i][k] * Z[k][j];
			}
		}
	}

	for(int i = 0; i < 6; i++)
	{
		#pragma HLS unroll
		for(int j = 0; j < 6; j++)
		{
			#pragma HLS unroll
			for(int k = 0; k < 6; k++)
			{
				#pragma HLS unroll
				U[i][j] += BtZ[i][k] * Bt[j][k];
			}
		}
	}
}

inline void UpointV(UTYPE U[6][6], WTYPE V[6][6], UVTYPE UV[6][6])
{
	#pragma HLS inline
	for (int i = 0; i < 6; i++)
	{
		#pragma HLS unroll
		for (int j = 0; j < 6; j++)
		{
			#pragma HLS unroll
			UV[i][j] = U[i][j] * V[i][j];
		}
	}
}

inline void UVtoY(UVTYPE UV[6][6], OUTTYPE Y[4][4])
{
	#pragma HLS inline
	UVTYPE AtUV[4][6]={};

	for(int i = 0; i < 4; i++)
	{
		#pragma HLS unroll
		for(int j = 0; j < 6; j++)
		{
			#pragma HLS unroll
			for(int k = 0; k < 6; k++)
			{
				#pragma HLS unroll
				AtUV[i][j] += At[i][k] * UV[k][j];
			}
		}
	}

	for(int i = 0; i < 4; i++)
	{
		#pragma HLS unroll
		for(int j = 0; j < 4; j++)
		{
			#pragma HLS unroll
			for(int k = 0; k < 6; k++)
			{
				#pragma HLS unroll
				Y[i][j] += AtUV[i][k] * At[j][k];
			}
		}
	}	
}

void conv_batch(BLOCKTYPE In_1[bCHin][bR_in][bC_in], OUTTYPE Out_1[bCHout][bR_out][bC_out], WTYPE W_1[bCHout][bCHin][KMax][KMax], ap_int<8> CHin_batch)
{
	if (CHin_batch)
	{
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 5
	loop_CHin:
		for (unsigned chi = 0; chi < bCHin && chi + (CHin_batch - bCHin) < CHin; chi++)
		{
			UTYPE U[6][6];
			ZtoU(In_1[chi], U);
		loop_CHout:
			for (unsigned cho = 0; cho < bCHout; cho++)
			{
				UVTYPE UV[6][6];
				UpointV(U, W_1[cho][chi], UV);
				UVtoY(UV, Out_1[cho]);
			}
		}
	}
}
void cnn(d_type *In, d_type *Out, d_type *W, int *Parameter)
{
#pragma HLS ALLOCATION instances = fmul limit = 32 operation
#pragma HLS ALLOCATION instances = fadd limit = 32 operation

/*
	In  : Input feature map, CHin*R*C
	Out : Output feature map, CHout*Rout*Cout
	W : weights, CHout*CHin*6*6
	Parameter:  CHin|CHout|R|C|K|S
	*/
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS INTERFACE m_axi depth = 256 port = In offset = slave //adjust the depth as you need
#pragma HLS INTERFACE m_axi depth = 256 port = Out offset = slave
#pragma HLS INTERFACE m_axi depth = 256 port = W offset = slave
#pragma HLS INTERFACE m_axi depth = 256 port = Parameter offset = slave

	// 当前block size :
	/*
	CHin : 32
	CHout : 64
	R_in : 128
	C_in : 128
	R_out : ?
	C_out : ?
	K : 3
	S : 1
	*/

	BLOCKTYPE In_1[bCHin][bR_in][bC_in];
	OUTTYPE Out_1[bCHout][bR_out][bC_out];
	BLOCKTYPE In_0[bCHin][bR_in][bC_in];
	WTYPE W_0[bCHout][bCHin][KMax][KMax];
	WTYPE W_1[bCHout][bCHin][KMax][KMax];
// #pragma HLS RESOURCE variable=Out_1 core=RAM_1P_LUTRAM
// #pragma HLS ARRAY_PARTITION variable = In_1 cyclic factor = 4 dim = 2
// #pragma HLS ARRAY_PARTITION variable = Out_1 complete dim = 3
// #pragma HLS ARRAY_PARTITION variable = Out_1 complete
// #pragma HLS ARRAY_PARTITION variable = W_1 complete dim = 4
// #pragma HLS ARRAY_PARTITION variable = W_0 complete dim = 4
	// #pragma HLS ARRAY_PARTITION variable=W_1 complete

	/*
	CHin : Input channels
	CHout : output channels
	R_in : Input rows
	C_in : Input columns
	K : kernel size (Kr = Kc)
	S : Stride
	*/

	CHin = Parameter[0];
	CHout = Parameter[1];
	R_in = Parameter[2];
	C_in = Parameter[3];
	// K = Parameter[4];
	// S = Parameter[5];

	if (R_in - K < 0 || C_in - K < 0)
	{
		return;
	}

	unsigned R_out = (((unsigned)(R_in - K))) + 1;
	unsigned C_out = (((unsigned)(C_in - K))) + 1;

	for (unsigned R_in_batch = 0, R_out_batch = 0; R_out_batch < R_out; (R_in_batch += vbR_in), (R_out_batch += vbR_out))
	{
#pragma HLS LOOP_TRIPCOUNT max = 1
		for (unsigned C_in_batch = 0, C_out_batch = 0; C_out_batch < C_out; (C_in_batch += vbC_in), (C_out_batch += vbC_out))
		{
#pragma HLS LOOP_TRIPCOUNT max = 1
			for (unsigned CHout_batch = 0; CHout_batch < CHout; CHout_batch += bCHout)
			{
#pragma HLS LOOP_TRIPCOUNT max = 1
			loop_Out:
				for (unsigned r2 = 0; r2 < vbR_out && r2 + R_out_batch < R_out; r2++)
				{
#pragma HLS LOOP_TRIPCOUNT max = 32
					for (unsigned c2 = 0; c2 < vbC_out && c2 + C_out_batch < C_out; c2++)
					{
#pragma HLS LOOP_TRIPCOUNT max = 30
						for (unsigned cho = 0; cho < bCHout && cho + CHout_batch < CHout; cho++)
						{
#pragma HLS PIPELINE
							// Out_1[r2][c2][cho] = 0;
							Out_1[cho][r2][c2] = Out[(cho + CHout_batch) * R_out * C_out + (r2 + R_out_batch) * C_out + (c2 + C_out_batch)];
						}
					}
				}
				bool ping_pong_flag = 1;
				for (unsigned CHin_batch = 0; CHin_batch < CHin + bCHin /* ping pong add 1 */; CHin_batch += bCHin)
				{
#pragma HLS LOOP_TRIPCOUNT max = 5
					printf("FUCKYOU! %u %u %u %u\n", (unsigned)CHin_batch, (unsigned)CHout_batch, (unsigned)R_in_batch, (unsigned)C_in_batch);
					// #pragma HLS LOOP_FLATTEN OFF

					unsigned w_offset = CHout_batch * (CHin * Kg * Kg) + CHin_batch * (Kg * Kg);
					unsigned in_offset = CHin_batch * (R_in * C_in) + R_in_batch * C_in + C_in_batch;

					if (ping_pong_flag)
					{
						load_w(W, W_1, CHout_batch, CHin_batch, w_offset);
						load_in(In, In_1, R_in_batch, C_in_batch, CHin_batch, in_offset);
						conv_batch(In_0, Out_1, W_0, CHin_batch);
					}
					else
					{
						load_w(W, W_0, CHout_batch, CHin_batch, w_offset);
						load_in(In, In_0, R_in_batch, C_in_batch, CHin_batch, in_offset);
						conv_batch(In_1, Out_1, W_1, CHin_batch);
					}

					ping_pong_flag = !ping_pong_flag;
				}
			loop_AddedOut:
				for (unsigned r2 = 0; r2 < vbR_out && r2 + R_out_batch < R_out; r2++)
				{
#pragma HLS LOOP_TRIPCOUNT max = 32
					for (unsigned c2 = 0; c2 < vbC_out && c2 + C_out_batch < C_out; c2++)
					{
#pragma HLS LOOP_TRIPCOUNT max = 30
						for (unsigned cho = 0; cho < bCHout && cho + CHout_batch < CHout; cho++)
						{
#pragma HLS PIPELINE
							Out[(cho + CHout_batch) * R_out * C_out + (r2 + R_out_batch) * C_out + (c2 + C_out_batch)] = Out_1[cho][r2][c2];
						}
					}
				}
			}
		}
	}
}
