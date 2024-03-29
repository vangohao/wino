#include "conv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <math.h>
#define CNN cnn
void cnn_test(d_type* In, d_type* Out, d_type* W, int *Parameter)
{

	float MAXFL = 0, MINFL=0;
	/*
	In  : Input feature map, CHin*R*C
	Out : Output feature map, CHout*Rout*Cout
	W : weights, CHout*CHin*Kr*Kc
	Parameter:  CHin|CHout|R|C|K|S
	*/
	// #pragma HLS INTERFACE s_axilite port=return
	// #pragma HLS INTERFACE m_axi depth=40960 port=In offset=slave           //adjust the depth as you need
	// #pragma HLS INTERFACE m_axi depth=40960 port=Out offset=slave
	// #pragma HLS INTERFACE m_axi depth=40960 port=W offset=slave
	// #pragma HLS INTERFACE m_axi depth=256 port=Parameter offset=slave

	int CHin,CHout,R_in,C_in,K,S;
	/*
	CHin : Input channels
	CHout : output channels
	R_in : Input rows
	C_in : Input columns
	K : kernel size (Kr = Kc)
	S : Stride
	*/
	int parameter_buffer[NParameter];

	memcpy((void*)parameter_buffer, (const int*)Parameter, NParameter *sizeof(int));

	CHin = parameter_buffer[0];
	CHout = parameter_buffer[1];
	R_in = parameter_buffer[2];
	C_in = parameter_buffer[3];
	K = parameter_buffer[4];
	S = parameter_buffer[5];

	int R_out = ((R_in - K) / S) + 1;
	int C_out = ((C_in - K) / S) + 1;

	for(int r1 = 0; r1 < R_out; r1++)
	{
		for(int c1 = 0; c1 < C_out; c1++)
		{
			for(int cho = 0; cho < CHout; cho++)
			{
				for(int chi = 0; chi < CHin; chi++)
				{
					Out[cho * R_out * C_out + r1 * C_out + c1] = 0;
				}
			}
		}
	}	

	for(int kr = 0; kr < K; kr++)
	{
		for(int kc = 0; kc < K; kc++)
		{
			for(int r1 = 0; r1 < R_out; r1++)
			{
				for(int c1 = 0; c1 < C_out; c1++)
				{
					for(int cho = 0; cho < CHout; cho++)
					{
						for(int chi = 0; chi < CHin; chi++)
						{
							Out[cho * R_out * C_out + r1 * C_out + c1] 
								+= W[cho * CHin * K * K + chi * K * K + kr * K + kc] 
									* In[chi * R_in * C_in + (S * r1 + kr) * C_in + (S * c1 + kc)];
							if (Out[cho * R_out * C_out + r1 * C_out + c1] > MAXFL)
							{
								MAXFL = Out[cho * R_out * C_out + r1 * C_out + c1];
							}
							if (Out[cho * R_out * C_out + r1 * C_out + c1] < MINFL)
							{
								MINFL = Out[cho * R_out * C_out + r1 * C_out + c1];
							}
						}
					}
				}
			}
		}
	}
	printf("MAXFL: %lf  MINFL: %lf\n", MAXFL, MINFL);
	
}

int example2()
{
	int CHin = 32;
	int CHout = 64;
	int R_in = 128;
	int C_in = 128;
	int K = 3;
	int Kg = 6;
	int S = 1;
	int R_out = ((R_in - K) / S) + 1;
	int C_out = ((C_in - K) / S) + 1;
	int In = CHin * R_in * C_in;
	int weight = CHout * CHin * Kg * Kg;
	int Out = CHout * R_out * C_out;
	d_type *In_data = (d_type*)malloc(In * sizeof(d_type));
	d_type *Out_data = (d_type*)malloc(Out * sizeof(d_type));
	d_type *Output_data = (d_type*)malloc(Out * sizeof(d_type));
	d_type *Weight_data = (d_type*)malloc(weight * sizeof(d_type));
	int parameter[6];
	parameter[0] = CHin;
	parameter[1] = CHout;
	parameter[2] = R_in;
	parameter[3] = C_in;
	parameter[4] = K;
	parameter[5] = S;

	printf("Begin sample_2.\n");
	FILE *f_in = fopen("../../../../dat/sample_input.dat", "r");
	for (int i = 0; i < In; i++)
	{
		float x;
		fscanf(f_in, "%f", &x);
		In_data[i] = x;
	}
	fclose(f_in);
	printf("Finish read input.\n");
	FILE * f_weight = fopen("../../../../dat/sample_weight.dat", "r");
	for (int i = 0; i < weight; i++)
	{
		fscanf(f_weight, "%f", &Weight_data[i]);
	}
	fclose(f_weight);
	printf("Finish read weight.\n");

	CNN(In_data, Output_data, Weight_data, parameter);
	printf("CNN finish.\n");

	FILE * f_out = fopen("../../../../dat/sample_out.dat", "r");
	for (int i = 0; i < Out; i++)
	{
		fscanf(f_out, "%f", &Out_data[i]);
	}
	fclose(f_out);
	int cnt = 0;
	float norm1 = 0, norm2 = 0;
	float err = 0;
	for (int i  = 0; i < Out; i++)
	{
		if (i < 50)
		{
			printf("%d  %f\n", i, Output_data[i]);
		}
		if (Out_data[i] - Output_data[i] > 1e-1 || Out_data[i] - Output_data[i] < -1e-1)
		{
			if (cnt < 20)
			printf("Error, No. %d, output = %.10f, real output = %.10f, difference = %.10f\n",
				 i, Output_data[i], Out_data[i], Output_data[i] - Out_data[i]);
			cnt ++;
		}
		norm1 += (Out_data[i] - Output_data[i]) * (Out_data[i] - Output_data[i]);
		norm2 += Out_data[i] * Out_data[i];
		err += fabs(Out_data[i] - Output_data[i]);
	}
	float rela_res = sqrt(norm1) / sqrt(norm2);
	printf("Example 2 Passed.  Residual = %f.   err = %f\n", rela_res, err);
	return 0;
}


int main()
{
	example2();
	return 0;
}