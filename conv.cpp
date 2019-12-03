#include "conv.h"

void cnn(d_type* In, d_type* Out, d_type* W, int *Parameter)
{

	/*
	In  : Input feature map, CHin*R*C
	Out : Output feature map, CHout*Rout*Cout
	W : converted weights, CHout*CHin*6*6
	Parameter:  CHout|CHin|R|C|K|S
	*/
	#pragma HLS INTERFACE s_axilite port=return
	#pragma HLS INTERFACE m_axi depth=XXX port=In offset=slave           //adjust the depth as you need
	#pragma HLS INTERFACE m_axi depth=XXX port=Out offset=slave
	#pragma HLS INTERFACE m_axi depth=XXX port=W offset=slave
	#pragma HLS INTERFACE m_axi depth=256 port=Parameter offset=slave

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

	// Your implementation

	
}

