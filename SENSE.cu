/**
 * Sensitivity Encoding (SENSE) Reconstruction
 * SENSE is a Parallel MRI reconstruction method. The inputs of SENSE reconstruction are aliased data from MRI scanner and
 * receiver coil sensitivity encoding matrices.   
 * The output of SENSE reconstruction is reconstructed MR image for clinical usage.
 * 
 * Mathematically, SENSE can be represented as: U = C x M  (1), where U is the aliased image obtained from scanner, M is MR image to be 
 * reconstructed and C is receiver coil sensitivity encoding matrix.
 * In order to find M, the above equation can be written as:  M = inv(C) x U   (2).
 * To solve equation 2, there is requirement to invert large number of small encoding matrices (in this simpler case the order of small matrices will be 2x2). 
 * To perform this task iteratively significant computation time is involved. Parallel implementation of SENSE using GPU is presented in this work where 
 * number of CUDA threads are launched as per required matrix inversions to perform the tasks in parallel hence reducing the computation
 * time (one of the main limitation in MRI). The GPU implementation using NVIDIA Titan XP GPU is more than 10x faster compared to CPU implementation
 * (core i7 with 8GB RAM)
 * In this work, the size of U, C and M are 128x256x2, 256x256x2 and 256x256 
 *
 * Note: The data in all the matrices is complex, therefore real part and imaginary part are handled separately in the code given below
 * 
 */

//declaration of header files
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define N 128
using namespace std;

//start of CUDA kernel "sense". 
 __global__ void sense( float * d_i1imag, float * d_i2imag, float * d_i1real, float * d_i2real, float * d_c1imag, float * d_c2imag, float * d_c1real, float * d_c2real, float * d_rmreal, float * d_rmimag) 
{
	float d_detreal;				// determinant real part
	float d_detimag;				// determinant imaginary part
	float d_divtemp;				// complex determinant
	float d_sreal[2][2];				//encoding matrix of size 2x2 real part
	float d_simag[2][2];				//encoding matrix of size 2x2 imaginary part
	float d_rpreal[2];				//reconstructed pixels real part 
	float d_rpimag[2];				//reconstructed pixels imaginary part
	float d_sinreal[2][2];				// to find inverse of 2x2 matrix (real part)
	float d_sinimag[2][2];				// to find inverse of 2x2 matrix (imaginary part)

	//CUDA thread index calculation 
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y +threadIdx.y;
	int index=col+row*N;

	//copying data from input receiver coil sensitivty matrix (real part)
	d_sreal[0][0] = d_c1real[index];
	d_sreal[0][1] = d_c1real[index+32768];
	d_sreal[1][0] = d_c2real[index];
	d_sreal[1][1] = d_c2real[index+32768];

	//copying data from input receiver coil sensitivty matrix (imaginary part)
	d_simag[0][0] = d_c1imag[index];
	d_simag[0][1] = d_c1imag[index+32768];
	d_simag[1][0] = d_c2imag[index];
	d_simag[1][1] = d_c2imag[index+32768];

	//calculation of complex determinant
	d_detreal = ((d_sreal[0][0]*d_sreal[1][1])-(d_simag[0][0] *d_simag[1][1]))-((d_sreal[0][1] * d_sreal[1][0]) - (d_simag[0][1] *d_simag[1][0]));
	d_detimag = ((d_simag[0][0]*d_sreal[1][1])+(d_sreal[0][0] *d_simag[1][1]))-((d_simag[0][1] * d_sreal[1][0]) + (d_sreal[0][1] *d_simag[1][0]));
	d_divtemp = (d_detreal*d_detreal) + (d_detimag * d_detimag);

	//dividing the adjoint of matrix with the determinant (real part)
	d_sinreal[0][0] = ((d_sreal[1][1] * d_detreal)   - (d_simag[1][1]*(-d_detimag)))  /d_divtemp;
	d_sinreal[0][1] = -(((d_sreal[0][1] * d_detreal) - (d_simag[0][1]*(-d_detimag)))) /d_divtemp;
	d_sinreal[1][0] = -(((d_sreal[1][0] * d_detreal) - (d_simag[1][0]*(-d_detimag)))) /d_divtemp;
	d_sinreal[1][1] = ((d_sreal[0][0] * d_detreal)   - (d_simag[0][0]*(-d_detimag)))  /d_divtemp;
	
	//dividing the adjoint of matrix with the determinant (imaginary part)
	d_sinimag[0][0] = ((d_simag[1][1] * d_detreal) + (d_sreal[1][1]*(-d_detimag)))/d_divtemp;
	d_sinimag[0][1] = -(((d_simag[0][1] * d_detreal) + (d_sreal[0][1]*(-d_detimag))))/d_divtemp;
	d_sinimag[1][0] = -(((d_simag[1][0] * d_detreal) + (d_sreal[1][0]*(-d_detimag))))/d_divtemp;
	d_sinimag[1][1] = ((d_simag[0][0] * d_detreal) + (d_sreal[0][0]*(-d_detimag)))/d_divtemp;
	
	//Multiplying the inverse of 2x2 encoding matrix (calculated above) with 
	//2x1 matrix of input aliased matrix (from scanner) real part 	
	d_rpreal[0] = ((d_i1real[index] * d_sinreal[0][0]) - (d_i1imag[index] * d_sinimag[0][0])) + ((d_i2real[index] * d_sinreal[0][1]) - (d_i2imag[index] * d_sinimag[0][1]));
	d_rpreal[1] = ((d_i1real[index] * d_sinreal[1][0]) - (d_i1imag[index] * d_sinimag[1][0])) + ((d_i2real[index] * d_sinreal[1][1]) - (d_i2imag[index] * d_sinimag[1][1]));
	
	//Multiplying the inverse of 2x2 encoding matrix (calculated above) with 
	//2x1 matrix of input aliased matrix (from scanner) imaginary part
	d_rpimag[0] = ((d_i1real[index] * d_sinimag[0][0]) + (d_i1imag[index] * d_sinreal[0][0])) + ((d_i2real[index] * d_sinimag[0][1]) + (d_i2imag[index] * d_sinreal[0][1]));
	d_rpimag[1] = ((d_i1real[index] * d_sinimag[1][0]) + (d_i1imag[index] * d_sinreal[1][0])) + ((d_i2real[index] * d_sinimag[1][1]) + (d_i2imag[index] * d_sinreal[1][1]));

	//Copying the resulting real and imaginary numbers to the final reconstructed image  
	d_rmreal[index] = d_rpreal[0]; 
	d_rmreal[index+32768] = d_rpreal[1];
	d_rmimag[index] = d_rpimag[0];
	d_rmimag[index+32768] = d_rpimag[1];
}


int main ()
{
	float temp;				    //temparary variable
	int i,j,k;				    //index variables 
	int maxx, maxim;		    //maxx is size of unaliased image i.e.256x256, maxim is size of aliased image i.e. 128x256 
	float * h_coil1imag;		//pointer for receiver coil 1 matrix imaginary part
	float * h_coil1real;		//pointer for receiver coil 1 matrix real part
	float * h_coil2imag;		//pointer for receiver coil 2 matrix imaginary part
	float * h_coil2real;		//pointer for receiver coil 2 matrix real part

	float * h_im1real;		//pointer for aliased image 1 matrix real part
	float * h_im1imag;		//pointer for aliased image 1 matrix imaginary part
	float * h_im2real;		//pointer for aliased image 2 matrix real part
	float * h_im2imag;		//pointer for aliased image 3 matrix imaginary part

	float * h_rimreal;		//pointer for reconstructed image matrix real part
	float * h_rimimag;		//pointer for reconstructed image matrix imaginary part
	
	maxim = 128*256;		//maxim is size of aliased image i.e. 128x256
	maxx  = 256*256;		//maxx is size of unaliased image i.e.256x256
	const int ARRAY_BYTES1 = maxx  * sizeof(float);		//it will be used for reconstructed image and receiver coils matrices
	const int ARRAY_BYTES2 = maxim * sizeof(float);		//it will be used for aliased image
	
	dim3 grid(1,128);		//dimensions of blocks in a grid
	dim3 block(128,2);		//dimensions of threads in a block

	//memory allocation for reconstructed image real and imaginary parts respectively
	h_rimreal = (float*) calloc (maxx,sizeof(float));
	if (h_rimreal==NULL) exit (1);			//exception handling
	h_rimimag = (float*) calloc (maxx,sizeof(float));
	if (h_rimimag==NULL) exit (1);			//exception handling
	
	//memory allocation for receiver coils matrices real and imaginary parts respectively
	h_coil1imag = (float*) calloc (maxx,sizeof(float));
	if (h_coil1imag==NULL) exit (1);

	h_coil1real = (float*) calloc (maxx,sizeof(float));
	if (h_coil1real==NULL) exit (1);

	h_coil2imag = (float*) calloc (maxx,sizeof(float));
	if (h_coil2imag==NULL) exit (1);

	h_coil2real = (float*) calloc (maxx,sizeof(float));
	if (h_coil2real==NULL) exit (1);
	
	//memory allocation for aliased images real and imaginary parts respectively

	h_im1real = (float*) calloc (maxim,sizeof(float));
	if (h_im1real==NULL) exit (1); 
	
	h_im1imag = (float*) calloc (maxim,sizeof(float));
	if (h_im1imag==NULL) exit (1);

	h_im2real = (float*) calloc (maxim,sizeof(float));
	if (h_im2real==NULL) exit (1);

	h_im2imag = (float*) calloc (maxim,sizeof(float));
	if (h_im2imag==NULL) exit (1);

	//Copying data from file for aliased image1 real  data 
	FILE *fptr5;
	fptr5 = fopen ("im1real.txt", "r");
	k=0;
	for(i=0;i<=127;i++)
	{
		for(j=0;j<=255;j++)
		{
			fscanf(fptr5, "%f,  ",&temp  );
			h_im1real[k] = temp; 
			k=k+1;
		}	
	}
	fclose (fptr5);

	//Copying data from file for aliased image1 imaginary  data 
	FILE *fptr6;
	fptr6 = fopen ("im1imag.txt", "r");
	k=0;
	for(i=0;i<=127;i++)
	{
		for(j=0;j<=255;j++)
		{
			fscanf(fptr6, "%f,  ",&temp  );
			h_im1imag[k] = temp; 
			k=k+1;
		}	
	}
	fclose (fptr6);

	//Copying data from file for  aliased image2 real data 
	FILE *fptr7;
	fptr7 = fopen ("im2real.txt", "r");
	k=0;
	for(i=0;i<=127;i++)
	{ 
		for(j=0;j<=255;j++)
		{
			fscanf(fptr7, "%f,  ",&temp  );
			h_im2real[k] = temp; 
			k=k+1;
		}	
	}
	fclose (fptr7);	

	//Copying data from file for aliased image2 imaginary  data 
	FILE *fptr8;
	fptr8 = fopen ("im2imag.txt", "r");
	k=0;
	for(i=0;i<=127;i++)
	{
		for(j=0;j<=255;j++)
		{
			fscanf(fptr8, "%f,  ",&temp  );
			h_im2imag[k] = temp; 
			k=k+1;
		}	
	}
	fclose (fptr8);
	
	//Copying data from file for rceiver coil 1 imaginary  data 
	FILE *fptr1;
	fptr1 = fopen ("coil1imag.txt", "r");
	k=0;
	for(i=0;i<=255;i++)
	{
		for(j=0;j<=255;j++)
		{
			fscanf(fptr1, "%f,  ",&temp  );
			h_coil1imag[k] = temp; 
			k=k+1;
		}	
	}
	fclose (fptr1);
		
	//Copying data from file for rceiver coil 1 real data 
	FILE *fptr2;
	fptr2 = fopen ("coil1real.txt", "r");
	k=0;
	for(i=0;i<=255;i++)
	{
		for(j=0;j<=255;j++)
		{
			fscanf(fptr2, "%f,  ",&temp  );
			h_coil1real[k] = temp; 
			k=k+1;
		}	
	}
	fclose (fptr2);

	//Copying data from file for rceiver coil 2 imaginary  data 
	FILE *fptr3;
	fptr3 = fopen ("coil2imag.txt", "r");
	k=0;
	for(i=0;i<=255;i++)
	{
		for(j=0;j<=255;j++)
		{
			fscanf(fptr3, "%f,  ",&temp  );
			h_coil2imag[k] = temp; 
			k=k+1;
		}	
	}
	fclose (fptr3);

	//Copying data from file for rceiver coil 2 real data 
	FILE *fptr4;
	fptr4 = fopen ("coil2real.txt", "r");
	k=0;
	for(i=0;i<=255;i++)
	{
		for(j=0;j<=255;j++)
		{
			fscanf(fptr4, "%f,  ",&temp  );
			h_coil2real[k] = temp; 
			k=k+1;
		}	
	}
	fclose (fptr4);
	
	// declaration of GPU memory pointers
	float * d_c1imag;  //device pointer for coil 1 matrix imaginary part
	float * d_c1real;  //device pointer for coil 1 matrix real part
	float * d_c2imag;  //device pointer for coil 2 matrix imaginary part
	float * d_c2real;  //device pointer for coil 2 matrix real part	
	float * d_i1real;  //device pointer for aliased image 1 matrix real part
	float * d_i1imag;  //device pointer for aliased image 1 matrix imaginary part
	float * d_i2real;  //device pointer for aliased image 2 matrix real part
	float * d_i2imag;  //device pointer for aliased image 2 matrix imaginary part
	float * d_rmreal;  //device pointer for image matrix to be reconstructed real part
	float * d_rmimag;  //device pointer for image matrix to be reconstructed imaginary part


	// Allocation of memory in device memory using CUDA Malloc function
	cudaMalloc( (void**) &d_rmreal, ARRAY_BYTES1); // for image to be reconstructed real part 
	cudaMalloc( (void**) &d_rmimag, ARRAY_BYTES1); // for image to be reconstructed imaginary part
	cudaMalloc( (void**) &d_c1imag, ARRAY_BYTES1); // for coil 1 imaginary part
	cudaMalloc( (void**) &d_c1real, ARRAY_BYTES1); // for coil 1 real part
	cudaMalloc( (void**) &d_c2imag, ARRAY_BYTES1); // for coil 2 imaginary part
	cudaMalloc( (void**) &d_c2real, ARRAY_BYTES1); // for coil 2 real part
	cudaMalloc( (void**) &d_i1real, ARRAY_BYTES2); // for aliased image 1 real part
	cudaMalloc( (void**) &d_i1imag, ARRAY_BYTES2); // for aliased image 1 imaginary part
	cudaMalloc( (void**) &d_i2real, ARRAY_BYTES2); // for aliased image 1 real part
	cudaMalloc( (void**) &d_i2imag, ARRAY_BYTES2); // for aliased image 1 imaginary part

	//Copying data from host memory to device memory
	cudaMemcpy(d_i1imag, h_im1imag, ARRAY_BYTES2, cudaMemcpyHostToDevice); //for image 1 imaginary part
	cudaMemcpy(d_i2imag, h_im2imag, ARRAY_BYTES2, cudaMemcpyHostToDevice); //for image 2 imaginary part
	cudaMemcpy(d_i1real, h_im1real, ARRAY_BYTES2, cudaMemcpyHostToDevice); //for image 1 real part
	cudaMemcpy(d_i2real, h_im2real, ARRAY_BYTES2, cudaMemcpyHostToDevice); //for image 2 real part
	cudaMemcpy(d_c1imag, h_coil1imag, ARRAY_BYTES1, cudaMemcpyHostToDevice); //for coil 1 imaginary part
	cudaMemcpy(d_c2imag, h_coil2imag, ARRAY_BYTES1, cudaMemcpyHostToDevice); //for coil 2 imaginary part
	cudaMemcpy(d_c1real, h_coil1real, ARRAY_BYTES1, cudaMemcpyHostToDevice); //for coil 1 real part
	cudaMemcpy(d_c2real, h_coil2real, ARRAY_BYTES1, cudaMemcpyHostToDevice); //for coil 2 real part
	cudaMemcpy(d_rmreal, h_rimreal, ARRAY_BYTES1, cudaMemcpyHostToDevice); //for image to be reconstructed real part
	cudaMemcpy(d_rmimag, h_rimimag, ARRAY_BYTES1, cudaMemcpyHostToDevice); //for image to be reconstructed imaginary part
	
	//initiation of CUDA timer's event to estimate the computation time of kernel
	float time; 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	//Launching the Kernel
	sense<<<grid,block>>>( d_i1imag, d_i2imag, d_i1real, d_i2real, d_c1imag, d_c2imag, d_c1real, d_c2real, d_rmreal, d_rmimag);

	cudaDeviceSynchronize(); //synchronization of threads
	
	//CUDA timer's events to estimate the computation time
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop) ;
	cudaEventElapsedTime(&time, start, stop) ;
	printf("Time for SENSE reconstruction:  %3.1f ms \n", time);
	
	//copying the results from device memory to host memory
	cudaMemcpy(h_rimreal, d_rmreal, ARRAY_BYTES1, cudaMemcpyDeviceToHost); //image to be reconstructed real part
	cudaMemcpy(h_rimimag, d_rmimag, ARRAY_BYTES1, cudaMemcpyDeviceToHost); //image to be reconstructed imaginary part

	//storing the results in output file (real part)
	FILE *fptr45;
	fptr45 = fopen ("reconstructionreal.txt", "w");
	k=0;
	for(i=0;i<=255;i++)
	{
		for(j=0;j<=255;j++)
		{
			fprintf(fptr45, " %f \t",h_rimreal[k]);			
			k=k+1;
		}
		fprintf(fptr45, " \n");
	}
	fclose (fptr45);
	//storing the results in output file (imaginary part)
	FILE *fptr445;
	fptr445 = fopen ("reconstructionimag.txt", "w");
	k=0;
	for(i=0;i<=255;i++)
	{
		for(j=0;j<=255;j++)
		{
			fprintf(fptr445, " %f \t",h_rimimag[k]);			
			k=k+1;
		}	
		fprintf(fptr445, " \n");
	}
	fclose (fptr445);
	
	printf("\n SENSE reconstruction is finished successfully \n");
	// free the host memory	
	free (h_coil1imag);
	free (h_coil1real);
	free (h_coil2imag);
	free (h_coil2real);
	free (h_im1real);
	free (h_im1imag);
	free (h_im2real);
	free (h_im2imag);
	free (h_rimreal);
	free (h_rimimag);

	// free the device memory
	cudaFree (d_c1imag);
	cudaFree (d_c1real);
	cudaFree (d_c2imag);
	cudaFree (d_c2real);
	cudaFree (d_i1real);
	cudaFree (d_i1imag);
	cudaFree (d_i2real);
	cudaFree (d_i2imag);
	cudaFree (d_rmreal);
	cudaFree (d_rmimag);

	getchar();
	return 0;
}
