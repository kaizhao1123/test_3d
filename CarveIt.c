/*=================================================================
*
* CarveIt.C	Does a simple space carve
*
* The calling syntax is:
*
*		./CarveIt inputfilepath outputfilepath
*
*=================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void carve(
	unsigned int sX, unsigned int sY, unsigned int sZ,
	unsigned char *V,
	double *P,
	unsigned int smX, unsigned int smY,
	int *mask,
	double	width, double height, double depth
	)
{
	double EPS = 1e-12;
	double dx = width / ((double)(sX));
	double dy = height / ((double)(sY));
	double dz = depth / ((double)(sZ));
	double x0 = -width / 2 + dx / 2.0;
	double y0 = -height / 2 + dy / 2.0;
	double z0 = -depth / 2 + dz / 2.0;

	int iqx, iqy;

	unsigned int iz = 0, ix = 0, iy = 0;
	for (iz = 0; iz<sZ; iz++){
		for (ix = 0; ix<sX; ix++){
			for (iy = 0; iy<sY; iy++){
				double x = x0 + (double)(ix + 1)*dx;
				double y = y0 + (double)(iy + 1)*dy;
				double z = z0 + (double)(iz + 1)*dz;

				double qx = P[0 + 3 * 0] * x + P[0 + 3 * 1] * y + P[0 + 3 * 2] * z + P[0 + 3 * 3];
				double qy = P[1 + 3 * 0] * x + P[1 + 3 * 1] * y + P[1 + 3 * 2] * z + P[1 + 3 * 3];
				double qz = P[2 + 3 * 0] * x + P[2 + 3 * 1] * y + P[2 + 3 * 2] * z + P[2 + 3 * 3];

				if (fabs(qz)<EPS){
					continue;
				}
				iqx = (int)floor(qx / qz + 0.5);
				iqy = (int)floor(qy / qz + 0.5);

				if (iqx >= 0 && iqx<smX && iqy >= 0 && iqy<smY){
					if (mask[iqy + smY*iqx] == 0){
						V[iy + sY*(ix + sX*iz)] = 0;
					}
				}
				else {
					V[iy + sY*(ix + sX*iz)] = 0;
				}
			}
		}
	}

	return;
}

//Input file layout binary without spaces:
//sX sY sZ Voxels P smX smY mask width height depth
void inputData(char* i_filename,
	unsigned int* o_sX, unsigned int* o_sY, unsigned int* o_sZ,
	unsigned char** o_voxels,
	double** o_P,
	unsigned int* o_smX, unsigned int* o_smY,
	int** o_mask,
	double* o_width, double* o_height, double* o_depth)
{
	FILE *fp;
	fp = fopen(i_filename, "rb");

	fread(o_sX, sizeof(unsigned int), 1, fp);
	fread(o_sY, sizeof(unsigned int), 1, fp);
	fread(o_sZ, sizeof(unsigned int), 1, fp);

	*o_voxels = (unsigned char*)malloc(sizeof(unsigned char)* (*o_sX) * (*o_sY) * (*o_sZ));
	fread(*o_voxels, sizeof(unsigned char), (*o_sX) * (*o_sY) * (*o_sZ), fp);

	*o_P = (double*)malloc(sizeof(double)* 3 * 4);
	fread(*o_P, sizeof(double), 3 * 4, fp);

	fread(o_smX, sizeof(unsigned int), 1, fp);
	fread(o_smY, sizeof(unsigned int), 1, fp);

	*o_mask = (int*)malloc(sizeof(int)* (*o_smX) * (*o_smY));
	fread(*o_mask, sizeof(int), (*o_smX) * (*o_smY), fp);

	fread(o_width, sizeof(double), 1, fp);
	fread(o_height, sizeof(double), 1, fp);
	fread(o_depth, sizeof(double), 1, fp);

	fclose(fp);
}

//Output file layout binary without spaces:
//Voxels
void outputData(char* i_filename, unsigned int n_voxels, unsigned char *voxels)
{
	FILE *fp;
	fp = fopen(i_filename, "wb");

	fwrite(voxels, sizeof(unsigned char), n_voxels, fp);

	fclose(fp);
}

void main(int argc, char *argv[])
{
	char *inputFileName, *outputFileName;

	unsigned int sX, sY, sZ;
	unsigned char *voxels;
	double *P;
	unsigned int smX, smY;
	int *mask;
	double width, height, depth;

	inputFileName = argv[1];
	outputFileName = argv[2];

	//Read input from input exchange file
	inputData(inputFileName, &sX, &sY, &sZ, &voxels, &P,
		&smX, &smY, &mask, &width, &height, &depth);

	carve(sX, sY, sZ, voxels, P, smX, smY, mask, width, height, depth);

	//Write voxel grid to output exchange file
	outputData(outputFileName, sX * sY * sZ, voxels);
}