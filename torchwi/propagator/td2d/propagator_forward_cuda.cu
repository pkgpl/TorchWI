#include <stdio.h>
#include "td2d_cuda.h"

void cuda_fdm_forward_only(
        float *forward,
        float *d_u1, float *d_u2, float *d_u3,
        float *d_vel, float *w,
        const int order, const int dimx, const int dimy, const int nt,
        const float h, const float dt,
        const float sx, const float sy, const float ry)
{
	// select fdm order
	void (*cuda_fdm)(float *u1, float *u2, float *u3, const int stride_y, float *vel, const float dtoh2);
	if(order==2){
		cuda_fdm = &cuda_fdm_o2;
	}else if(order==4){
		cuda_fdm = &cuda_fdm_o4;
	}else if(order==8){
		cuda_fdm = &cuda_fdm_o8;
	}else{
		puts("Wrong order!\n");
	}

    int ne = order/2;
    int stride_y = dimx;

    int dimxy=dimx*dimy;
    int nx=dimx-order;
    int ny=dimy-order;
    int nxne = dimx-ne;
    int nyne = dimy-ne;

	dim3 dimBlock(BDIMX,BDIMY);
	dim3 dimGrid((nx+BDIMX-1)/BDIMX,(ny+BDIMY-1)/BDIMY);
    dim3 dimGridX((dimy+BDIMY-1)/BDIMY);
    dim3 dimGridY((dimx+BDIMX-1)/BDIMX);
	dim3 dimBlock1(BDIM1);
	dim3 dimGrid1((dimxy+BDIM1-1)/BDIM1);

    float dt2=dt*dt;
    float h2=h*h;
    float dtoh2=dt2/h2;
    float hdt=h*dt;
    float *tmp;

    int isx=sx/h;
    int isy=sy/h;
    int isrc=(isy+ne)*stride_y+(isx+ne);
    int iry = ry/h;
    int ir0 =(iry+ne)*stride_y+ne;

    size_t dimsize = dimxy*sizeof(float);

    // initialization
    cudaMemset(d_u1,0,dimsize);
    cudaMemset(d_u2,0,dimsize);
    cudaMemset(d_u3,0,dimsize);

    for(int it=0;it<nt;it++)
    {
        // fdm
        cuda_fdm<<<dimGrid,dimBlock>>>(d_u1,d_u2,d_u3,stride_y,d_vel,dtoh2);
        // add source
        cuda_addsrc1<<<1,1>>>(d_u3,isrc,w[it],d_vel,dt2);
        // boundary
        bc_keys_zn<<<dimGridY,dimBlock>>>(d_vel,d_u1,d_u2,d_u3,nxne,nyne,ne,dimx,dimy,stride_y,h2,dt2,hdt);
        bc_keys_x0<<<dimGridX,dimBlock>>>(d_vel,d_u1,d_u2,d_u3,nxne,nyne,ne,dimx,dimy,stride_y,h2,dt2,hdt);
        bc_keys_xn<<<dimGridX,dimBlock>>>(d_vel,d_u1,d_u2,d_u3,nxne,nyne,ne,dimx,dimy,stride_y,h2,dt2,hdt);
        // foward modeled data
        cudaMemcpy(&forward[it*nx],&d_u3[ir0],sizeof(float)*nx,cudaMemcpyDeviceToDevice);
        // time march
        tmp  = d_u1;
        d_u1 = d_u2;
        d_u2 = d_u3;
        d_u3 = tmp ;
    }// it
}//forward


