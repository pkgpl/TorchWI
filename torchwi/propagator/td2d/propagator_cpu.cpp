#include <stdio.h>
#include <string.h>
#include "td2d_cpu.h"

void c_fdm_forward(
        float *forward,
        float *virt,
        float *u1, float *u2, float *u3,
        float *vel, float *w,
        const int order, const int dimx, const int dimy, const int nt,
        const float h, const float dt,
        const float sx, const float sy, const float ry)
{
	// select fdm order
    void (*fdm)(const int nx, const int ny,
        const float* __restrict__ u1, const float* __restrict__ u2, float* __restrict__ u3,
        const float* __restrict__ vel, const float dtoh2, const int ne, const int stride_y);
	if(order==2){
		fdm = &fdm_O2;
	}else if(order==4){
		fdm = &fdm_O4;
	}else if(order==8){
		fdm = &fdm_O8;
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

    float dt2=dt*dt;
    float h2=h*h;
    float dtoh2=dt2/h2;
    float hdt=h*dt;
    float cons = -2.0/dt2;
    float *tmp;

    int isx=sx/h;
    int isy=sy/h;
    int isrc=(isy+ne)*stride_y+(isx+ne);
    int iry = ry/h;
    int ir0 =(iry+ne)*stride_y+ne;

    size_t dimsize = dimxy*sizeof(float);

    // initialization
    memset(u1,0,dimsize);
    memset(u2,0,dimsize);
    memset(u3,0,dimsize);

    for(int it=0;it<nt;it++)
    {
        // fdm
        fdm(nx,ny, u1,u2,u3,vel, dtoh2, ne,stride_y);
        // add source
        u3[isrc]+=w[it]*vel[isrc]*vel[isrc]*dt2;
        // boundary
        bc_keys_v(u1,u2,u3,vel, nx,ny,ne,nxne,nyne,dimx,dimy,stride_y, h2,dt2,hdt);
        // foward modeled data
        memcpy(&forward[it*nx],&u3[ir0],sizeof(float)*nx);
        // virtual source
        for(int i=0;i<dimxy;i++)
        {
            float v3=vel[i]*vel[i]*vel[i];
            virt[it*dimxy+i]=cons/v3*(u1[i]-2.f*u2[i]+u3[i]);
        }
        // time march
        tmp  = u1;
        u1 = u2;
        u2 = u3;
        u3 = tmp ;
    }// it
}//forward


void c_fdm_backward(
        float *grad,
        float *virt,
        float *u1, float *u2, float *u3,
        float *vel, float *resid,
        const int order, const int dimx, const int dimy, const int nt,
        const float h, const float dt,
        const float ry)
{
	// select fdm order
    void (*fdm)(const int nx, const int ny,
        const float* __restrict__ u1, const float* __restrict__ u2, float* __restrict__ u3,
        const float* __restrict__ vel, const float dtoh2, const int ne, const int stride_y);
	if(order==2){
		fdm = &fdm_O2;
	}else if(order==4){
		fdm = &fdm_O4;
	}else if(order==8){
		fdm = &fdm_O8;
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

    float dt2=dt*dt;
    float h2=h*h;
    float dtoh2=dt2/h2;
    float hdt=h*dt;
    float *tmp;

    int iry = ry/h;
    int ir0 =(iry+ne)*stride_y+ne;

    size_t dimsize = dimxy*sizeof(float);

    // initialization
    memset(u1,0,dimsize);
    memset(u2,0,dimsize);
    memset(u3,0,dimsize);

    for(int it=nt-1;it>=0;it--)
    {
        // fdm
        fdm(nx,ny, u1,u2,u3,vel, dtoh2, ne,stride_y);
        // add source
        for(int ix=0;ix<nx;ix++)
        {
            int j=ir0+ix;
            u3[j] += resid[it*nx+ix]*vel[j]*vel[j]*dt2;
        }
        // boundary
        bc_keys_v(u1,u2,u3,vel, nx,ny,ne,nxne,nyne,dimx,dimy,stride_y, h2,dt2,hdt);
        // gradient
        for(int i=0;i<dimxy;i++)
            grad[i] += -virt[it*dimxy+i]*u3[i];
        // time march
        tmp  = u1;
        u1 = u2;
        u2 = u3;
        u3 = tmp ;

    }// it
}//backward

