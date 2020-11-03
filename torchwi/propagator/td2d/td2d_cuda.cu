#include "td2d_cuda.h"

__device__ int d_idx_body(const int ix,const int iy,const int stride_y,const int ne)
{
	return (iy+ne)*stride_y + (ix+ne);
}

// inject source

__global__ void cuda_addsrc1(REAL *u3, const int isrc, REAL w, REAL *vel,const REAL dt2)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	float v=vel[isrc];
	if(id == 0)
		u3[isrc] += w * v*v * dt2;
}

__global__ void cuda_addsrcs(REAL *u3,const int nx,const int rcvy,const int stride_y,const int ne,REAL *residual,REAL *vel,const REAL dt2)
{
	int ix=blockIdx.x*blockDim.x+threadIdx.x;
	if(ix>=nx) return;
	int j,k;
	k=ix;
	j=d_idx_body(ix,rcvy,stride_y,ne);
	float v=vel[j];
	float vdt2 = v*v * dt2;
	u3[j] += residual[k] * vdt2;
}

// FDM

__global__ void cuda_fdm_o2(REAL *u1, REAL *u2, REAL *u3, const int stride_y,REAL *v,const REAL dtoh2)
{
#define NE1 1
#define C10 -2.f
#define C11 1.f
	int ix=blockIdx.x*blockDim.x+threadIdx.x;
	int iy=blockIdx.y*blockDim.y+threadIdx.y;

	__shared__ REAL s_data[BDIMY+2*NE1][BDIMX+2*NE1];

	int tx=threadIdx.x+NE1;
	int ty=threadIdx.y+NE1;
	int tilex,tiley; // tile width x, y
    tilex=BDIMX;
    tiley=BDIMY;

	REAL div,current;
	int i=(iy+NE1)*stride_y+ix+NE1;

    // update the data slice in shared memory
    if(threadIdx.y<NE1) // halo above/below
    {
        s_data[threadIdx.y][tx] = u2[i-NE1*stride_y];
        s_data[threadIdx.y+tiley+NE1][tx] = u2[i+tiley*stride_y];
    }
    if(threadIdx.x<NE1) // halo left/right
    {
        s_data[ty][threadIdx.x] = u2[i-NE1];
        s_data[ty][threadIdx.x+tilex+NE1] = u2[i+tilex];
    }

    current=u2[i];
    s_data[ty][tx]=current;
    __syncthreads();

    div = C10*2*current;
    div+= C11*(s_data[ty-1][tx]+s_data[ty+1][tx]+s_data[ty][tx-1]+s_data[ty][tx+1]);
    u3[i]=2.f*current-u1[i]+div*v[i]*v[i]*dtoh2;

}


__global__ void cuda_fdm_o4(REAL *u1, REAL *u2, REAL *u3, const int stride_y, REAL *v,const REAL dtoh2)
{
#define NE2 2
#define C20 -5.f/2.f
#define C21 4.f/3.f
#define C22 -1.f/12.f
	int ix=blockIdx.x*blockDim.x+threadIdx.x;
	int iy=blockIdx.y*blockDim.y+threadIdx.y;

	__shared__ REAL s_data[BDIMY+2*NE2][BDIMX+2*NE2];

	int tx=threadIdx.x+NE2;
	int ty=threadIdx.y+NE2;
	int tilex,tiley; // tile width x, y
    tilex=BDIMX;
    tiley=BDIMY;

	REAL div,current;
	int i=(iy+NE2)*stride_y+ix+NE2;

    // update the data slice in shared memory
    if(threadIdx.y<NE2) // halo above/below
    {
        s_data[threadIdx.y][tx] = u2[i-NE2*stride_y];
        s_data[threadIdx.y+tiley+NE2][tx] = u2[i+tiley*stride_y];
    }
    if(threadIdx.x<NE2) // halo left/right
    {
        s_data[ty][threadIdx.x]           = u2[i-NE2];
        s_data[ty][threadIdx.x+tilex+NE2] = u2[i+tilex];
    }

	current=u2[i];
    s_data[ty][tx]=current;
    __syncthreads();

    div = C20*2*current;
    div+= C21*(s_data[ty-1][tx]+s_data[ty+1][tx]+s_data[ty][tx-1]+s_data[ty][tx+1]);
    div+= C22*(s_data[ty-2][tx]+s_data[ty+2][tx]+s_data[ty][tx-2]+s_data[ty][tx+2]);
    u3[i]=2.f*current-u1[i]+div*v[i]*v[i]*dtoh2;

}

__global__ void cuda_fdm_o8(REAL *u1, REAL *u2, REAL *u3, const int stride_y, REAL *v,const REAL dtoh2)
{
#define NE4 4
#define C40 -205.f/72.f
#define C41 8.f/5.f
#define C42 -1.f/5.f
#define C43 8.f/315.f
#define C44 -1./560.f
	int ix=blockIdx.x*blockDim.x+threadIdx.x;
	int iy=blockIdx.y*blockDim.y+threadIdx.y;

	__shared__ REAL s_data[BDIMY+2*NE4][BDIMX+2*NE4];

	int tx=threadIdx.x+NE4;
	int ty=threadIdx.y+NE4;
	int tilex,tiley; // tile width x, y
    tilex=BDIMX;
    tiley=BDIMY;

	REAL div,current;
	int i=(iy+NE4)*stride_y+ix+NE4; // center
	
    // update the data slice in shared memory
    if(threadIdx.y<NE4) // halo above/below
    {
        s_data[threadIdx.y][tx] = u2[i-NE4*stride_y];
        s_data[threadIdx.y+tiley+NE4][tx] = u2[i+tiley*stride_y];
    }
    if(threadIdx.x<NE4) // halo left/right
    {
        s_data[ty][threadIdx.x]           = u2[i-NE4];
        s_data[ty][threadIdx.x+tilex+NE4] = u2[i+tilex];
    }

	current=u2[i];
    s_data[ty][tx]=current;
    __syncthreads();

    div = C40*2*current;
    div+= C41*(s_data[ty-1][tx]+s_data[ty+1][tx]+s_data[ty][tx-1]+s_data[ty][tx+1]);
    div+= C42*(s_data[ty-2][tx]+s_data[ty+2][tx]+s_data[ty][tx-2]+s_data[ty][tx+2]);
    div+= C43*(s_data[ty-3][tx]+s_data[ty+3][tx]+s_data[ty][tx-3]+s_data[ty][tx+3]);
    div+= C44*(s_data[ty-4][tx]+s_data[ty+4][tx]+s_data[ty][tx-4]+s_data[ty][tx+4]);
    u3[i]=2.f*current-u1[i]+div*v[i]*v[i]*dtoh2;
}

// for gradient calculation

__global__ void cuda_virt(float *d_virt,const float cons,float *d_vel,float *d_u1,float *d_u2,float *d_u3,const int dimxy)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i>dimxy) return;
    float v=d_vel[i];
	d_virt[i]=cons/(v*v*v)*(d_u3[i]-2.f*d_u2[i]+d_u1[i]);
}

__global__ void cuda_grad(float *d_virt,float *d_u3,float* d_grad,const int dimxy)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i>dimxy) return;
	d_grad[i] += -d_virt[i]*d_u3[i];
}


// Keys Boundary Condition

#define NEMAX 4	// NE=order/2

// theta1 = M_PI/6 = 0.5235987755982988
// theta2 = M_PI/12= 0.2617993877991494
// ct1 = cos(theta1) = 0.8660254037844387
// ct2 = cos(theta2) = 0.9659258262890683
// ct12m = ct1*ct2 = 0.8365163037378079
// ct12p = ct1+ct2 = 1.831951230073507
#define CT12M 0.8365163037378079
#define CT12P 1.831951230073507

__device__ REAL keysval(const REAL v0, const REAL u10, const REAL u11,
		const REAL u20, const REAL u21, const REAL u22,
		const REAL h2, const REAL dt2, const REAL hdt)
{
	REAL uxx,uxt;
	uxx=(u20-2.*u21+u22)/h2;
	uxt=((u20-u10)-(u21-u11))/hdt;
	return -v0*v0*dt2/CT12M*(uxx+CT12P/v0*uxt)+2.*u20-u10;
}

__global__ void bc_keys_zn(const REAL *v,const REAL *u1,const REAL *u2, REAL *u3, const int nxne, const int nzne, const int ne, const int dimx, const int dimz,
		const int stride_z,const REAL h2,const REAL dt2,const REAL hdt)
{
	int ix=blockIdx.x*blockDim.x+threadIdx.x;
	if(ix>=dimx) return;
	REAL u[2][NEMAX+2];
	int ig=(nzne-2)*stride_z+ix;
	int j,stride;
	for(j=0;j<ne+2;j++)
	{
		stride=j*stride_z;
		u[0][j]=u1[ig+stride];
		u[1][j]=u2[ig+stride];
	}
	int k,k0=nzne*stride_z+ix;
	int i,i1,i2;
	for(j=0;j<ne;j++)
	{
		k=k0+j*stride_z; // global idx
		i=j+2; // local u idx
		i1=i-1;
		i2=i-2;
		u3[k]=keysval(v[k],u[0][i],u[0][i1],u[1][i],u[1][i1],u[1][i2],h2,dt2,hdt);
	}
}

__global__ void bc_keys_xn(const REAL *v,const REAL *u1,const REAL *u2, REAL *u3, const int nxne, const int nzne, const int ne, const int dimx, const int dimz,
		const int stride_z,const REAL h2,const REAL dt2,const REAL hdt)
{
	int iz=blockIdx.x*blockDim.x+threadIdx.x;
	if(iz>=dimz) return;
	REAL u[2][NEMAX+2];
	int ig=iz*stride_z+nxne-2;
	int j,stride;
	for(j=0;j<ne+2;j++)
	{
		stride=j;
		u[0][j]=u1[ig+stride];
		u[1][j]=u2[ig+stride];
	}
	int k,k0=iz*stride_z+nxne;
	int i,i1,i2;
	for(j=0;j<ne;j++)
	{
		k=k0+j; // global idx
		i=j+2; // local u idx
		i1=i-1;
		i2=i-2;
		u3[k]=keysval(v[k],u[0][i],u[0][i1],u[1][i],u[1][i1],u[1][i2],h2,dt2,hdt);
	}
}

__global__ void bc_keys_x0(const REAL *v,const REAL *u1,const REAL *u2, REAL *u3, const int nxne, const int nzne, const int ne, const int dimx, const int dimz,
		const int stride_z,const REAL h2,const REAL dt2,const REAL hdt)
{
	int iz=blockIdx.x*blockDim.x+threadIdx.x;
	if(iz>=dimz) return;
	REAL u[2][NEMAX+2];
	int ig=iz*stride_z;
	int j,stride;
	for(j=0;j<ne+2;j++)
	{
		stride=j;
		u[0][j]=u1[ig+stride];
		u[1][j]=u2[ig+stride];
	}
	int k,k0=ig;
	int i,i1,i2;
	for(j=ne-1;j>=0;j--)
	{
		k=k0+j; // global idx
		i=j; // local u idx
		i1=i+1;
		i2=i+2;
		u3[k]=keysval(v[k],u[0][i],u[0][i1],u[1][i],u[1][i1],u[1][i2],h2,dt2,hdt);
	}
}

