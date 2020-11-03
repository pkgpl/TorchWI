#define REAL float

#define BDIMX 16
#define BDIMY 16
#define BDIM1 128
#define BDIMBC 128

__global__ void cuda_addsrc1(REAL *u3, const int isrc, REAL w, REAL *vel,const REAL dt2);
__global__ void cuda_addsrcs(REAL *u3,const int nx,const int rcvy,const int stride_y,const int ne,REAL *residual,REAL *vel,const REAL dt2);

__global__ void cuda_fdm_o2(REAL *u1, REAL *u2, REAL *u3, const int stride_y, REAL *v,const REAL dtoh2);
__global__ void cuda_fdm_o4(REAL *u1, REAL *u2, REAL *u3, const int stride_y, REAL *v,const REAL dtoh2);
__global__ void cuda_fdm_o8(REAL *u1, REAL *u2, REAL *u3, const int stride_y, REAL *v,const REAL dtoh2);

__global__ void cuda_virt(float *d_virt,const float cons,float *d_vel,float *d_u1,float *d_u2,float *d_u3,const int dimxy);
__global__ void cuda_grad(float *d_virt,float *d_u3,float* d_grad,const int dimxy);

__global__ void bc_keys_zn(const REAL *v,const REAL *u1,const REAL *u2, REAL *u3, const int nxne, const int nzne, const int ne, const int dimx, const int dimz, const int stride_z,const REAL h2,const REAL dt2,const REAL hdt);
__global__ void bc_keys_xn(const REAL *v,const REAL *u1,const REAL *u2, REAL *u3, const int nxne, const int nzne, const int ne, const int dimx, const int dimz, const int stride_z,const REAL h2,const REAL dt2,const REAL hdt);
__global__ void bc_keys_x0(const REAL *v,const REAL *u1,const REAL *u2, REAL *u3, const int nxne, const int nzne, const int ne, const int dimx, const int dimz, const int stride_z,const REAL h2,const REAL dt2,const REAL hdt);
