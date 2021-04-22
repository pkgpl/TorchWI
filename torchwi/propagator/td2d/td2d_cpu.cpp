#include <stdlib.h>
#include <omp.h>

#define C02 -2.0
#define C12  1.0
void fdm_O2(const int nx, const int ny,
        const float* __restrict__ const u1, const float* __restrict__ const u2, float* __restrict__ const u3, const float* __restrict__ const v,
        const float dtoh2, const int ne, const int stride_y)
{
	int i, ix, iy, iyne;
	float div;
	//float C0=-2.;
	//float C1=1.;

	#pragma omp parallel for private(iy,iyne,ix,i,div)
    //#pragma prefetch u2:0:4
	for (iy=0; iy < ny; iy++)
	{
		iyne = iy+ne;
        for (ix=0; ix < nx; ix++)
        {
            i = iyne*stride_y + ix + ne;
            div = C12*(u2[i-1]+u2[i-stride_y]
                      +u2[i+1]+u2[i+stride_y])
                 +C02*2.f*u2[i] ;
            u3[i] = 2.f*u2[i]-u1[i]+(v[i]*v[i]*dtoh2)*div;
        }
    }
}

#define C04 -2.5
#define C14  1.3333333333333333
#define C24 -0.08333333333333333
void fdm_O4(const int nx, const int ny,
        const float* __restrict__ const u1, const float* __restrict__ const u2, float* __restrict__ const u3, const float* __restrict__ const v,
        const float dtoh2, const int ne, const int stride_y)
{
	int i, ix, iy, iyne;
	float div;
	//float C0 = -5./2.;
	//float C1 = 4./3.;
	//float C2 = -1./12.;
	
	#pragma omp parallel for private(iy,iyne,ix,i,div)
    //#pragma prefetch u2:0:4
	for (iy=0; iy < ny; iy++)
	{
		iyne=iy+ne;
        for (ix=0; ix < nx; ix++)
        {
            i = iyne*stride_y + ix + ne;
            div = C24*(u2[i-2*stride_y]+u2[i-2]+u2[i+2]+u2[i+2*stride_y]) 
                + C14*(u2[i-  stride_y]+u2[i-1]+u2[i+1]+u2[i+  stride_y]) 
                + C04*2.f*u2[i] ;
            u3[i] = div * (v[i]*v[i]*dtoh2) + 2.f*u2[i]-u1[i];
        }
	}
}


//#define C083  -8.541666666666668
#define C08 -2.8472222222222223
#define C18  1.6
#define C28 -0.2
#define C38  0.025396825396825397
#define C48 -0.0017857142857142857
// stride_x = 1
// stride_y = dimx
void fdm_O8(const int nx, const int ny,
        const float* __restrict__ const u1, const float* __restrict__ const u2, float* __restrict__ const u3, const float* __restrict__ const v,
        const float dtoh2, const int ne, const int stride_y)
{
	int i, ix, iy, iyne;
	float div;
	//float C0 = -205./72.;
	//float C1 = 8./5.;
	//float C2 = -1./5.;
	//float C3 = 8./315.;
	//float C4 = -1./560.;

	#pragma omp parallel for private(iy,ix,iyne,i,div)
    //#pragma prefetch u2:0:4
	for (iy=0; iy < ny; iy++)
	{
		iyne=iy+ne;
        //#pragma omp simd private(ix,iy,iyne,i,div)
        for (ix=0; ix < nx; ix++)
        {
            i = iyne*stride_y + ix + ne;
            div	= C48*(u2[i-4*stride_y]+u2[i-4]+u2[i+4]+u2[i+4*stride_y])
                + C38*(u2[i-3*stride_y]+u2[i-3]+u2[i+3]+u2[i+3*stride_y])
                + C28*(u2[i-2*stride_y]+u2[i-2]+u2[i+2]+u2[i+2*stride_y])
                + C18*(u2[i-  stride_y]+u2[i-1]+u2[i+1]+u2[i+  stride_y])
                + C08*2.f*u2[i];
            u3[i] = div * (v[i]*v[i]*dtoh2) + 2.f*u2[i]-u1[i];
        }
	}
}


// theta1 = M_PI/6 = 0.5235987755982988
// theta2 = M_PI/12= 0.2617993877991494
// ct1 = cos(theta1) = 0.8660254037844387
// ct2 = cos(theta2) = 0.9659258262890683
// ct12m = ct1*ct2 = 0.8365163037378079
// ct12p = ct1+ct2 = 1.831951230073507
#define CT12M 0.8365163037378079
#define CT12P 1.831951230073507
// um(minus)=u1,uo(zero)=u2,up(plus)=u3
void bc_keys_v(const float* __restrict__ const um, const float* __restrict__ const uo, float* __restrict__ const up, const float* __restrict__ const v,
        const int nx, const int ny, const int ne, const int nxne,const int nyne,const int dimx,const int dimy,const int stride_y,
        const float h2,const float dt2,const float hdt)
{
    float uxx,uxt;
    int ix,iy,i;
    // right
    #pragma omp parallel for private(ix,iy,i,uxx,uxt)
    for(iy=ne;iy<nyne;iy++)
        for(ix=nxne;ix<dimx;ix++)
        {
            i=iy*stride_y+ix;
            uxx=(uo[i]-2.*uo[i-1]+uo[i-2])/h2;
            uxt=((uo[i]-um[i])-(uo[i-1]-um[i-1]))/hdt;
            up[i]=-v[i]*v[i]*dt2/CT12M*(uxx+CT12P/v[i]*uxt)+2.*uo[i]-um[i];
        }
    // left
    #pragma omp parallel for private(ix,iy,i,uxx,uxt)
    for(iy=ne;iy<nyne;iy++)
        for(ix=ne-1;ix>=0;ix--)
        {
            i=iy*stride_y+ix;
            uxx=(uo[i]-2.*uo[i+1]+uo[i+2])/h2;
            uxt=((uo[i]-um[i])-(uo[i+1]-um[i+1]))/hdt;
            up[i]=-v[i]*v[i]*dt2/CT12M*(uxx+CT12P/v[i]*uxt)+2.*uo[i]-um[i];
        }
    // bottom
    #pragma omp parallel for private(ix,iy,i,uxx,uxt)
    for(iy=nyne;iy<dimy;iy++)
        for(ix=0;ix<dimx;ix++)
        {
            i=iy*stride_y+ix;
            uxx=(uo[i]-2.*uo[i-stride_y]+uo[i-2*stride_y])/h2;
            uxt=((uo[i]-um[i])-(uo[i-stride_y]-um[i-stride_y]))/hdt;
            up[i]=-v[i]*v[i]*dt2/CT12M*(uxx+CT12P/v[i]*uxt)+2.*uo[i]-um[i];
        }
}
