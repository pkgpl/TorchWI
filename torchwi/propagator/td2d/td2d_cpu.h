
void fdm_O2(const int nx, const int ny,
        const float* __restrict__ const u1, const float* __restrict__ const u2, float* __restrict__ const u3, const float* __restrict__ const v,
        const float dtoh2, const int ne, const int stride_y);

void fdm_O4(const int nx, const int ny,
        const float* __restrict__ const u1, const float* __restrict__ const u2, float* __restrict__ const u3, const float* __restrict__ const v,
        const float dtoh2, const int ne, const int stride_y);

void fdm_O8(const int nx, const int ny,
        const float* __restrict__ const u1, const float* __restrict__ const u2, float* __restrict__ const u3, const float* __restrict__ const v,
        const float dtoh2, const int ne, const int stride_y);

void bc_keys_v(const float* __restrict__ const um, const float* __restrict__ const uo, float* __restrict__ const up, const float* __restrict__ const v,
        const int nx, const int ny, const int ne, const int nxne,const int nyne,const int dimx,const int dimy,const int stride_y,
        const float h2,const float dt2,const float hdt);
