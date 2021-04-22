
void cuda_fdm_forward(
    float *forward,
    float *virt,
    float *d_u1,
    float *d_u2,
    float *d_u3,
    float *d_vel,
    float *w,
    const int order,
    const int dimx,
    const int dimy,
    const int nt,
    const float h,
    const float dt,
    const float sx,
    const float sy,
    const float ry);


void cuda_fdm_backward(
    float *grad,
    float *d_virt,
    float *d_u1,
    float *d_u2,
    float *d_u3,
    float *d_vel,
    float *resid,
    const int order,
    const int dimx,
    const int dimy,
    const int nt,
    const float h,
    const float dt,
    const float ry);


void cuda_fdm_forward_only(
    float *forward,
    float *d_u1,
    float *d_u2,
    float *d_u3,
    float *d_vel,
    float *w,
    const int order,
    const int dimx,
    const int dimy,
    const int nt,
    const float h,
    const float dt,
    const float sx,
    const float sy,
    const float ry);
