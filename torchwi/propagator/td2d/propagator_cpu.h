
void c_fdm_forward(
    float *forward,
    float *virt,
    float *u1,
    float *u2,
    float *u3,
    float *vel,
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

void c_fdm_backward(
    float *grad,
    float *virt,
    float *u1,
    float *u2,
    float *u3,
    float *vel,
    float *resid,
    const int order,
    const int dimx,
    const int dimy,
    const int nt,
    const float h,
    const float dt,
    const float ry);


void c_fdm_forward_only(
    float* forward,
    float* u1,
    float* u2,
    float* u3,
    float* vel,
    float* w,
    const int order,
    const int dimx,
    const int dimy,
    const int nt,
    const float h,
    const float dt,
    const float sx,
    const float sy,
    const float ry
);
