#include <torch/extension.h>

// cuda
void c_fdm_forward(
    float* forward,
    float* source_wavefield,
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

void c_fdm_backward(
    float* grad,
    float* source_wavefield,
    float* u1,
    float* u2,
    float* u3,
    float* vel,
    float* resid,
    const int order,
    const int dimx,
    const int dimy,
    const int nt,
    const float h,
    const float dt,
    const float ry
);

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

// cpp
void fdm_forward(
    torch::Tensor forward,
    torch::Tensor source_wavefield,
    torch::Tensor u1,
    torch::Tensor u2,
    torch::Tensor u3,
    torch::Tensor vel,
    torch::Tensor w,
    const int order,
    const int dimx,
    const int dimy,
    const int nt,
    const float h,
    const float dt,
    const float sx,
    const float sy,
    const float ry)
{
    CHECK_INPUT(forward);
    CHECK_INPUT(source_wavefield);
    CHECK_INPUT(u1);
    CHECK_INPUT(u2);
    CHECK_INPUT(u3);
    CHECK_INPUT(vel);
    CHECK_INPUT(w);
    c_fdm_forward(
            forward.data_ptr<float>(),
            source_wavefield.data_ptr<float>(),
            u1.data_ptr<float>(),
            u2.data_ptr<float>(),
            u3.data_ptr<float>(),
            vel.data_ptr<float>(),
            w.data_ptr<float>(),
            order,dimx,dimy,nt, h,dt,sx,sy,ry);
}

torch::Tensor fdm_backward(
    torch::Tensor source_wavefield,
    torch::Tensor u1,
    torch::Tensor u2,
    torch::Tensor u3,
    torch::Tensor vel,
    torch::Tensor resid,
    const int order,
    const int dimx,
    const int dimy,
    const int nt,
    const float h,
    const float dt,
    const float ry)
{
    CHECK_INPUT(source_wavefield);
    CHECK_INPUT(u1);
    CHECK_INPUT(u2);
    CHECK_INPUT(u3);
    CHECK_INPUT(vel);
    CHECK_INPUT(resid);
    torch:: Tensor grad = torch::zeros_like(vel);
    c_fdm_backward(
            grad.data_ptr<float>(),
            source_wavefield.data_ptr<float>(),
            u1.data_ptr<float>(),
            u2.data_ptr<float>(),
            u3.data_ptr<float>(),
            vel.data_ptr<float>(),
            resid.data_ptr<float>(),
            order,dimx,dimy,nt,h,dt,ry);
    return grad;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &fdm_forward, "FDM forward (C)");
    m.def("backward",&fdm_backward,"FDM backward (C)");
}
