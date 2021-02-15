#include <torch/extension.h>

// cuda
void cuda_fdm_forward_only(
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

void cuda_fdm_backward(
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// cpp
void fdm_forward(
    torch::Tensor forward,
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
    CHECK_INPUT(u1);
    CHECK_INPUT(u2);
    CHECK_INPUT(u3);
    CHECK_INPUT(vel);
    //CHECK_INPUT(w); //w in cpu
    cuda_fdm_forward_only(
            forward.data_ptr<float>(),
            u1.data_ptr<float>(),
            u2.data_ptr<float>(),
            u3.data_ptr<float>(),
            vel.data_ptr<float>(),
            w.data_ptr<float>(),
            order,dimx,dimy,nt, h,dt,sx,sy,ry);
}

torch::Tensor fdm_backward()
{
    torch:: Tensor grad = torch::zeros(1);
    return grad;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &fdm_forward, "FDM forward (CUDA)");
    m.def("backward",&fdm_backward,"FDM backward (CUDA)");
}
