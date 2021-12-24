import torch

def _get_propagator(device, prop='ext', exa=False, forward_only=False):
    if device == 'cuda':
        if forward_only:
            from torchwi.propagator import td2d_forward_cuda as prop
        elif exa:
            from torchwi.propagator import td2d_exa_cuda as prop
        else:
            from torchwi.propagator import td2d_cuda as prop
    else: # cpu
        if prop == 'ext':
            if forward_only:
                from torchwi.propagator import td2d_forward_cpu as prop
            elif exa:
                #from torchwi.propagator import td2d_exa_cpu as prop
                raise NotImplementedError("cpu exa")
            else:
                from torchwi.propagator import td2d_cpu as prop
        elif prop == 'numba':
            if forward_only:
                raise NotImplementedError("numba forward_only")
            elif exa:
                raise NotImplementedError("numba exa")
            else:
                from torchwi.propagator import td2d_numba as prop
        else:
            raise NotImplementedError("Check prop=[ext|numba]")
    return prop


def get_operator(device, prop='ext', exa=False, forward_only=False):
    _propagator = _get_propagator(device, prop, exa, forward_only)

    if forward_only:

        class TimeForwardOperator(torch.autograd.Function):

            @staticmethod
            def forward(ctx, vel, args):
                # batch size=1
                # input/output vel,grad: (nx,ny)
                # frd: (nt,nx)
                # virt: (nt,dimxy)
                # internal vpad: (dimy,dimx)
                sx,sy,ry,m = args

                _propagator.forward(m.frd, 
                        m.u1, m.u2, m.u3, m.vpad, m.w,
                        m.order, m.dimx, m.dimy, m.nt,
                        m.h, m.dt,
                        sx.item(),sy.item(),ry.item())
                        
                return m.frd.view(m.nt,m.nx)[:,:m.nx_org]

            @staticmethod
            def backward(ctx, grad_output):
                return None, None

        return TimeForwardOperator.apply

    elif exa:

        class TimeEXAOperator(torch.autograd.Function):

            @staticmethod
            def forward(ctx, vel, args):
                # batch size=1
                # input/output vel,grad: (nx,ny)
                # frd: (nt,nx)
                # virt: (nt,dimxy)
                # internal vpad: (dimy,dimx)
                sx,sy,ry,m = args

                _propagator.forward(m.frd, m.exa, m.iexa,
                        m.u1, m.u2, m.u3, m.vpad, m.w,
                        m.order, m.dimx, m.dimy, m.nt,
                        m.h, m.dt,
                        sx.item(),sy.item(),ry.item())
                        
                # save for gradient calculation
                ctx.model = m
                ctx.save_for_backward(m.exa, m.iexa, ry)
                return m.frd.view(m.nt,m.nx)[:,:m.nx_org]

            @staticmethod
            def backward(ctx, grad_output):
                # resid = grad_output: (nt,nx)
                # grad_input: (nx,ny)
                exa, iexa, ry = ctx.saved_tensors
                m = ctx.model

                resid = grad_output
                grad = _propagator.backward(exa, iexa,
                        m.u1, m.u2, m.u3, m.vpad, resid,
                        m.order, m.dimx, m.dimy, m.nt,
                        m.h, m.dt, ry.item())
                # grad shape=(dimx,) # x fast
                grad = grad.view(m.dimy,m.dimx)[m.ne:m.ne+m.ny_org,m.ne:m.ne+m.nx_org]
                # grad_input: (nx,ny)
                grad_input = grad.transpose(0,1)
                return grad_input, None

        return TimeEXAOperator.apply

    else:

        class TimeOperator(torch.autograd.Function):

            @staticmethod
            def forward(ctx, vel, args):
                # batch size=1
                # input/output vel,grad: (nx,ny)
                # frd: (nt,nx)
                # virt: (nt,dimxy)
                # internal vpad: (dimy,dimx)
                sx,sy,ry,m = args

                _propagator.forward(m.frd, m.virt,
                        m.u1, m.u2, m.u3, m.vpad, m.w,
                        m.order, m.dimx, m.dimy, m.nt,
                        m.h, m.dt,
                        sx.item(),sy.item(),ry.item())
                        
                # save for gradient calculation
                ctx.model = m
                ctx.save_for_backward(m.virt,ry)
                return m.frd.view(m.nt,m.nx)[:,:m.nx_org]

            @staticmethod
            def backward(ctx, grad_output):
                # resid = grad_output: (nt,nx)
                # grad_input: (nx,ny)
                virt,ry = ctx.saved_tensors
                m = ctx.model

                resid = grad_output
                grad = _propagator.backward(virt,
                        m.u1, m.u2, m.u3, m.vpad, resid,
                        m.order, m.dimx, m.dimy, m.nt,
                        m.h, m.dt, ry.item())
                # grad shape=(dimx,) # x fast
                grad = grad.view(m.dimy,m.dimx)[m.ne:m.ne+m.ny_org,m.ne:m.ne+m.nx_org]
                # grad_input: (nx,ny)
                grad_input = grad.transpose(0,1)
                return grad_input, None

        return TimeOperator.apply

