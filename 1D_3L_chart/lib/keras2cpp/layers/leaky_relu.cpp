#include "leaky_relu.h"

namespace keras2cpp{
    namespace layers{
        LeakyReLU::LeakyReLU(Stream& file) : alpha_(file) {}    
        Tensor LeakyReLU::operator()(const Tensor& in) const noexcept {
            kassert(in.ndim());
            Tensor out;
            out.data_.resize(in.size());
            out.dims_ = in.dims_;

            std::transform(in.begin(), in.end(), out.begin(), [this](float x) {
                if (x >= 0.f)
                    return x;
                return alpha_ * x;
            });
            return out;
        }
    }
}