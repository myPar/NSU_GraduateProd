#pragma once
#include "../baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class LeakyReLU final : public Layer<LeakyReLU> {
            float alpha_{0.01f};

        public:
            LeakyReLU(Stream& file);
            Tensor operator()(const Tensor& in) const noexcept override;
        };
    }
}