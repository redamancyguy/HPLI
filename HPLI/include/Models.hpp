
#ifndef Q_MODEL_HPP
#define Q_MODEL_HPP

#include <DEFINES.h>
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/nn/module.h>

#include <utility>


std::string index_type = "hits";
#define hidden_size 1

class CDF_TRAN : public torch::nn::Module {
    torch::nn::BatchNorm1d bn = nullptr;
    torch::nn::Linear fc_in = nullptr;
    torch::nn::Linear fc_out = nullptr;
public:
    CDF_TRAN(std::int32_t input_dim, std::int32_t output_dim) {
        bn = register_module("bn", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(input_dim)));
        fc_in = register_module("fc_in", torch::nn::Linear(input_dim, hidden_size));
        fc_out = register_module("fc_out", torch::nn::Linear(hidden_size, output_dim));
        fc_in->weight = torch::nn::init::uniform_(fc_in->weight, -1e-1, 1e-1);
        fc_out->weight = torch::nn::init::uniform_(fc_out->weight, -1e-1, 1e-1);
        this->to(torch::kDouble);
        *fc_in->weight.data_ptr<double>() = 1;
        *fc_in->bias.data_ptr<double>() = 0;
        *fc_out->weight.data_ptr<double>() = 1;
        *fc_out->bias.data_ptr<double>() = 0;
    }

    void shuffle(){
        *fc_in->weight.data_ptr<double>() = 1;
        *fc_in->bias.data_ptr<double>() = 0;
        *fc_out->weight.data_ptr<double>() = 1;
        *fc_out->bias.data_ptr<double>() = 0;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = bn->forward(x);
        x = fc_in->forward(x);
        x = torch::sigmoid(x);
        x = fc_out->forward(x);
        return x;
    }

    void print() {
        std::cout << fc_in->weight << std::endl;
        std::cout << fc_in->bias << std::endl;
        std::cout << fc_out->weight << std::endl;
        std::cout << fc_out->bias << std::endl;
    }

    std::tuple<std::double_t, std::double_t, std::double_t, std::double_t> parameter() {
        auto result = std::make_tuple(
                std::double_t(*fc_in->weight.data_ptr<double>()),
                std::double_t(*fc_in->bias.data_ptr<double>()),
                std::double_t(*fc_out->weight.data_ptr<double>()),
                std::double_t(*fc_out->bias.data_ptr<double>())
        );
        return result;
    }
};


namespace SM {
    class XCubic {
    public:
        inline static double forward(double x) {
            return x * x * x;
        }

        inline static double inverse(double x) {
            return std::pow(x, 1.0 / 3.0);
        }

        inline static double derivation(double x) {
            return 3 * x * x;
        }
    };

    class XSquare_Inverse {
    public:
        inline static double forward(double x) {
            if (x > 0) {
                return x * x;
            } else {
                return -x * x;
            }
        }

        inline static double inverse(double x) {
            if (x < 0) {
                return std::pow(-x, 1.0 / 2.0);
            } else {
                return std::pow(x, 1.0 / 2.0);
            }
        }

        inline static double derivation(double x) {
            if (x < 0) {
                return -2 * x;
            } else {
                return 2 * x;
            }
        }
    };

    class XSquare {
    public:
        inline static double forward(double x) {
            if (x > 0) {
                return x * x;
            } else {
                return -x * x;
            }
        }

        inline static double inverse(double x) {
            if (x < 0) {
                return std::pow(-x, 1.0 / 2.0);
            } else {
                return std::pow(x, 1.0 / 2.0);
            }
        }

        inline static double derivation(double x) {
            if (x < 0) {
                return -2 * x;
            } else {
                return 2 * x;
            }
        }
    };

    class X {
    public:
        inline static double forward(double x) {
            return x;
        }

        inline static double inverse(double x) {
            return x;
        }

        inline static double derivation(double x) {
            return 1;
        }
    };

    class Sigmoid {
    public:
        inline static double forward(double x) {
            return 1 / (std::exp(-x) + 1);
        }

        inline static double inverse(double x) {
            return -std::log(1 / x - 1);
        }

        inline static double derivation(double x) {
            x = forward(x);
            return x * (1 - x);
        }
    };

    class SC {
    public:
        inline static double forward(double x) {
            return Sigmoid::forward(XCubic::forward(x));
        }

        inline static double inverse(double x) {
            return Sigmoid::inverse(XCubic::inverse(x));
        }

        inline static double derivation(double x) {
            return Sigmoid::derivation(XCubic::forward(x)) * XCubic::derivation(x);
        }
    };

    class SS {
    public:
        inline static double forward(double x) {
            return Sigmoid::forward(XSquare::forward(x));
        }

        inline static double derivation(double x) {
            return Sigmoid::derivation(XSquare::forward(x)) * XSquare::derivation(x);
        }
    };

    class Sin_and_X {
    public:
        inline static double forward(double x) {
            return 1.01 * x + std::sin(x);
        }

        inline static double derivation(double x) {
            return 1.01 + std::cos(x);
        }
    };

    class MSELoss {
    public:
        inline static double forward(double pred, double label) {
            auto x = pred - label;
            return x * x;
        }

        inline static double derivation(double pred, double label) {
            return 2 * (pred - label);
        }
    };
//#define activation Sigmoid
//#define activation SS
//#define activation SC
//#define activation X
//#define activation XSquare
//#define activation XCubic
//#define activation Sin_and_X

#define activation Sigmoid

#define loss_func MSELoss

    class TranModel {
        double w_1, b_1, w_2, b_2;
    public:
        [[nodiscard]] std::vector<double> forward(const std::vector<double> &x) const {
            std::vector<double> result;
            result.reserve(x.size());
            for (auto i: x) {
                result.push_back(activation::forward(w_1 * i + b_1) * w_2 + b_2);
            }
            return result;
        }

        [[nodiscard]] double forward(double x) const {
            return activation::forward(w_1 * x + b_1) * w_2 + b_2;
        }

        void backward(const std::vector<double> &x, const std::vector<double> &y, double lr = 0.01) {
            auto n = x.size();
            double d_l_d_w_2 = 0;
            double d_l_d_b_2 = 0;
            double d_l_d_w_1 = 0;
            double d_l_d_b_1 = 0;
            for (std::size_t i = 0; i < n; i++) {
                double d_b_2 = 1;
                double d_w_2 = activation::forward(w_1 * x[i] + b_1);
                double d_b_1 = w_2 * activation::derivation(w_1 * x[i] + b_1);
                double d_w_1 = d_b_1 * x[i];
                double pred = d_w_2 * w_2 + b_2;
                double d_loss = loss_func::derivation(pred, y[i]);
                d_l_d_w_2 += d_loss * d_w_2;
                d_l_d_b_2 += d_loss * d_b_2;
                d_l_d_w_1 += d_loss * d_w_1;
                d_l_d_b_1 += d_loss * d_b_1;
            }
            d_l_d_w_2 /= double(n);
            d_l_d_b_2 /= double(n);
            d_l_d_w_1 /= double(n);
            d_l_d_b_1 /= double(n);
            w_2 -= lr * d_l_d_w_2;
            b_2 -= lr * d_l_d_b_2;
            w_1 -= lr * d_l_d_w_1;
            b_1 -= lr * d_l_d_b_1;
        }

        std::tuple<double, double, double, double> parameter() {
            auto result = std::make_tuple(w_1, b_1, w_2, b_2);
            return result;
        }

        void show() const {
            std::cout << w_2 << " " << b_2 << " " << w_1 << " " << b_1 << " " << std::endl;
        }

        TranModel() {
            w_1 = w_2 = 1;
            b_1 = b_2 = 0;
        }
    };
}


#endif //Q_MODEL_HPP
