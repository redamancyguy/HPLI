#include<ATen/ATen.h>//引入头文件
#include<iostream>
#include <torch/torch.h>
using namespace std;
int main(){
//    auto a = torch::linspace(-3,2,5);
//    auto ze = torch::zeros(a.sizes());
//    std::cout<<a<<std::endl;
//    std::cout<<ze<<std::endl;
//    auto b = torch::hstack({a.view({-1,1}),ze.view({-1,1})});
//    auto c = b.max(1,true);
//    std::cout<<get<0>(c)<<std::endl;
//    std::cout<<get<1>(c)<<std::endl;
    for (auto weight = float(0.003); weight < 0.5; weight *= 1.1) {
        std::cout<<weight<<std::endl;
    }
}