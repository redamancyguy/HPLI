//
// Created by redamancyguy on 23-3-29.
//

#ifndef HITS_PLOT_UTILS_HPP
#define HITS_PLOT_UTILS_HPP
#include <matplotlibcpp.h>
#include <vector>
#include "experience.hpp"

template<class T>
void plt_pdf_exp(std::vector<T> &input) {
    auto min = std::min_element(input.begin(), input.end());
    auto max = std::max_element(input.begin(), input.end());
    std::unordered_map<double,int> bucket;
    for(auto i:input){
        if(bucket.find(i) != bucket.end()){
            ++bucket[i];
        }else{
            bucket[i] = 1;
        }
    }
    namespace plt = matplotlibcpp;
    for(auto )
    plt::plot();
}

void plt_cdf_exp(experience_t &exp_input) {
    std::vector<double> y;
    double acc = 0;
    for (float i: exp_input.distribution) {
        acc += i;
        y.push_back(acc);
    }
    namespace plt = matplotlibcpp;
    plt::plot(y);
}
#endif //HITS_PLOT_UTILS_HPP
