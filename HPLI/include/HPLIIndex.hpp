//
// Created by redamancyguy on 23-4-2.
//

#ifndef HITS_HITSINDEX_HPP
#define HITS_HITSINDEX_HPP


#include <cmath>
#include <PreciseIndex.hpp>
#include <Models.hpp>
#include <experience.hpp>


#ifdef show_curve

#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;
#endif


template<typename key_T, typename value_T>
class HitsIndex {
    char index[sizeof(PreciseIndex<key_T, value_T>)];
    double w_1, b_1, w_2, b_2;
public:
    inline PreciseIndex<key_T, value_T> *get_index() {
        return (PreciseIndex<key_T, value_T> *) (index);
    }

    std::vector<std::int_fast32_t> get_fan_out() {
        std::vector<std::int_fast32_t> result;
        for (int i = 0; i < MAX_LAYER; i++) {
            result.push_back(get_index()->fan_out_knob[i]);
        }
        return result;
    }
    void set_fan_out(std::vector<std::int_fast32_t> fan_out) {
        for (int i = 0; i < MAX_LAYER; i++) {
            get_index()->fan_out_knob[i] = fan_out[i];
        }
        get_index()->fan_out_knob[MAX_LAYER] = default_inner_fan_out;
    }

    HitsIndex(std::vector<std::pair<key_T, value_T>> dataset,
              const std::vector<int> &fan_out_knob) {
        std::vector<double> keys;
        std::vector<double> positions;
        for (std::size_t i = 0; i < dataset.size(); ++i) {
            keys.push_back(dataset[i].first);
            positions.push_back(
                    (double(i) - double(dataset.size()) / 2) / (double(dataset.size()) / 2));
        }
        std::sort(keys.begin(), keys.end());
        Standard std_x{};
        std_x.fit(keys);
        keys = std_x.forward(keys);
        SM::TranModel model;
        std::vector<double> x_vec;
        std::vector<double> y_vec;

#ifdef show_curve
        plt::ion();
#endif
        std::size_t batch_size = 10000;
        double min = 100000;
        double max = -100000;
        for (int _ = 0; _ < 3333; ++_) {
            for (std::size_t j = 0; j < batch_size; ++j) {
                auto random_index = e() % dataset.size();
                x_vec.push_back(keys[random_index]);
                y_vec.push_back(positions[random_index]);
            }
            model.backward(x_vec,y_vec,0.08);
#ifdef show_curve
            plt::cla();
            std::sort(x_vec.begin(), x_vec.end());
            std::sort(y_vec.begin(), y_vec.end());

            std::vector<double> pred_x;
            min = std::min(x_vec.front(),min);
            max = std::max(x_vec.back(),max);
            double step = (max - min) / 333;
            for(int ii = 0;ii< 333;ii++){
                pred_x.push_back(min + ii * step);
            }
            auto pred_y = model.forward(pred_x);
            std::map<std::string, std::string> color;
            color["color"] = "orange";
            color["label"] = "y_vec";
            plt::scatter(x_vec, y_vec, 1, color);
            color["color"] = "purple";
            color["label"] = "pred_vec";
            plt::plot(pred_x, pred_y, color);
            plt::pause(0.00001);
#endif
            x_vec.clear();
            y_vec.clear();
        }
        auto parameters = model.parameter();
        std::cout <<std::get<0>(parameters)<< std::endl;
        std::cout <<std::get<1>(parameters)<< std::endl;
        std::cout <<std::get<2>(parameters)<< std::endl;
        std::cout <<std::get<3>(parameters)<< std::endl;
        w_1 = std::get<0>(parameters) / std_x.var;
        b_1 = std::get<1>(parameters) - std::get<0>(parameters) * std_x.mean / std_x.var;
        w_2 = std::get<2>(parameters) * (double(dataset.size()) / 2);
        b_2 = std::get<3>(parameters) * (double(dataset.size()) / 2) + (double(dataset.size()) / 2);

        auto min_max = get_min_max(dataset);
        auto a = (min_max.second - min_max.first) / (tran_full(min_max.second) - tran_full(min_max.first));
        auto b = min_max.first - a * tran_full(min_max.first);
        w_2 = a * w_2;
        b_2 = a * b_2 + b;
        auto new_min = tran_full(min_max.first);
        auto new_max = tran_full(min_max.second);
        new(index)PreciseIndex<key_T, value_T>(new_min, new_max, fan_out_knob);
        get_index()->root->intercept = get_index()->root->slope * b_2 + get_index()->root->intercept;
        get_index()->root->slope = get_index()->root->slope * w_2;
    }

    ~HitsIndex() {
        get_index()->~PreciseIndex();
    }

    inline double sigmoid(double x) {
        return 1 / (std::exp(-x) + 1);
    }

    inline key_T tran(key_T key) {
//        return sigmoid(w_1 * key + b_1) * w_2 + b_2;
        return sigmoid(w_1 * key + b_1);
    }

    key_T tran_full(key_T key) {
        return sigmoid(w_1 * key + b_1) * w_2 + b_2;
    }

    inline bool add(key_T key, const value_T &value) {
        return get_index()->add(tran(key), value);
    }

    inline bool get(key_T key, value_T &value) {
        return get_index()->get(tran(key), value);
    }

    inline bool erase(key_T key) {
        return get_index()->erase(tran(key));
    }

    inline void clear() {
        get_index()->clear(get_index()->root);
    }

    inline float memory_occupied() {
        return (float(seg_memory_count) + sizeof(HitsIndex<key_T, value_T>)) / (1024 * 1024);
    }
};

#endif //HITS_HITSINDEX_HPP
