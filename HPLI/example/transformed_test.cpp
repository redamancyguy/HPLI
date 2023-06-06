//
// Created by redamancyguy on 23-4-2.
//

#include <DEFINES.h>
#include <iostream>
#include<vector>
#include <experience.hpp>
#include <TimerClock.hpp>
#include "HitsIndex.hpp"
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;
#define layer_size 8

int main() {
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    for (auto &i: dataset_names) {
        std::vector<std::int_fast32_t> fan_outs = {512 * 1024 * 1024, layer_size, layer_size, layer_size, layer_size,
                                                   layer_size, layer_size, layer_size};
        auto data_set = dataset_source::get_dataset<std::pair<key_type, value_type>>(i);
        fan_outs.front() = data_set.size();
        HitsIndex<key_type, value_type> h(data_set, fan_outs);
        experience_t exp_chosen;
        auto min_max = get_min_max(data_set);
        StatisticsDistribution(data_set, min_max.first, min_max.second, exp_chosen);
        exp_chosen.type = experience_t::FLOAT_64;
        exp_chosen.data_size = data_set.size();
        exp_chosen.hardware_insert_score = hardware_insert_score_value;
        exp_chosen.hardware_select_score = hardware_select_score_value;
        exp_chosen.hardware_erase_score = hardware_erase_score_value;
        exp_chosen.hardware_clear_score = hardware_clear_score_value;
        for (int ii = 0; ii < MAX_LAYER; ++ii) {
            exp_chosen.fan_outs[ii] = fan_outs[ii];
        }
        std::cout << "knob:" << setw(40) << fan_outs << RED
                  << "  test_count:" << setw(5) << data_set.size() / (1000000)
                  << "*10**6" << RESET << std::endl;
        TimerClock tc;
        tc.synchronization();
        for (auto &ii: data_set) {
            h.add(ii.first, ii.second);
        }
        exp_chosen.cost.add = (std::float_t) ((std::double_t) tc.get_timer_nanoSec() /
                                              ((std::double_t) data_set.size()));
        exp_chosen.cost.memory = h.memory_occupied();
        tc.synchronization();
        value_type value;
        for (auto &ii: data_set) {
            if (!h.get(ii.first, value) || value != ii.second) {
                std::cout << "error:" << ii << std::endl;
            }
        }
        exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) data_set.size()));
        tc.synchronization();
        for (auto &ii: data_set) {
            h.erase(ii.first);
        }
        exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) data_set.size()));
        tc.synchronization();
        h.clear();
        exp_chosen.cost.clear = (float) ((double) tc.get_timer_nanoSec() / ((double) data_set.size()));
        std::cout << "   real :" << setw(6) << exp_chosen.cost.memory;
        std::cout << " : " << setw(6) << exp_chosen.cost.add;
        std::cout << " : " << setw(6) << exp_chosen.cost.get;
        std::cout << " : " << setw(6) << exp_chosen.cost.erase;
        std::cout << " : " << setw(6) << exp_chosen.cost.clear << std::endl;
        std::cout << "knob:" << fan_outs << "  test_count:" << data_set.size() / (1000000)
                  << "*10**6" << std::endl;
        auto index = new PreciseIndex<key_type, value_type>(min_max.first, min_max.second, fan_outs);
        tc.synchronization();
        for (auto &ii: data_set) {
            index->add(ii.first, ii.second);
        }
        exp_chosen.cost.add = (std::float_t) ((std::double_t) tc.get_timer_nanoSec() /
                                              ((std::double_t) data_set.size()));
        exp_chosen.cost.memory = h.memory_occupied();
        tc.synchronization();
        for (auto &ii: data_set) {
            if (!index->get(ii.first, value)) {
                std::cout << "error:" << ii << std::endl;
            }
        }
        exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) data_set.size()));
        tc.synchronization();
        for (auto &ii: data_set) {
            index->erase(ii.first);
        }
        exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) data_set.size()));
        tc.synchronization();
        index->clear(index->root);
        exp_chosen.cost.clear = (float) ((double) tc.get_timer_nanoSec() / ((double) data_set.size()));
        puts("raw-index");
        std::cout << "   real :" << setw(6) << exp_chosen.cost.memory;
        std::cout << " : " << setw(6) << exp_chosen.cost.add;
        std::cout << " : " << setw(6) << exp_chosen.cost.get;
        std::cout << " : " << setw(6) << exp_chosen.cost.erase;
        std::cout << " : " << setw(6) << exp_chosen.cost.clear << std::endl;
        puts("==================");
        delete index;
    }
    return 0;
}