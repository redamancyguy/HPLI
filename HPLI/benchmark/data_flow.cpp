//
// Created by redamancyguy on 23-4-6.
//
//
// Created by 孙文礼 on 2023/1/25.
//
#include <DEFINES.h>
#include <iostream>
#include <PreciseIndex.hpp>
#include "DataSet.hpp"
#include <TimerClock.hpp>
#include <random>
#include<functional>
#include<algorithm>
#include <HitsIndex.hpp>
std::size_t windows_size = 30000000;
void precise_index(const std::vector<pair<key_type,value_type>> &data_set){
    auto end = data_set.size() - windows_size;
    auto min_max = get_min_max(data_set);
    TimerClock tc;
    tc.synchronization();
    std::vector<int> fan_out_knob;
    for (int i = 0; i < MAX_LAYER; ++i) {
        fan_out_knob.push_back(default_inner_fan_out);
    }
    fan_out_knob.front() = windows_size;
    std::cout<<"fan_out:"<<fan_out_knob<<std::endl;
    PreciseIndex<key_type, value_type> index(
            (double) min_max.first, (double) min_max.second, fan_out_knob);
    for (std::size_t i = 0; i < windows_size; i++) {
        index.add(data_set[i].first, data_set[i].second);
    }
    tc.synchronization();
    for (std::size_t i = 0; i < end; i++) {
        index.add(data_set[i + windows_size].first, data_set[i + windows_size].second);
        value_type value;
        index.get(data_set[i].first, value);
        index.erase(data_set[i].first);
    }
    index.clear(index.root);
    std::cout << "time:" << ((double) tc.get_timer_nanoSec() / ((double) data_set.size())) << std::endl;
    std::cout << "memory:" << index.memory_occupied() << "MB" << std::endl;
    tc.synchronization();
    auto data_get = std::vector<std::pair<key_type, value_type>>();
    index.get_all_data(index.root, data_get);
    index.re_build(data_get);
    std::cout << "rebuild-time:" << ((double) tc.get_timer_nanoSec() / ((double) end)) << std::endl;
}

void hits_index(const std::vector<pair<key_type,value_type>> &data_set){
    auto end = data_set.size() - windows_size;
    TimerClock tc;
    tc.synchronization();
    std::vector<int> fan_out_knob;
    for (int i = 0; i < MAX_LAYER; ++i) {
        fan_out_knob.push_back(default_inner_fan_out);
    }
    fan_out_knob.front() = windows_size;
    std::cout<<"fan_out:"<<fan_out_knob<<std::endl;
    HitsIndex<key_type, value_type> index(data_set, fan_out_knob);
    for (std::size_t i = 0; i < windows_size; i++) {
        index.add(data_set[i].first, data_set[i].second);
    }
    tc.synchronization();
    for (std::size_t i = 0; i < end; i++) {
        index.add(data_set[i + windows_size].first, data_set[i + windows_size].second);
        value_type value;
        index.get(data_set[i].first, value);
        index.erase(data_set[i].first);
    }
    index.clear();
    std::cout << "time:" << ((double) tc.get_timer_nanoSec() / ((double) data_set.size())) << std::endl;
    std::cout << "memory:" << index.memory_occupied() << "MB" << std::endl;
}

int main() {
//    auto dataset = dataset_source::get_dataset<std::pair<key_type , value_type>>("longlat.bin");
//    auto scores = standard_score(dataset);
//    hardware_insert_score_value = scores[0];
//    hardware_select_score_value = scores[1];
//    hardware_erase_score_value = scores[2];
//    hardware_clear_score_value = scores[3];
    std::vector<std::string> dataset_names = experiments_dataset_names;

//    for(auto filename:dataset_names){
//        auto data_set = dataset_source::get_dataset<std::pair<key_type,value_type>>(filename);
//        index_type = "precise";
//        precise_index(data_set);
//    }
    puts("=============");
    for(auto filename:dataset_names){
        auto data_set = dataset_source::get_dataset<std::pair<key_type,value_type>>(filename);
        index_type = "hits";
        hits_index(data_set);
    }
}

