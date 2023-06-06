//
// Created by 孙文礼 on 2023/1/25.
//
#include <DEFINES.h>
#include <iostream>
#include <PreciseIndex.hpp>
#include "DataSet.hpp"
#include "experience.hpp"
#include "HitsIndex.hpp"
#include <TimerClock.hpp>
#include <random>
#include<functional>
#include<algorithm>

//#define max_root_size (256 * 1024 * 1024)
//#define max_seg_size 16

#define max_root_size (256 * 1024 * 1024)
#define max_seg_size 16

//memory:26969.9MB
//time:158.999
//speed:6.28934
//memory:26969.9MB
//time:129.043
//speed:7.74933

int bulkload_num = 1000000;
void precise_index(std::vector<std::pair<key_type, value_type>> dataset) {
    TimerClock tc;
    FanOutGenerator root_fg(min_root_fan_out, max_root_fan_out, granularity);
    FanOutGenerator inner_fg(min_inner_fan_out, max_inner_fan_out, granularity);
    FanOutGenerator data_size_std(min_data_size, max_data_size, granularity);
    auto exp_chosen = experience_t();
    char index_block[sizeof(PreciseIndex<key_type, value_type>)];
    auto min_max = get_min_max(dataset);
    StatisticsDistribution(dataset, min_max.first, min_max.second, exp_chosen);
    exp_chosen.type = experience_t::FLOAT_64;
    exp_chosen.data_size = dataset.size();
    exp_chosen.hardware_insert_score = hardware_insert_score_value;
    exp_chosen.hardware_select_score = hardware_select_score_value;
    exp_chosen.hardware_erase_score = hardware_erase_score_value;
    exp_chosen.hardware_clear_score = hardware_clear_score_value;
    std::vector<int> fan_out_knob;
    fan_out_knob.push_back(dataset.size());
    for (int i = 0; i < MAX_LAYER; ++i) {
        fan_out_knob.push_back(default_inner_fan_out);
    }

    for (int i = 0; i < MAX_LAYER; ++i) {
        exp_chosen.fan_outs[i] = fan_out_knob[i];
    }
    auto index = new(index_block)PreciseIndex<key_type, value_type>(min_max.first, min_max.second, fan_out_knob);
    for (int i = 0;i<bulkload_num;++i) {
        index->add(dataset[i].first, dataset[i].second);
    }
    dataset.erase(dataset.begin(),dataset.begin() + bulkload_num);

    std::cout << "knob:"
              << setw(11) << fan_out_knob[0]
              << setw(4) << fan_out_knob[1]
              << setw(4) << fan_out_knob[2]
              << setw(4) << fan_out_knob[3]
              << setw(4) << fan_out_knob[4]
              << setw(4) << fan_out_knob[5]
              << setw(4) << fan_out_knob[6]
              << setw(4) << fan_out_knob[7]
              << RED << "  test_count:" << setw(4) << dataset.size() / (1000000)
              << "*10**6" << RESET << "  weight:" << setw(6) << std::endl;
    tc.synchronization();
    for (auto &i: dataset) {
        index->add(i.first, i.second);
    }
    exp_chosen.cost.add = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    exp_chosen.cost.memory = index->memory_occupied();
    tc.synchronization();
    value_type value;
    for (auto &i: dataset) {
        if (!index->get(i.first, value)) {
            std::cout << "error:" << i << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    tc.synchronization();
    for (auto &i: dataset) {
        index->erase(i.first);
    }
    exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    tc.synchronization();
    index->clear(index->root);
    exp_chosen.cost.clear = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    index->~PreciseIndex();
    std::cout << setw(6) << exp_chosen.cost.memory;
    std::cout << " : " << setw(6) << exp_chosen.cost.add;
    std::cout << " : " << setw(6) << exp_chosen.cost.get;
    std::cout << " : " << setw(6) << exp_chosen.cost.erase;
    std::cout << " : " << setw(6) << exp_chosen.cost.clear << std::endl;
}

void hits_index(std::vector<std::pair<key_type, value_type>> dataset) {
    TimerClock tc;
    FanOutGenerator root_fg(min_root_fan_out, max_root_fan_out, granularity);
    FanOutGenerator inner_fg(min_inner_fan_out, max_inner_fan_out, granularity);
    FanOutGenerator data_size_std(min_data_size, max_data_size, granularity);
    auto exp_chosen = experience_t();
    char index_block[sizeof(HitsIndex<key_type, value_type>)];
    auto min_max = get_min_max(dataset);
    StatisticsDistribution(dataset, min_max.first, min_max.second, exp_chosen);
    exp_chosen.type = experience_t::FLOAT_64;
    exp_chosen.data_size = dataset.size();
    exp_chosen.hardware_insert_score = hardware_insert_score_value;
    exp_chosen.hardware_select_score = hardware_select_score_value;
    exp_chosen.hardware_erase_score = hardware_erase_score_value;
    exp_chosen.hardware_clear_score = hardware_clear_score_value;
    std::vector<int> fan_out_knob;
    fan_out_knob.push_back(dataset.size());
    for (int i = 0; i < MAX_LAYER; ++i) {
        fan_out_knob.push_back(default_inner_fan_out);
    }
    for (int i = 0; i < MAX_LAYER; ++i) {
        exp_chosen.fan_outs[i] = fan_out_knob[i];
    }
    auto index = new(index_block)HitsIndex<key_type, value_type>(dataset, fan_out_knob);
    for (int i = 0;i<bulkload_num;++i) {
        index->add(dataset[i].first, dataset[i].second);
    }
    dataset.erase(dataset.begin(),dataset.begin() + bulkload_num);
    std::cout << "knob:"
              << setw(11) << fan_out_knob[0]
              << setw(4) << fan_out_knob[1]
              << setw(4) << fan_out_knob[2]
              << setw(4) << fan_out_knob[3]
              << setw(4) << fan_out_knob[4]
              << setw(4) << fan_out_knob[5]
              << setw(4) << fan_out_knob[6]
              << setw(4) << fan_out_knob[7]
              << RED << "  test_count:" << setw(4) << dataset.size() / (1000000)
              << "*10**6" << RESET << "  weight:" << setw(6) << std::endl;
    tc.synchronization();
    for (auto &i: dataset) {
        index->add(i.first, i.second);
    }
    exp_chosen.cost.add = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    exp_chosen.cost.memory = index->memory_occupied();
    tc.synchronization();
    value_type value;
    for (auto &i: dataset) {
        if (!index->get(i.first, value)) {
            std::cout << "error:" << i << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    tc.synchronization();
    for (auto &i: dataset) {
        index->erase(i.first);
    }
    exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    tc.synchronization();
    index->clear();
    exp_chosen.cost.clear = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    index->~HitsIndex();
    std::cout << setw(6) << exp_chosen.cost.memory;
    std::cout << " : " << setw(6) << exp_chosen.cost.add;
    std::cout << " : " << setw(6) << exp_chosen.cost.get;
    std::cout << " : " << setw(6) << exp_chosen.cost.erase;
    std::cout << " : " << setw(6) << exp_chosen.cost.clear << std::endl;
}

int main() {
//    auto dataset_1 = dataset_source::get_dataset<key_type>("longlat_raw.bin");
//    auto dataset_2 = dataset_source::get_dataset<std::pair<key_type,value_type>>("longlat_cv.bin");
//    for(long long i = 0;i<dataset_1.size();i++){
//        if(dataset_1[i] != dataset_2[i].first){
//            puts("error");
//        }
//    }
//    return 0;
//    auto dataset = dataset_source::get_dataset<std::pair<key_type, value_type>>("longlat.bin");
//    auto scores = standard_score(dataset);
//    hardware_insert_score_value = scores[0];
//    hardware_select_score_value = scores[1];
//    hardware_erase_score_value = scores[2];
//    hardware_clear_score_value = scores[3];

    std::vector<std::string> dataset_names = experiments_dataset_names;
    default_inner_fan_out = 10;
//    for (auto filename: dataset_names) {
//        auto data_set = dataset_source::get_dataset<std::pair<key_type, value_type>>(filename);
//        index_type = "precise";
//        precise_index(data_set);
//    }

    std::cout << "H" << std::endl;
    default_inner_fan_out = 7;
    for (auto filename: dataset_names) {
        auto data_set = dataset_source::get_dataset<std::pair<key_type, value_type>>(filename);
        index_type = "hits";
        hits_index(data_set);
    }
    return 0;



    for (auto filename: dataset_names) {
        auto data_set = dataset_source::get_dataset<std::pair<key_type, value_type>>(filename);
        data_set.erase(data_set.begin() + data_set.size() * 0.5, data_set.end());
        for (auto weight = float(0.003); weight < 1; weight *= 1.5) {
            index_type = "precise";
            precise_index(data_set);
            continue;
            /////////////////////
//        index_type = "hits";
//        std::cout<<"HO"<<std::endl;
            hits_index(data_set);
//            std::cout << "HT" << std::endl;
            hits_index(data_set);
        }
    }
    for (auto filename: dataset_names) {
        auto data_set = dataset_source::get_dataset<std::pair<key_type, value_type>>(filename);
        data_set.erase(data_set.begin() + data_set.size() * 0.5, data_set.end());
        for (auto weight = float(0.003); weight < 1; weight *= 1.5) {
//            index_type = "precise";
//            precise_index(data_set, true);
//            continue;
            /////////////////////
//        index_type = "hits";
//        std::cout<<"HO"<<std::endl;
//            hits_index(data_set,false);
//            std::cout << "HT" << std::endl;
            hits_index(data_set);
        }
    }
    return 0;
}

