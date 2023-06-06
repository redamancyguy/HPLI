//
// Created by 孙文礼 on 2023/1/10.
//

#ifndef DATASET_HPP
#define DATASET_HPP

#include <DEFINES.h>
#include <iostream>
#include <unordered_map>
using namespace std;

class dataset_source {
public:
    template<class data_type>
    static void set_data(const std::string &filename, const std::vector<data_type> &dataset_input) {
        std::FILE *out_file = std::fopen((data_father_path + filename).c_str(), "wb");
        auto *buffer = (data_type *) std::malloc(sizeof(data_type) * dataset_input.size());
        std::copy(dataset_input.begin(), dataset_input.end(), buffer);
        std::fwrite(buffer, sizeof(data_type), dataset_input.size(), out_file);
        std::free(buffer);
        std::fclose(out_file);
    }

    template<class data_type>
    static std::vector<data_type> get_dataset(const std::string &filename) {
        std::cout <<GREEN<< "filename:" << filename <<RESET<< std::endl;
        std::FILE *in_file = std::fopen((std::string(data_father_path) + filename).c_str(), "rb");
        std::fseek(in_file, 0, SEEK_END);
        std::size_t size = ::ftell(in_file) / sizeof(data_type);
        auto *buffer = (data_type *) std::malloc(size * sizeof(data_type));
        std::fseek(in_file, 0, SEEK_SET);
        std::vector<data_type> result(size);
        if (std::fread(buffer, size, sizeof(data_type), in_file) == 0) {}
        std::fclose(in_file);
        std::copy(buffer, buffer + size, result.begin());
        free(buffer);
        return result;
    }

    template<class data_type>
    static std::vector<data_type> get_path_dataset(const std::string &filename) {
        std::cout <<GREEN<< "filename:" << filename <<RESET<< std::endl;
        std::FILE *in_file = std::fopen(filename.c_str(), "rb");
        std::fseek(in_file, 0, SEEK_END);
        std::size_t size = ::ftell(in_file) / sizeof(data_type);
        auto *buffer = (data_type *) std::malloc(size * sizeof(data_type));
        std::fseek(in_file, 0, SEEK_SET);
        std::vector<data_type> result(size);
        if (std::fread(buffer, size, sizeof(data_type), in_file) == 0) {}
        std::fclose(in_file);
        std::copy(buffer, buffer + size, result.begin());
        free(buffer);
        return result;
    }

    template<class key_T, class value_T>
    static std::vector<std::pair<key_T, value_T>> random_dataset(double min, double max,
                                                                       int num) {
        std::unordered_map<key_T, value_T> result_map;
        std::vector<std::pair<key_T, value_T>> result;
        int i = 0;
        while (int(result_map.size()) < num) {
            result_map[(max - min) * random_u_0_1()] = i++;
        }
        for(auto &ii:result_map){
            result.push_back(i);
        }
        std::shuffle(result.begin(), result.end(),e);
        return result;
    }

private:
};

template<class key_T,class value_T>
std::pair<key_T, key_T> get_min_max(const std::vector<std::pair<key_T, value_T>> &dataset) {
    std::pair<key_T, key_T> result = {std::numeric_limits<key_T>::max(),
                                            -std::numeric_limits<key_T>::max()};
    for (auto i: dataset) {
        if (i.first < result.first) {
            result.first = i.first;
        }
        if (i.first > result.second) {
            result.second = i.first;
        }
    }
    return result;
}

template<class key_T,class value_T>
std::pair<std::vector<double>,std::vector<double>>get_pdf(const std::vector<std::pair<key_T, value_T>> &dataset) {
    int bucket_size = BUCKET_SIZE;
    std::vector<double> bucket(bucket_size);
    std::vector<double> keys(bucket_size);
    auto min_max = get_min_max(dataset);
    auto slope = (double(bucket_size) - 2 * epsilon_delta) / (min_max.second - min_max.first);
    auto intercept = (bucket_size - slope * (min_max.second + min_max.first)) / 2;
    for(auto &i:dataset){
        ++bucket[int(i.first * slope + intercept)];
    }
    for(double i = 0;i < bucket_size;++i){
        auto key = (i - intercept)/slope;
        keys[int(i)] = key;
    }
    return {keys,bucket};
}

template<class key_T,class value_T>
std::pair<std::vector<double>,std::vector<double>> get_cdf(const std::vector<std::pair<key_T, value_T>> &dataset) {
    auto pdf = get_pdf(dataset);
    int bucket_size = BUCKET_SIZE;
    std::vector<double> bucket;
    bucket.reserve(bucket_size);
    double cdf_count = 0;
    for(auto i:pdf.second){
        cdf_count += i;
        bucket.push_back(cdf_count);
    }
    return {pdf.first,bucket};
}
std::vector<std::string> experiments_dataset_names = {
        "longitudes.bin","longlat.bin",
        "asia-latitude-id-1.bin","north-america-latitude-id-0.bin",
        "ycsb.bin1"};



#endif //DATASET_HPP
