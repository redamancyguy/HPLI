//
// Created by 孙文礼 on 2023/1/26.
//

#ifndef EXPERIENCE_HPP
#define EXPERIENCE_HPP

#include <torch/torch.h>
#include <DEFINES.h>
#include <DataSet.hpp>
#include <unistd.h>

class Cost {
public:
    float memory;
    float add;
    float get;
    float erase;
    float clear;
};

class Standard {
public:
    double mean;
    double var;

    void save(std::string filename) {
        auto *file = ::fopen((model_father_path + filename).c_str(), "w");
        std::fwrite(&mean, sizeof(double), 1, file);
        std::fwrite(&var, sizeof(double), 1, file);
        fclose(file);
    }

    void load(std::string filename) {
        auto *file = ::fopen((model_father_path + filename).c_str(), "r");
        if (std::fread(&mean, sizeof(double), 1, file) == 0) {}
        if (std::fread(&var, sizeof(double), 1, file) == 0) {}
        fclose(file);
    }

    void fit(const std::vector<double> &data) {
        mean = 0;
        var = 0;
        for (auto i: data) {
            mean += i;
        }
        mean /= double(data.size());
        for (auto i: data) {
            var += (i - mean) * (i - mean);
        }
        var /= double(data.size());
        var = std::pow(var, 0.5);
    }

    void fit(const std::vector<float> &data) {
        mean = 0;
        var = 0;
        for (auto i: data) {
            mean += i;
        }
        mean /= double(data.size());
        for (auto i: data) {
            var += (i - mean) * (i - mean);
        }
        var /= double(data.size());
        var = std::pow(var, 0.5);
    }

    [[nodiscard]] torch::Tensor forward(const torch::Tensor &data) const {
        return data.sub(float(mean)).div(float(var));
    }

    [[nodiscard]] std::vector<float> forward(std::vector<float> data) const {
        for (auto &i: data) {
            i = (i - float(mean)) / (float(var));
        }
        return data;
    }

    [[nodiscard]] std::vector<double> forward(std::vector<double> data) const {
        for (auto &i: data) {
            i = (i - mean) / var;
        }
        return data;
    }

    [[nodiscard]] std::vector<double> inverse(std::vector<double> data) const {
        for (auto &i: data) {
            i = i * var + mean;
        }
        return data;
    }

    [[nodiscard]] std::vector<float> inverse(std::vector<float> data) const {
        for (auto &i: data) {
            i = (i * float(var)) + (float(mean));
        }
        return data;
    }

    [[nodiscard]] torch::Tensor inverse(const torch::Tensor &data) const {
        return data.mul(float(var)).add(float(mean));
    }
};

class FanOutGenerator {
public:
    double min;
    double max;
    double fineness;

    FanOutGenerator(double min, double max, double fineness)
            : fineness(fineness) {
        this->min = log(min) / log(fineness);
        this->max = log(max) / log(fineness);
    }

    double get_code(double value) {
        value = log(value) / log(fineness);
        return (value - min) / (max - min);
    }


    torch::Tensor forward(torch::Tensor value) {
        value = torch::log(value) / float(std::log(fineness));
        return (value - float(min)) / float(max - min);
    }


    torch::Tensor inverse(torch::Tensor code) {
        code = code * float(max - min) + float(min);
        return torch::pow(float(fineness), code);
    }

    double get_value(double code) {
        code = code * (max - min) + min;
        return pow(fineness, code);
    }

    double random_code() {
        auto random_op = random_u_0_1();
        return pow(random_op, 1.9);
    }

    double random_value() {
        return get_value(random_code());
    }
};

class experience_t {
public:
    enum KeyType {
        FLOAT_64 = 0,
        INT_64 = 1,
        UINT_64 = 2,
        INT_32 = 3,
    };
    float distribution[BUCKET_SIZE];
    float data_size;
    float hardware_insert_score;
    float hardware_select_score;
    float hardware_erase_score;
    float hardware_clear_score;
    int fan_outs[MAX_LAYER];
    Cost cost;
    int type;

    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    to_tensor(const std::vector<experience_t> &exp_batch) {
        auto count = int(exp_batch.size());
        auto result = std::make_tuple(
                torch::rand({count, BUCKET_SIZE}),
                torch::rand({count, V_STATE_SIZE}),
                torch::rand({count, ACTION_SIZE}),
                torch::rand({count, REWARD_SIZE})
        );
        auto dis_state_ptr = get<0>(result).data_ptr<float>();
        auto v_state_ptr = get<1>(result).data_ptr<float>();
        auto action_ptr = get<2>(result).data_ptr<float>();
        auto reward_ptr = get<3>(result).data_ptr<float>();
        for (auto &i: exp_batch) {
            std::copy(i.distribution, i.distribution + BUCKET_SIZE, dis_state_ptr);
            dis_state_ptr += BUCKET_SIZE;
            v_state_ptr[0] = i.data_size;
            v_state_ptr[1] = float(i.type);
            v_state_ptr[2] = i.hardware_insert_score;
            v_state_ptr[3] = i.hardware_select_score;
            v_state_ptr[4] = i.hardware_erase_score;
            v_state_ptr[5] = i.hardware_clear_score;
            v_state_ptr += V_STATE_SIZE;
            std::copy(i.fan_outs, i.fan_outs + MAX_LAYER, action_ptr);
            action_ptr += MAX_LAYER;
            reward_ptr[0] = i.cost.memory;
            reward_ptr[1] = i.cost.add;
            reward_ptr[2] = i.cost.get;
            reward_ptr[3] = i.cost.erase;
            reward_ptr[4] = i.cost.clear;
            reward_ptr += REWARD_SIZE;
        }
        return result;
    }

    friend ostream &operator<<(ostream &out, experience_t &input)    //进来后又出去
    {
        out << "seg_size:";
        for (auto &i: input.fan_outs) {
            out << i << " ";
        }
        out << std::endl;
        out << input.cost.memory << " ";
        out << input.cost.add << " ";
        out << input.cost.get << " ";
        out << input.cost.erase << " ";
        out << input.cost.clear << " ";
        out << std::endl;
        return out;
    }
    bool operator<(const experience_t &another) const {
        for (int i = 0; i < MAX_LAYER; i++) {
            if (this->fan_outs[0] < another.fan_outs[0]) {
                return true;
            }
        }
        return false;
    }
    bool operator>(const experience_t &another) const {
        for (int i = 0; i < MAX_LAYER; i++) {
            if (this->fan_outs[0] > another.fan_outs[0]) {
                return true;
            }
        }
        return false;
    }
    bool operator==(const experience_t &another) const {
        for (int i = 0; i < MAX_LAYER; i++) {
            if (this->fan_outs[0] != another.fan_outs[0]) {
                return false;
            }
        }
        return true;
    }
};
template<class key_T,class value_T>
void StatisticsDistribution(const std::vector<std::pair<key_T, value_T>> &data_set_input,
                            double lower,
                            double upper,
                            experience_t &exp_input) {
    std::size_t test_count = data_set_input.size();
    exp_input.data_size = float(test_count);
    std::memset(exp_input.distribution,0,BUCKET_SIZE * sizeof(float));
    double slope = (double(BUCKET_SIZE - 2 * epsilon_delta)) / (upper - lower);
    double intercept = epsilon_delta - lower * slope;
    for (std::size_t i = 0; i < test_count; ++i) {
        ++exp_input.distribution[int(slope * data_set_input[i].first + intercept)];
    }
    for (float &i: exp_input.distribution) {
        i = (float) ((double) i * (double) (BUCKET_SIZE) / (double) test_count);
    }
}

std::vector<experience_t> read_precise_exp(const std::vector<std::string> &files) {
    std::vector<experience_t> all_exps;
    for (auto &i: files) {
        std::cout << "loading:" << i << std::endl;
        std::FILE *exp_file = std::fopen((precise_experience_father_path + i).c_str(), "rb");
        std::fseek(exp_file, 0, SEEK_END);
        std::size_t size = std::ftell(exp_file) / sizeof(experience_t);
        std::fseek(exp_file, 0, SEEK_SET);
        auto *result = (experience_t *) std::malloc(sizeof(experience_t) * size);
        if (std::fread(result, sizeof(experience_t), size, exp_file) == 0) {
            std::free(result);
            std::cout << "zero:" << i << std::endl;
            sleep(1);
            continue;
        }
        for (std::size_t j = 0; j < size; ++j) {
            all_exps.push_back(result[j]);
        }
        std::free(result);
    }
    return all_exps;
}


std::vector<experience_t> read_hits_exp(const std::vector<std::string> &files) {
    std::vector<experience_t> all_exps;
    for (auto &i: files) {
        std::cout << "loading:" << i << std::endl;
        std::FILE *exp_file = std::fopen((hits_experience_father_path + i).c_str(), "rb");
        std::fseek(exp_file, 0, SEEK_END);
        std::size_t size = std::ftell(exp_file) / sizeof(experience_t);
        std::fseek(exp_file, 0, SEEK_SET);
        auto *result = (experience_t *) std::malloc(sizeof(experience_t) * size);
        if (std::fread(result, sizeof(experience_t), size, exp_file) == 0) {
            std::free(result);
            std::cout << "zero:" << i << std::endl;
            sleep(1);
            continue;
        }
        for (std::size_t j = 0; j < size; ++j) {
            all_exps.push_back(result[j]);
        }
        std::free(result);
    }
    return all_exps;
}

#include <regex>
#include <PreciseIndex.hpp>
#include <TimerClock.hpp>

void Convert_hits_Experience(const std::string &filename) {
    std::FILE *exp_file = std::fopen((hits_experience_father_path + filename).c_str(), "rb");
    std::fseek(exp_file, 0, SEEK_END);
    std::size_t size = std::ftell(exp_file) / sizeof(experience_t);
    std::fseek(exp_file, 0, SEEK_SET);
    auto *result = (experience_t *) std::malloc(sizeof(experience_t) * size);
    if (std::fread(result, sizeof(experience_t), size, exp_file) == 0) {
        throw exception();
    }
    ::fclose(exp_file);
    for (std::size_t i = 0; i < size; i++) {
        result[i].hardware_insert_score = hardware_insert_score_value;
        result[i].hardware_select_score = hardware_select_score_value;
    }
    exp_file = std::fopen((hits_experience_father_path + filename + ".out").c_str(), "wb");
    if (std::fwrite(result, sizeof(experience_t), size, exp_file) == 0) {
        throw exception();
    }
    ::fclose(exp_file);
}

template<class key, class value>
void ConvertDataset(const std::string &filename) {
    std::FILE *exp_file = std::fopen((data_father_path + filename).c_str(), "rb");
    std::fseek(exp_file, 0, SEEK_END);
    std::size_t size = std::ftell(exp_file) / sizeof(key);
    std::fseek(exp_file, 0, SEEK_SET);
    auto *result = (key *) std::malloc(sizeof(key) * size);
    if (std::fread(result, sizeof(key), size, exp_file) == 0) {
        throw exception();
    }
    ::fclose(exp_file);
    exp_file = std::fopen((data_father_path + filename + "_").c_str(), "wb");
    value tmp;
    for (std::size_t i = 0; i < size; i++) {
        if (std::fwrite(&result[i], sizeof(key), 1, exp_file) == 0) {
            throw exception();
        }
        tmp = e();
        if (std::fwrite(&tmp, sizeof(value), 1, exp_file) == 0) {
            throw exception();
        }
    }
    ::fclose(exp_file);
}

template<class key_T,class value_T>
std::vector<float> standard_score(std::vector<std::pair<key_T,value_T>> dataset) {
    std::vector<float> result;
    dataset.erase(dataset.begin(), dataset.end() - 80000000);
    auto min_max = get_min_max(dataset);
    experience_t exp_gen{};
    StatisticsDistribution(dataset, min_max.first, min_max.second, exp_gen);
    std::vector<int> fan_outs;
    for (int i = 0; i < 2 * MAX_LAYER; i++) {
        fan_outs.push_back(20);
    }
    fan_outs.front() = 1024 * 1024;
    TimerClock tc;
    auto index = new PreciseIndex<key_T, value_T>(min_max.first, min_max.second, fan_outs);
    tc.synchronization();
    for (auto &i: dataset) {
        index->add(i.first, i.second);
    }
    result.push_back((float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size())));
    tc.synchronization();
    value_T value;
    for (auto &i: dataset) {
        if (!index->get(i.first, value)) {
            std::cout << "error:" << i << std::endl;
        }
    }
    result.push_back((float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size())));
    tc.synchronization();
    for (auto &i: dataset) {
        index->erase(i.first);
    }
    result.push_back((float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size())));
    tc.synchronization();
    index->clear(index->root);
    result.push_back((float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size())));
    std::cout << "score:" << result << std::endl;
    delete index;
    return result;
}

#endif //EXPERIENCE_HPP
