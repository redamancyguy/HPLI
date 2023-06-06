//
// Created by 孙文礼 on 2023/1/11.
//

#ifndef DEFINES_H
#define DEFINES_H

#include <random>
#include <ctime>

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

#include <iostream>

typedef double key_type;
typedef long long int value_type;

class MyException : public std::exception {
public:
    MyException() : message("Error.") {}

    explicit MyException(const std::string &str) : message("Error : " + str) {}

    ~MyException() noexcept override = default;

    [[nodiscard]] const char *what() const noexcept override {
        return message.c_str();
    }

private:
    std::string message;
};
#define epsilon_delta 0.1

std::random_device rd;
std::mt19937 e(rd());

double random_u_0_1() {
    return (double) (e() % std::numeric_limits<int>::max()) / (double) std::numeric_limits<int>::max();
}


#define FATHER_PATH "/media/redamancyguy/high_speed_data/new_dataset/"
static const std::string father_path = FATHER_PATH;
static const std::string data_father_path = father_path + "data_set/";
static const std::string model_father_path = father_path + "model/";
static const std::string precise_experience_father_path = father_path + "precise_experience/";
static const std::string hits_experience_father_path = father_path + "hits_experience/";

#include <torch/torch.h>

//#define GPU_DEVICE torch::kCUDA
#define GPU_DEVICE torch::kCUDA
#define CPU_DEVICE torch::kCPU

std::size_t BATCH_SIZE = 512;
#define window_size 100
std::size_t train_steps = 5;
std::size_t test_steps = 20;
#define train_lr 0.0000885396
#define train_wd 0.0000005


#define MAX_LAYER 8

#define REWARD_SIZE 5
#define BUCKET_SIZE (16 * 1024)
#define ACTION_SIZE MAX_LAYER
#define DIS_STATE_SIZE BUCKET_SIZE
#define V_STATE_SIZE 6

#define default_error_bound 0.2

float hardware_insert_score_value;
float hardware_select_score_value;
float hardware_erase_score_value;
float hardware_clear_score_value;
#define min_root_fan_out (256 * 1024)
#define max_root_fan_out (1024*1024*1024)
#define min_inner_fan_out 3
#define max_inner_fan_out 133
#define min_data_size (1 * 1000000)
#define max_data_size (200 * 1000000)
#define granularity 1.1
std::int_fast32_t default_inner_fan_out = 10;
std::int_fast32_t default_root_fan_out = 1024*1024;


#include <sys/types.h>
#include <dirent.h>
#include <cstdio>
#include <cerrno>
#include <iostream>
#include <regex>
std::vector<std::string> scanFiles(std::string inputDirectory) {
    std::vector<std::string> fileList;
    inputDirectory = inputDirectory.append("/");

    DIR *p_dir;
    const char *str = inputDirectory.c_str();

    p_dir = opendir(str);
    if (p_dir == nullptr) {
        std::cout << "can't open :" << inputDirectory << std::endl;
    }

    struct dirent *p_dirent;

    while ((p_dirent = readdir(p_dir))) {
        std::string tmpFileName = p_dirent->d_name;
        if (tmpFileName == "." || tmpFileName == "..") {
            continue;
        } else {
            fileList.push_back(tmpFileName);
        }
    }
    closedir(p_dir);
    return fileList;
}

std::vector<std::string> like_filter(std::vector<std::string> input, const std::string &limit) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (!regex_match(input[i], std::regex(".*" + limit + ".*"))) {
            input.erase(input.begin() + (int) i);
            --i;
        }
    }
    return input;
}

#define show_curve
#define hits_translator_wd 0.0
#endif //DEFINES_H
