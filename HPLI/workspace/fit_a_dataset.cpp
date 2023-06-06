////
//// Created by redamancyguy on 23-4-4.
////
//
//#include <cmath>
//#include <PreciseIndex.hpp>
//#include <Models.hpp>
//#include <experience.hpp>
//#include <iostream>
//#define show_curve
//
//#ifdef show_curve
//
//#include <matplotlibcpp.h>
//
//namespace plt = matplotlibcpp;
//#endif
//
//int main(){
////    FanOutGenerator root_fg(min_root_fan_out, max_root_fan_out, granularity);
////    FanOutGenerator inner_fg(min_inner_fan_out, max_inner_fan_out, granularity);
////    std::vector<std::pair<double,int>> vec;
////    for(int i  = 0;i<1000000;i++){
//////        vec.push_back({root_fg.random_value(),0});
////        vec.push_back({inner_fg.random_value(),0});
////    }
////    auto cdf = get_cdf(vec);
////    plt::plot(cdf.first,cdf.second);
////    plt::show();
////    return 0;
//    auto dataset = dataset_source::get_dataset<std::pair<key_type, value_type>>(
////            "/media/redamancyguy/high_speed_data/dataset/old_dataset/lognormal.bin1");
//            "europe-longitude-id-9.bin");
//    std::vector<std::double_t> keys;
//    std::vector<std::double_t> positions;
//    for (std::size_t i = 0; i < dataset.size(); ++i) {
//        keys.push_back(dataset[i].first);
//        positions.push_back(
//                (std::double_t(i) - std::double_t(dataset.size()) / 2) / (std::double_t(dataset.size()) / 2));
//    }
//    std::sort(keys.begin(), keys.end());
//    Standard std_x{};
//    std_x.fit(keys);
//    keys = std_x.forward(keys);
//    CDF_TRAN model(1, 1);
//    auto optimizer = torch::optim::Adam(
//            model.parameters(),
//            torch::optim::AdamOptions(0.1).weight_decay(hits_translator_wd));
////        optimizer.state().clear();
////        model.shuffle();
//    std::vector<std::double_t> x_vec;
//    std::vector<std::double_t> y_vec;
//
//#ifdef show_curve
//    plt::ion();
//#endif
//    std::size_t batch_size = 10000;
//    for (int _ = 0; _ < 1333; ++_) {
//        for (std::size_t j = 0; j < batch_size; ++j) {
//            auto random_index = e() % dataset.size();
//            x_vec.push_back(keys[random_index]);
//            y_vec.push_back(positions[random_index]);
//        }
//        auto x_batch = torch::tensor(x_vec, torch::TensorOptions().dtype(torch::kDouble)).view({-1, 1});
//        auto y_batch = torch::tensor(y_vec, torch::TensorOptions().dtype(torch::kDouble)).view({-1, 1});
//        auto pred = model.forward(x_batch);
//        auto loss = (pred - x_batch);
//        loss = torch::norm(loss,0.5);
//        optimizer.zero_grad();
//        loss.backward();
//        optimizer.step();
//        if (_ > 100) {
//            optimizer.param_groups()[0].options().set_lr(
//                    optimizer.param_groups()[0].options().get_lr() * 0.7);
//        }
//#ifdef show_curve
//        plt::cla();
//            std::vector<std::double_t> pred_vec(pred.data_ptr<std::double_t>(),
//                                                pred.data_ptr<std::double_t>() + pred.numel());
//            std::sort(x_vec.begin(), x_vec.end());
//            std::sort(y_vec.begin(), y_vec.end());
//            std::sort(pred_vec.begin(), pred_vec.end());
//            std::map<std::string, std::string> color;
//            color["color"] = "orange";
//            color["label"] = "y_vec";
//            plt::scatter(x_vec, y_vec, 1, color);
//            color["color"] = "purple";
//            color["label"] = "pred_vec";
//            plt::plot(x_vec, pred_vec, color);
//            plt::pause(0.00001);
//            ofstream f("data.txt", ios::app);
//#endif
//        x_vec.clear();
//        y_vec.clear();
//    }
//    plt::close();
//    return 0;
//}


//
// Created by redamancyguy on 23-4-4.
//

#include <cmath>
#include <PreciseIndex.hpp>
#include <Models.hpp>
#include <experience.hpp>
#include <iostream>
#include <HitsIndex.hpp>

#define show_curve

#ifdef show_curve

#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;
#endif

int main() {
    auto dataset = dataset_source::get_dataset<std::pair<key_type, value_type>>(
            "europe-longitude-id-9.bin");
    Standard std1;
    Standard std2;
    Standard std3;
    auto fanout = std::vector<int>({int(dataset.size()), 10, 10, 10, 10, 10, 10, 10});
    auto index = HitsIndex<key_type, value_type>(dataset, fanout);
    std::vector<std::pair<double,long long int>>tran_keys;
    std::pair<std::vector<double>,std::vector<double>>tran_model;
    auto min_max = get_min_max(dataset);
    double step = (min_max.second - min_max.first)/dataset.size();
    double this_step = min_max.first;
    for(int i = 0;i<dataset.size();i++){
        tran_model.first.push_back(this_step);
        tran_model.second.push_back(index.tran_full(this_step));
        this_step += step;
    }
    for(auto i:dataset){
        tran_keys.push_back({index.tran_full(i.first),i.second});
    }
    auto data_cdf = get_cdf(dataset);
    auto cdf_tran = get_cdf(tran_keys);
    std1.fit(tran_model.second);
    tran_model.second = std1.forward(tran_model.second);
    std2.fit(cdf_tran.second);
    cdf_tran.second = std2.forward(cdf_tran.second);
    std3.fit(data_cdf.second);
    data_cdf.second = std3.forward(data_cdf.second);
    std::unordered_map<std::string,std::string> color;
    color["color"] = "red";
    color["label"] = "data_cdf";
    plt::plot(data_cdf.first,data_cdf.second);
    color["color"] = "red";
    color["label"] = "cdf_tran";
    plt::plot(cdf_tran.first,cdf_tran.second);
    color["color"] = "red";
    color["label"] = "tran_model";
    plt::plot(tran_model.first,tran_model.second);
    plt::legend();
    plt::show();
    ofstream f1("data_cdf.txt");
    ofstream f2("cdf_tran.txt");
    ofstream f3("tran_model.txt");
    for(auto i:data_cdf.first){
        f1 << i <<",";
    }
    f1 <<"\n";
    for(auto i:data_cdf.second){
        f1 << i <<",";
    }
    /////////
    for(auto i:cdf_tran.first){
        f2 << i <<",";
    }
    f2 <<"\n";
    for(auto i:cdf_tran.second){
        f2 << i <<",";
    }
    ///////
    for(auto i:tran_model.first){
        f3 << i <<",";
    }
    f3 <<"\n";
    for(auto i:tran_model.second){
        f3 << i <<",";
    }
    return 0;
}