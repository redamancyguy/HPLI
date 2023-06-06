//
// Created by 孙文礼 on 2023/1/26.
//


#include "DEFINES.h"
#include <iostream>
#include "DataSet.hpp"
#include "TimerClock.hpp"
#include <random>
#include<functional>
#include<algorithm>
#include<vector>
#include<stack>
#include "PreciseIndex.hpp"
#include "experience.hpp"
#include <unistd.h>

experience_t exp_g;
enum query_type {
    add = 0,
    get = 1,
    erase = 2,
    clear = 3,
};
struct query_bucket {
    int count[4];
};
struct Query{
    query_type type;
    std::pair<key_type,value_type> data;
};

int main() {


    TimerClock tc;
    auto dataset = dataset_source::get_dataset<std::pair<key_type, value_type>>("latitude-id-0.bin");
    std::vector<Query> queries;
    for (auto &i: dataset) {
        auto type = (query_type) (e() % 3);
        queries.push_back({type, i});
    }
    auto min_max = get_min_max(dataset);
    std::vector<std::int_fast32_t> fan_out_knob;
    for (int i = 0; i < 2 * MAX_LAYER; ++i) {
        fan_out_knob.push_back(20);
    }
    auto *index = new PreciseIndex<key_type, value_type>((std::double_t) min_max.first,
                                                         (std::double_t) min_max.second,fan_out_knob);

    unsigned int rebuild_factor = 2000000;//Adjusting speed ratio of rebuild and query,dynamic adjustment
    for (std::size_t i = 0; i < queries.size();) {
        PreciseSegment<key_type, value_type> *father = index->root;
        auto *another_index = new PreciseIndex<key_type, value_type>((std::double_t) min_max.first,
                                                                     (std::double_t) min_max.second,fan_out_knob);
        boost::context::continuation tsk;
        tsk = boost::context::callcc(
                [&](boost::context::continuation &&c) {
                    std::cout << RED << "start_rebuild" << RESET << std::endl;
                    std::stack<std::pair<PreciseSegment<key_type, value_type> *, int>> routes;//which segment , where to continue
                    routes.emplace(father, 0);
                    unsigned int step_count = 0;//Used to control the coroutine
                    while (!routes.empty()) {
                        //delete segment
                        //go on the last father node
                        father = routes.top().first;
                        int i = routes.top().second + 1;
                        routes.pop();
                        for (; i < father->_capacity; i++, step_count++) {
                            if ((*father)[i].status == Slot<key_type, value_type>::Status::Child) {
                                //if the node is a inner one ,switch to deal with the child node
                                routes.emplace(father, i);
                                father = (PreciseSegment<key_type, value_type> *) (*father)[i].workload.child;
                                i = -1;
                            } else if ((*father)[i].status == Slot<key_type, value_type>::Status::Data) {
                                another_index->add((*father)[i].workload.data.first, (*father)[i].workload.data.second);
                                //delete the moved data
                                father->de_set_data(i);
                            }
                        }
                        //delete the moved segment
                        if (!routes.empty()) {
                            (*routes.top().first).de_set_child(routes.top().second);
                        }
                        //yield to go on main coroutine
                        if (step_count > rebuild_factor) {
                            c = c.resume();
                            step_count = 0;
                        }
                    }
                    std::cout << RED << "finished_rebuild" << RESET << std::endl;
                    return std::move(c);
                });
        do {
            for (std::size_t j = 0; i + j < queries.size() && j < 1000000; j++, i++) {
                auto query = queries[i + j];
                if (query.type == query_type::add) {
                    //如果前面已经插入了
                    if (index->add(query.data.first, query.data.second)) {
                        another_index->add(query.data.first, query.data.second);
                    } else {
                        //bad add
                    }
                } else if (query.type == query_type::erase) {
                    index->erase(query.data.first);
                    another_index->erase(query.data.first);
                } else if (query.type == query_type::get) {
                    value_type value;
                    index->get(query.data.first, value);
                    another_index->get(query.data.first, value);
                }
            }
            if (tsk) {
                std::cout << i << std::endl;
                tsk = tsk.resume();
            }
        } while (tsk);

        delete (PreciseIndex<key_type, value_type> *) index;
        index = another_index;
    }


    return 0;
}