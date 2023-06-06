//
// Created by 孙文礼 on 2023/1/22.
//

#ifndef PRECISEINDEX_HPP
#define PRECISEINDEX_HPP

#include <DEFINES.h>
#include <PreciseSegment.hpp>
#include <TimerClock.hpp>
#include <thread>
#include <stack>
#include <boost/context/continuation.hpp>

template<class key_T, class value_T>
class PreciseIndex {
private:
public:
    int fan_out_knob[MAX_LAYER + 1];
    PreciseSegment<key_T, value_T> *root{};

    float memory_occupied() {
        return ((float) seg_memory_count + sizeof(PreciseIndex<key_T, value_T>)) / (1024 * 1024);
    }
    PreciseIndex(double lower, double upper, const std::vector<int> &fan_out_knob) {
        if(root == nullptr){
            for (int i = 0; i < MAX_LAYER; i++) {
                this->fan_out_knob[i] = fan_out_knob[i];
            }
            this->fan_out_knob[MAX_LAYER] = default_inner_fan_out;
            root = PreciseSegment<key_T, value_T>::new_segment(this->fan_out_knob[0], lower, upper);
        }
    }

    ~PreciseIndex() {
        if(root != nullptr){
            delete_tree(root);
            root = nullptr;
        }
    }

    void
    get_all_data(PreciseSegment<key_T, value_T> *node, std::vector<std::pair<key_T, value_T>> &dataset) {
        for (int i = 0; i < node->_capacity; ++i) {
            if ((*node)[i].status == Slot<key_T, value_T>::Status::Child) {
                auto child = (PreciseSegment<key_T, value_T> *) (*node)[i].workload.child;
                get_all_data(child, dataset);
            } else if ((*node)[i].status == Slot<key_T, value_T>::Status::Data) {
                dataset.push_back({(*node)[i].workload.data});
            }
        }
    }

    void
    rebuild_to_another(PreciseSegment<key_T, value_T> *node,
                       PreciseIndex<key_T, value_T> *another_index) {
        std::stack<std::pair<PreciseSegment<key_T, value_T> *, int>> routes;//which segment , where to continue
        routes.push({node, 0});
        while (!routes.empty()) {
            //delete segment
            //继续处理上一个父亲节点的遍历
            node = routes.top().first;
            int i = routes.top().second + 1;
            routes.pop();
            for (; i < node->_capacity; ++i) {
                if ((*node)[i].status == Slot<key_T, value_T>::Status::Child) {
                    //if the node is a inner one ,switch to deal with the child node
                    routes.emplace(node, i);
                    node = (PreciseSegment<key_T, value_T> *) (*node)[i].workload.child;
                    i = -1;
                } else if ((*node)[i].status == Slot<key_T, value_T>::Status::Data) {
                    another_index->add((*node)[i].workload.data.first, (*node)[i].workload.data.second);
                    //delete the moved data
                    node->de_set_data(i);
                }
            }
            //delete the moved segment
            if (!routes.empty()) {
                (*routes.top().first).de_set_child(routes.top().second);
            }
        }
    }

    //after the new seg_size is input into index
    void re_build(std::vector<std::pair<key_T, value_T>> &data_get) {
        this->get_all_data(this->root, data_get);
        double up_b = root->upper, lo_b = root->lower;
        delete_tree(root);
        root = PreciseSegment<key_T, value_T>::new_segment(this->fan_out_knob[0], lo_b, up_b);
        for (auto &i: data_get) {
            this->add(i.first, i.second);
        }
    }

    bool add(key_T key, const value_T &value) {
        int temp_layer = 1;
        auto pointer = root;
        auto position = (int) pointer->hypothesis(key);
        while ((*pointer)[position].status == Slot<key_T, value_T>::Status::Child) {
            pointer = ((PreciseSegment<key_T, value_T> *) (*pointer)[position].workload.child);
            ++temp_layer;
            position = (int) pointer->hypothesis(key);
        }
        while ((*pointer)[position].status == Slot<key_T, value_T>::Status::Data) {
            if (pointer->_array[position].workload.data.first == key) {
                return false;
            }
            int fan_out = this->fan_out_knob[std::min(int(MAX_LAYER),temp_layer)];
            pointer->split(position, fan_out);
            pointer = ((PreciseSegment<key_T, value_T> *) (*pointer)[position].workload.child);
            ++temp_layer;
            position = (int) pointer->hypothesis(key);
        }
        pointer->set_data(position, {key, value});
        return true;
    }

    bool get(key_T key, value_T &value) {
        auto pointer = root;
        auto position = (int) pointer->hypothesis(key);
        while ((*pointer)[position].status == Slot<key_T, value_T>::Status::Child) {
            pointer = ((PreciseSegment<key_T, value_T> *) (*pointer)[position].workload.child);
            position = (int) pointer->hypothesis(key);
        }
        if ((*pointer)[position].status == Slot<key_T, value_T>::Status::None
            || (*pointer)[position].workload.data.first != key) {
            return false;
        }
        value = (*pointer)[position].workload.data.second;
        return true;
    }

    bool erase(key_T key) {
        auto pointer = root;
        auto position = (int) pointer->hypothesis(key);
        while ((*pointer)[position].status == Slot<key_T, value_T>::Status::Child) {
            pointer = ((PreciseSegment<key_T, value_T> *) (*pointer)[position].workload.child);
            position = (int) pointer->hypothesis(key);
        }
        if ((*pointer)[position].status == Slot<key_T, value_T>::Status::None
            || (*pointer)[position].workload.data.first != key) {
            return false;
        }
        pointer->de_set_data(position);
        return true;
    }

    void clear(PreciseSegment<key_T, value_T> *node) {
        for (int i = 0; i < node->_capacity; ++i) {
            if ((*node)[i].status == Slot<key_T, value_T>::Status::Child) {
                auto child = (PreciseSegment<key_T, value_T> *) (*node)[i].workload.child;
                clear(child);
                bool is_null = true;
                for (int j = 0; j < child->_capacity; ++j) {
                    if (child->_array[j].status != Slot<key_T, value_T>::None) {
                        is_null = false;
                        break;
                    }
                }
                if (is_null) {
                    auto to_do_position = int(node->hypothesis((child->lower + child->upper) / 2));
                    node->de_set_child(to_do_position);
                }
            }
        }
    }

    void show_tree(PreciseSegment<key_T, value_T> *node, int count = 0) {
        if (!node) {
            return;
        }
        std::cout << node->lower << "<-[ : ]->" << node->upper << " _capacity: " << node->_capacity;
        for (int i = 0; i < node->_capacity; ++i) {
            if (node->_array[i].status == Slot<key_T, value_T>::Status::Data) {
                std::cout << "{position:" << i << ",key:" << node->_array[i].workload.data.first << "}, ";
            }
        }
        std::cout << std::endl;
        for (int i = 0; i < node->_capacity; ++i) {
            if (node->_array[i].status == Slot<key_T, value_T>::Status::Child) {
                for (int j = 0; j < count; ++j) {
                    std::cout << " ";
                }
                std::cout << i << " ";
                show_tree((PreciseSegment<key_T, value_T> *) node->_array[i].workload.child, count + 1);
            }
        }
    }

    void delete_tree(PreciseSegment<key_T, value_T> *node) {
        for (int i = 0; i < node->_capacity; ++i) {
            if (node->_array[i].status == Slot<key_T, value_T>::Status::Child) {
                delete_tree((PreciseSegment<key_T, value_T> *) node->_array[i].workload.child);
            }
        }
        PreciseSegment<key_T, value_T>::delete_segment(node);
    }
};


#endif //PRECISEINDEX_HPP
