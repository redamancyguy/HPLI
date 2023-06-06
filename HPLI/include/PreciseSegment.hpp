//
// Created by 孙文礼 on 2023/1/11.
//


#ifndef PRECISE_SEGMENT_HPP
#define PRECISE_SEGMENT_HPP

#include <DEFINES.h>
#include <cstdlib>
#include <iostream>
#include <random>

long long int seg_memory_count = 0;

template<class key_T, class value_T>
class Slot {
public:
    enum Status {
        //一定是按顺序 先 None 后 Data 再 Child
        None = 0,
        Data = 1,
        Child = 2
    };

    union Workload {
        std::pair<key_T, value_T> data;
        void *child{};

        Workload() {}
    };

    Status status{Status::None};
    Workload workload;

    Slot() = default;
};


//#define SHOW_construct_TYPE
template<class key_T, class value_T>
class PreciseSegment {
public:
    double lower;
    double upper;
public:
    double slope;
    double intercept;
public:
    int _capacity;
    Slot<key_T, value_T> _array[0];

public:
    static PreciseSegment<key_T, value_T> *
    new_segment(int capacity, double lower, double upper) {
        auto result = (PreciseSegment<key_T, value_T> *) std::malloc(
                sizeof(Slot<key_T, value_T>) * capacity + sizeof(PreciseSegment<key_T, value_T>));
        result->_capacity = capacity;
        result->lower = lower;
        result->upper = upper;
        result->slope = (double(capacity) - 2 * epsilon_delta) / (upper - lower);
        result->intercept = epsilon_delta - lower * result->slope;
        for (int i = 0; i < capacity; ++i) {
            result->_array[i].status = Slot<key_T, value_T>::Status::None;
        }
        seg_memory_count += (sizeof(PreciseSegment<key_T, value_T>) +
                             capacity * sizeof(Slot<key_T, value_T>));
        return result;
    }

    static void delete_segment(PreciseSegment<key_T, value_T> *pointer) {
        seg_memory_count -= (sizeof(PreciseSegment<key_T, value_T>) +
                             pointer->_capacity * sizeof(Slot<key_T, value_T>));
        std::free(pointer);
    }

    inline double hypothesis(double key) {
        return slope * key + intercept;
    }

    inline double hypothesis_(double position) {
        return (position - intercept) / slope;
    }


    Slot<key_T, value_T> &operator[](int index) {
        return _array[index];
    }

    void set_data(const std::int32_t position, const std::pair<key_T, value_T> &data) {
        _array[position].status = Slot<key_T, value_T>::Status::Data;
        _array[position].workload.data = data;
    }

    void set_child(const std::int32_t position, void *child) {
        _array[position].status = Slot<key_T, value_T>::Status::Child;
        _array[position].workload.child = child;
    }

    void de_set_data(const std::int32_t position) {
        _array[position].status = Slot<key_T, value_T>::Status::None;
    }


    void de_set_child(const std::int32_t position) {
        _array[position].status = Slot<key_T, value_T>::Status::None;
        PreciseSegment<key_T, value_T>::delete_segment(
                (PreciseSegment<key_T, value_T> *) _array[position].workload.child);
    }

    bool is_sorted() {
        for (int i = 0; i < _capacity; ++i) {
            if (_array[i].status != Slot<key_T, value_T>::Status::None) { continue; }
            if (_array[i].first > _array[i + 1].first) {
                return false;
            }
        }
        return true;
    }

    void split(int position, int new_capacity) {
        double new_lower = hypothesis_(double(position));
        double new_upper = hypothesis_(double(position + 1));
        auto new_child = PreciseSegment<key_T, value_T>::new_segment(new_capacity, new_lower, new_upper);
        auto new_position = int(new_child->hypothesis(_array[position].workload.data.first));
        new_child->set_data(new_position, _array[position].workload.data);
        this->set_child(position, (void *) new_child);
    }
};


#endif //PRECISE_SEGMENT_HPP
