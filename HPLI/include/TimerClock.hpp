//
// Created by 孙文礼 on 2022/12/29.

//

#ifndef TimerClock_hpp
#define TimerClock_hpp

#include <iostream>
#include <chrono>

class TimerClock {
public:
    TimerClock() {
        synchronization();
    }

    ~TimerClock() = default;

    void synchronization() {
        _ticker = std::chrono::high_resolution_clock::now();
    }

    double get_timer_second() {
        return (double) get_timer_microSec() * 0.000001;
    }

    double get_timer_milliSec() {
        return (double) get_timer_microSec() * 0.001;
    }

    long long get_timer_microSec() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - _ticker).count();
    }
    long long get_timer_nanoSec() {
        return get_timer_microSec() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _ticker;
};


#endif //TimerClock_hpp
