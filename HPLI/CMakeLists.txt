cmake_minimum_required(VERSION 3.22.1)
project(HPLI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE " -Wall  -O3 -g  -Wextra ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS " -DBoost_NO_BOOST_CMAKE=TRUE ${CMAKE_CXX_FLAGS}")


set(DEVICE_DEVICE MY_UBUNTU)
#set(DEVICE_DEVICE MY_MAC)
#set(DEVICE_DEVICE GROUP_UBUNTU)

message(DEVICE_DEVICE "   ${DEVICE_DEVICE}")
if (${DEVICE_DEVICE} STREQUAL MY_UBUNTU)
    set(ENV_NAME utils)
    set(PYTHONHOME /home/redamancyguy/anaconda3/envs/${ENV_NAME})
    set(ENV{LD_LIBRARY_PATH} /home/redamancyguy/anaconda3/pkgs/cuda-cudart-11.7.99-0/lib)
    set(data_father_path "\"/media/redamancyguy/high_speed_data/new_dataset/\"")
    set(cpu_device torch::kCUDA)
#    set(cpu_device torch::kCPU)
elseif (${DEVICE_DEVICE} STREQUAL GROUP_UBUNTU)
    set(ENV_NAME tools)
    set(PYTHONHOME /home/sun/anaconda3/envs/${ENV_NAME})
    set(ENV{LD_LIBRARY_PATH} /home/sun/anaconda3/pkgs/cuda-cudart-11.7.99-0/lib)
    set(data_father_path "\"/media/sun/data/dataset/\"")
    set(cpu_device torch::kCUDA)
elseif (${DEVICE_DEVICE} STREQUAL MY_MAC)
    set(ENV_NAME tensorflow)
    set(PYTHONHOME /Users/sunwenli/opt/anaconda3/envs/${ENV_NAME})
    set(data_father_path "\"/Users/sunwenli/Downloads/dataset-20230206/\"")
    set(cpu_device torch::kMPS)
endif ()

execute_process(COMMAND conda activate ${ENV_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(Python3_EXECUTABLE ${PYTHONHOME}/bin/python)
set(Python3_INCLUDE_DIRS ${PYTHONHOME}/include/python3.11)
set(Python3_LIBRARIES ${PYTHONHOME}/lib)

set(Python3_NumPy_INCLUDE_DIRS ${PYTHONHOME}/lib/python3.11/site-packages/numpy/core/include)
set(Python3_NumPy_LIBRARIES ${Python3_LIBRARIES})
set(CMAKE_LIBRARY_PATH ${PYTHONHOME}/lib)
set(CMAKE_INCLUDE_PATH ${PYTHONHOME}/include)

find_package(PythonInterp REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

set(CUDA_TOOLKIT_ROOT_DIR ${PYTHONHOME})
set(CUDAToolkit_ROOT ${PYTHONHOME})
set(CMAKE_CUDA_COMPILER ${PYTHONHOME}/bin/nvcc)
set(ENV{LD_LIBRARY_PATH} ${PYTHONHOME}/lib)
enable_language(CXX)
enable_language(CUDA)
FIND_PACKAGE(CUDA REQUIRED)
set(Torch_DIR ${PYTHONHOME}/lib/python3.10/site-packages/torch/share/cmake/Torch)


find_package(Torch REQUIRED)

find_package(Boost REQUIRED COMPONENTS fiber date_time system thread filesystem)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)
configure_file(
        "${PROJECT_SOURCE_DIR}/include/DEFINES.h.in"
        "${PROJECT_SOURCE_DIR}/include/DEFINES.h"
)

add_executable(main
        main.cpp
        include/experience.hpp
        include/DEFINES.h
        include/PreciseIndex.hpp
        include/PreciseSegment.hpp
        include/Models.hpp
        include/TimerClock.hpp
        include/HPLIIndex.hpp
        )
target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}")

add_executable(parallel_rebuild example/parallel_rebuild.cpp)
target_link_libraries(parallel_rebuild PRIVATE "${TORCH_LIBRARIES}" Python3::Python Python3::NumPy Boost::headers Boost::fiber Boost::system Boost::filesystem)

add_executable(transformed_index example/transformed_test.cpp)
target_link_libraries(transformed_index PRIVATE "${TORCH_LIBRARIES}" Python3::Python Python3::NumPy Boost::headers Boost::fiber Boost::system Boost::filesystem)

add_executable(data_flow benchmark/data_flow.cpp)
target_link_libraries(data_flow PUBLIC Python3::Python Python3::NumPy "${TORCH_LIBRARIES}")

add_executable(insert_select_erase_clear benchmark/insert_select_erase_clear.cpp)
target_link_libraries(insert_select_erase_clear PUBLIC Python3::Python Python3::NumPy "${TORCH_LIBRARIES}")

add_executable(fit workspace/fit_a_dataset.cpp)
target_link_libraries(fit PUBLIC Python3::Python Python3::NumPy ${TORCH_LIBRARIES})





