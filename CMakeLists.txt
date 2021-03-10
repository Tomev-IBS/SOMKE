cmake_minimum_required(VERSION 3.17)
project(SOMKE)

set(CMAKE_CXX_STANDARD 17)

IF (EXISTS "k:\\Libs\\knnl\\include\\")
    set(knnl_include "k:\\Libs\\knnl\\include\\")
ENDIF()

IF(EXISTS d:/boost_1_75_0/)
    set(BOOST_ROOT d:/boost_1_75_0/)
ENDIF()
IF(EXISTS k:/libs/boost_1_75_0/)
    set(BOOST_ROOT k:/libs/boost_1_75_0/)
ENDIF()

FIND_PACKAGE(Boost)

IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()


add_executable(SOMKE include/SOMKE/SOMSequenceEntry.h src/SOMKEAlgorithm.cpp include/SOMKE/SOMKEAlgorithm.h include/SOMKE/Kernel.h)
target_include_directories(SOMKE PUBLIC ${knnl_include})
