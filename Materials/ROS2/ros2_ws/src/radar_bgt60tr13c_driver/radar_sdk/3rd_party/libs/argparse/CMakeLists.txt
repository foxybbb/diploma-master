cmake_minimum_required(VERSION 3.5)

add_library(argparse STATIC argparse.c argparse.h)
target_include_directories(argparse PUBLIC .)