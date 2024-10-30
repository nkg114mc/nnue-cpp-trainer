#pragma once
#ifndef MY_LOGGER_H
#define MY_LOGGER_H

#include <fstream>
#include <string>
#include <tensorboard_logger.h>

class MyFileLogger {
private:
    std::ofstream outf;

public:
    MyFileLogger(std::string fn);
    ~MyFileLogger();
    void log_txt(std::string message, bool show_on_screen);
};


class MyTensorBoardLogger {
private:
    TensorBoardLogger *tsbd_logger;

public:
    MyTensorBoardLogger(std::string dir_fn, std::string versoin_name);
    ~MyTensorBoardLogger();
    void log_scalar(const std::string &tag, int step, float value);
};

#endif // #define MY_LOGGER_H