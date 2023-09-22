#pragma once
#ifndef MY_LOGGER_H
#define MY_LOGGER_H

#include <fstream>
#include <string>

class MyFileLogger {
private:
    std::ofstream outf;

public:
    MyFileLogger(std::string fn);
    ~MyFileLogger();
    void log_txt(std::string message, bool show_on_screen);
    void log_tensorboard();

};

#endif // #define MY_LOGGER_H