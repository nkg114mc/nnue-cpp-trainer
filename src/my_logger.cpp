#include <iostream>
#include "my_logger.h"


MyFileLogger::MyFileLogger(std::string fn) {
    this->outf.open(fn);
}

MyFileLogger::~MyFileLogger() {
    this->outf.close();
}

void MyFileLogger::log_txt(std::string message, bool show_on_screen) {
    this->outf << message;
    this->outf.flush();
    if (show_on_screen) {
        std::cout << message;
    }
}

void MyFileLogger::log_tensorboard() {

}
