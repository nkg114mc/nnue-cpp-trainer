#include <iostream>
#include <sys/stat.h>
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


MyTensorBoardLogger::MyTensorBoardLogger(std::string dir_fn, std::string versoin_name) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (versoin_name == "") {
        versoin_name = "v99999";
        struct stat sb;
        for (int i = 0; i < 65536; i++) {
            versoin_name = "v" + std::to_string(i);
            std::string tmp_path = dir_fn + "/" + versoin_name;
            // Calls the function with path as argument
            // If the file/directory exists at the path returns 0
            // If block executes if path exists
            if (stat(tmp_path.c_str(), &sb) == 0 && (sb.st_mode & S_IFDIR)) {
                continue;
            } else {
                const int dir_err = mkdir(tmp_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                if (dir_err == -1){
                    std::cerr << "Error creating directory!" << std::endl;
                    exit(1);
                }
                break;
            }
        }
    }

    std::string path = dir_fn + "/" + versoin_name + "/tfevents.pb"; 
    std::cout << "Create log folder: " << (dir_fn + "/" + versoin_name) << std::endl;
    tsbd_logger = new TensorBoardLogger(path);
}

MyTensorBoardLogger::~MyTensorBoardLogger() {
    delete tsbd_logger;
    google::protobuf::ShutdownProtobufLibrary();
}

void MyTensorBoardLogger::log_scalar(const std::string &tag, int step, float value) {
    tsbd_logger->add_scalar(tag, step, value);
}
