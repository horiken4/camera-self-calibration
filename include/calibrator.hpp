#ifndef CALIBRATOR_HPP
#define CALIBRATOR_HPP

#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

class Calibrator {
   public:
    Calibrator();
    virtual ~Calibrator();

    virtual void Calibrate(const vector<Mat>& imgs, Matx33d& int_mat) = 0;
};

#endif
