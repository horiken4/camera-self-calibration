
#ifndef CHESSBOARD_CALIBRATOR_HPP
#define CHESSBOARD_CALIBRATOR_HPP

#include <opencv2/opencv.hpp>
#include <random>

#include "calibrator.hpp"

using namespace std;
using namespace cv;

class ChessboardCalibrator : public Calibrator {
   public:
    ChessboardCalibrator(const Size& chessborard_size);
    virtual ~ChessboardCalibrator();

    void Calibrate(const vector<Mat>& imgs, Matx33d& int_mat);

   private:
    Size chessboard_size_;
};

#endif
