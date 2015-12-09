#include <opencv2/opencv.hpp>
#include <random>

#include "chessboardcalibrator.hpp"
#include "selfcalibrator.hpp"

using namespace std;
using namespace cv;

int main() {
    const auto kNumImgs = 3;

    // Load chessboard images
    vector<Mat> chessboard_imgs;
    for (auto i = 0; i < kNumImgs; ++i) {
        stringstream fpath;
        fpath << "./chessboard_imgs/" << (i + 1) << ".jpg";
        chessboard_imgs.push_back(imread(fpath.str(), IMREAD_GRAYSCALE));
    }

    // Load planar object images
    vector<Mat> plane_imgs;
    for (auto i = 0; i < kNumImgs; ++i) {
        stringstream fpath;
        fpath << "./plane_imgs/" << (i + 1) << ".jpg";
        plane_imgs.push_back(imread(fpath.str(), IMREAD_GRAYSCALE));
    }

    // Estimate intrinsic matrix by chessboard calibrator (regarded as GT)
    ChessboardCalibrator chessboard_calibrator(Size(10, 7));
    Matx33d gt_int_mat;
    chessboard_calibrator.Calibrate(chessboard_imgs, gt_int_mat);
    cout << "GT intrinsic matrix:" << endl;
    cout << gt_int_mat << endl;

    // Estimate intrinsic parameter by self calibrator
    SelfCalibrator self_calibrator;
    Matx33d int_mat;
    self_calibrator.Calibrate(plane_imgs, int_mat);
    cout << "Estimated intrinsic matrix:" << endl;
    cout << int_mat << endl;

    return 0;
}
