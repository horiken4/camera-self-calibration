#include "chessboardcalibrator.hpp"

ChessboardCalibrator::ChessboardCalibrator(const Size& chessboard_size)
    : chessboard_size_(chessboard_size) {}
ChessboardCalibrator::~ChessboardCalibrator() {}

void ChessboardCalibrator::Calibrate(const vector<Mat>& imgs,
                                     Matx33d& int_mat) {
    // Validate images

    if (imgs.size() < 3) {
        throw runtime_error("Need at least three images of a planar scene\n");
    }

    auto ref_img = imgs.at(0);
    for (auto img : imgs) {
        if (ref_img.cols != img.cols || ref_img.rows != img.rows) {
            throw runtime_error("Size of all images must be same\n");
        }
    }

    vector<vector<Point2f>> all_img_pts;
    vector<vector<Point3f>> all_obj_pts;

    auto criteria =
        TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);

    for (auto img : imgs) {
        vector<Point2f> img_pts;
        auto found = findChessboardCorners(img, chessboard_size_, img_pts);
        if (!found) {
            continue;
        }
        cornerSubPix(img, img_pts, Size(11, 11), Size(-1, -1), criteria);
        all_img_pts.push_back(img_pts);

        vector<Point3f> obj_pts;
        for (auto i = 0; i < chessboard_size_.height; ++i) {
            for (auto j = 0; j < chessboard_size_.width; ++j) {
                obj_pts.push_back(Point3f(j, i, 0));
            }
        }
        all_obj_pts.push_back(obj_pts);
    }

    Mat dist_coeff;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    calibrateCamera(all_obj_pts, all_img_pts, Size(ref_img.cols, ref_img.rows),
                    int_mat, dist_coeff, rvecs, tvecs);
}
