#ifndef SELF_CALIBRATOR_HPP
#define SELF_CALIBRATOR_HPP

#include <opencv2/opencv.hpp>
#include <random>

#include "calibrator.hpp"

using namespace std;
using namespace cv;

class SelfCalibrator : public Calibrator {
   public:
    SelfCalibrator();
    virtual ~SelfCalibrator();

    void Calibrate(const vector<Mat>& imgs, Matx33d& int_mat);

   private:
    void EnumerateParallelogramProjectionMat(
        const vector<Mat>& imgs,
        vector<pair<Matx33d, Matx33d>>& para_proj_pairs) const;

    void FindParallelogramVertices(
        const vector<Mat>& imgs,
        vector<pair<vector<Point2f>, vector<Point2f>>>& img_vertices_pairs,
        vector<Matx33d>& homos) const;

    void ExtractTriangle(const vector<Point2f>& inlier_pts,
                         tuple<int, int, int>& tri) const;

    void CalculateHomography(const vector<Point2f>& inlier_pts1,
                             const vector<Point2f>& inlier_pts2,
                             Matx33d& homo) const;
    void DecideParallelogramProjectedVertices(
        const vector<Point2f>& inlier_pts1, const vector<Point2f>& inlier_pts2,
        vector<Point2f>& img_vertices1, vector<Point2f>& img_vertices2);

    void CalculateParallelogramProjectionMat(
        const Matx33d& homo, const vector<Point2f>& img_vertices1,
        vector<Point2f>& img_vertices2, Matx33d& para_proj1,
        Matx33d& para_proj2) const;

    void ParamToProjAbsConic(const Mat& param, Matx33d& proj_abs_conic) const;
    void ProjAbsConicToParam(const Matx33d& proj_abs_conic, Mat& param) const;
    void ParamToIntrinsicMat(const Mat& param, Matx33d& int_mat) const;
    void CalculateError(const vector<pair<Matx33d, Matx33d>>& para_proj_pairs,
                        const Mat& param, Mat& value) const;
    void CalculateJacobian(
        const vector<pair<Matx33d, Matx33d>>& para_proj_pairs, const Mat& param,
        Mat& jaco) const;
    void CalculateInitialIntrinsicMat(
        const vector<pair<Matx33d, Matx33d>>& para_proj_pairs,
        const Size& img_size, Matx33d& int_mat) const;
    void Optimize(const vector<pair<Matx33d, Matx33d>>& para_proj_pairs,
                  Mat& param) const;
};

#endif
