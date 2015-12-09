
#include "selfcalibrator.hpp"

#define DEBUG_SELF_CALIBRATOR 0

SelfCalibrator::SelfCalibrator() {}
SelfCalibrator::~SelfCalibrator() {}

void SelfCalibrator::Calibrate(const vector<Mat>& imgs, Matx33d& int_mat) {
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

    vector<pair<Matx33d, Matx33d>> para_proj_pairs;
    EnumerateParallelogramProjectionMat(imgs, para_proj_pairs);

    // Estimate projection of absolute conic

    // Calculate initial guess
    Size img_size(ref_img.cols, ref_img.rows);
    Matx33d initial_int_mat;
    Mat param;
    CalculateInitialIntrinsicMat(para_proj_pairs, img_size, initial_int_mat);
    // initial_int_mat = Matx33d(538, 0, 242, 0, 538, 317, 0, 0, 1);
    // initial_int_mat = Matx33d(538.9365142502786, 0, 242.412441993206, 0,
    // 538.9500430574705, 317.3260534950892, 0, 0, 1);
    // initial_int_mat = Matx33d(500, 0, 200, 0, 500, 300, 0, 0, 1);
    ProjAbsConicToParam((initial_int_mat * initial_int_mat.t()).inv(), param);

#if DEBUG_SELF_CALIBRATOR
    cout << "Para proj pairs: " << endl;
    for (auto pair : para_proj_pairs) {
        cout << "---" << endl;
        cout << pair.first << endl;
        cout << pair.second << endl;
    }
    cout << "Initial intrinsic mat: " << endl;
    cout << initial_int_mat << endl;
    cout << "Initial param: " << endl;
    cout << param << endl;
#endif

    Optimize(para_proj_pairs, param);

#if DEBUG_SELF_CALIBRATOR
    cout << "Estimated param: " << endl;
    cout << param << endl;
#endif

    ParamToIntrinsicMat(param, int_mat);
}

void SelfCalibrator::EnumerateParallelogramProjectionMat(
    const vector<Mat>& imgs,
    vector<pair<Matx33d, Matx33d>>& para_proj_pairs) const {
    // Compute first and second (N1, N2)
    vector<pair<vector<Point2f>, vector<Point2f>>> img_vertices_pairs;
    vector<Matx33d> homos;
    FindParallelogramVertices(imgs, img_vertices_pairs, homos);

    Matx33d para_proj1;
    Matx33d para_proj2;
    CalculateParallelogramProjectionMat(
        homos.at(0), img_vertices_pairs.at(0).first,
        img_vertices_pairs.at(0).second, para_proj1, para_proj2);

    // Enumerate
    auto detector = AKAZE::create();
    auto matcher = BFMatcher(NORM_HAMMING);

    para_proj_pairs.push_back(make_pair(para_proj1, para_proj2));
    vector<Matx33d> para_projs = {para_proj1, para_proj2};

    for (size_t i = 0; i < imgs.size() - 1; ++i) {
        for (size_t j = i + 1; j < imgs.size(); ++j) {
            if (i == 0 && j == 1) {
                continue;
            }

            if (para_projs.size() <= j) {
                // Calculate Nj

                // Compute homography i to j
                auto img1 = imgs.at(i);
                auto img2 = imgs.at(j);

                vector<KeyPoint> kps1;
                vector<KeyPoint> kps2;
                Mat descs1;
                Mat descs2;

                detector->detectAndCompute(img1, noArray(), kps1, descs1);
                detector->detectAndCompute(img2, noArray(), kps2, descs2);

                vector<DMatch> matches;
                matcher.match(descs1, descs2, matches);

                vector<Point2f> pts1;
                vector<Point2f> pts2;
                for (auto m : matches) {
                    pts1.push_back(kps1.at(m.queryIdx).pt);
                    pts2.push_back(kps2.at(m.trainIdx).pt);
                }

                vector<char> isinlier;
                Mat h = findHomography(pts1, pts2, RANSAC, 1, isinlier);
                if (h.empty()) {
                    throw runtime_error("Failed to compute homography\n");
                }

                vector<Point2f> inlier_pts1;
                vector<Point2f> inlier_pts2;
                for (size_t i = 0; i < isinlier.size(); ++i) {
                    if (isinlier.at(i)) {
                        inlier_pts1.push_back(pts1.at(i));
                        inlier_pts2.push_back(pts2.at(i));
                    }
                }

                // Calculate homography not normalized
                Matx33d homo;
                // homo = h;
                CalculateHomography(inlier_pts1, inlier_pts2, homo);
                auto pp = homo * para_projs.at(i);

                // Normalize
                // pp *= 1.0 / pp(2, 2);

                para_projs.push_back(pp);
            }

            para_proj_pairs.push_back(
                make_pair(para_projs.at(i), para_projs.at(j)));
        }
    }

#if DEBUG_SELF_CALIBRATOR
    // Check parallelogram projection matrix
    vector<Point2d> para_pts = {
        Point2d(-1, 1), Point2d(1, 1), Point2d(1, -1),
    };
    for (size_t i = 0; i < para_proj_pairs.size(); ++i) {
        auto para_proj_pair = para_proj_pairs.at(i);
        auto img_vertices_pair = img_vertices_pairs.at(i);

        auto gt_pts1 = img_vertices_pair.first;
        auto gt_pts2 = img_vertices_pair.second;

        vector<Point2d> calc_pts1;
        vector<Point2d> calc_pts2;

        perspectiveTransform(para_pts, calc_pts1, para_proj_pair.first);
        perspectiveTransform(para_pts, calc_pts2, para_proj_pair.second);

        cout << "--- pair" << i << endl;
        cout << "--- img1" << endl;
        cout << "para_proj: " << endl;
        cout << para_proj_pair.first << endl;
        for (auto j = 0; j < 3; ++j) {
            cout << "gt: " << gt_pts1.at(j) << "  calc: " << calc_pts1.at(j)
                 << endl;
        }
        cout << "--- img2" << endl;
        cout << "para_proj: " << endl;
        cout << para_proj_pair.second << endl;
        for (auto j = 0; j < 3; ++j) {
            cout << "gt: " << gt_pts2.at(j) << "  calc: " << calc_pts2.at(j)
                 << endl;
        }
    }
#endif
}

void SelfCalibrator::FindParallelogramVertices(
    const vector<Mat>& imgs,
    vector<pair<vector<Point2f>, vector<Point2f>>>& img_vertices_pairs,
    vector<Matx33d>& homos) const {
    auto detector = AKAZE::create();
    auto matcher = BFMatcher(NORM_HAMMING);

    // Compute triangle vertices on all images
    auto img1 = imgs.at(0);
    vector<KeyPoint> kps1;
    Mat descs1;
    detector->detectAndCompute(img1, noArray(), kps1, descs1);

    vector<Point2f> first_vertices;
    vector<vector<Point2f>> all_vertices;

    for (size_t i = 1; i < imgs.size(); ++i) {
        auto img2 = imgs.at(i);
        vector<KeyPoint> kps2;
        Mat descs2;
        detector->detectAndCompute(img2, noArray(), kps2, descs2);

        vector<DMatch> matches;
        matcher.match(descs1, descs2, matches);

        vector<Point2f> pts1;
        vector<Point2f> pts2;
        for (auto m : matches) {
            pts1.push_back(kps1.at(m.queryIdx).pt);
            pts2.push_back(kps2.at(m.trainIdx).pt);
        }

        vector<char> isinlier;
        Mat h = findHomography(pts1, pts2, RANSAC, 1, isinlier);
        if (h.empty()) {
            throw runtime_error("Failed to compute homography\n");
        }

        vector<Point2f> inlier_pts1;
        vector<Point2f> inlier_pts2;
        for (size_t i = 0; i < isinlier.size(); ++i) {
            if (isinlier.at(i)) {
                inlier_pts1.push_back(pts1.at(i));
                inlier_pts2.push_back(pts2.at(i));
            }
        }

        // Decide triangle
        if (first_vertices.empty()) {
            tuple<int, int, int> tri;
            ExtractTriangle(inlier_pts1, tri);

            first_vertices = {
                inlier_pts1.at(get<0>(tri)), inlier_pts1.at(get<1>(tri)),
                inlier_pts1.at(get<2>(tri)),
            };

            all_vertices.push_back(first_vertices);
        }

        // Calculate homography not normalized
        Matx33d homo;
        // homo = h;
        CalculateHomography(inlier_pts1, inlier_pts2, homo);

        vector<Point2f> vertices;
        perspectiveTransform(first_vertices, vertices, homo);

        all_vertices.push_back(vertices);
    }

    // Compute homography between all image each other
    img_vertices_pairs.clear();
    homos.clear();
    for (size_t i = 0; i < imgs.size() - 1; ++i) {
        for (size_t j = i + 1; j < imgs.size(); ++j) {
            auto img1 = imgs.at(i);
            auto img2 = imgs.at(j);

            vector<KeyPoint> kps1;
            vector<KeyPoint> kps2;
            Mat descs1;
            Mat descs2;

            detector->detectAndCompute(img1, noArray(), kps1, descs1);
            detector->detectAndCompute(img2, noArray(), kps2, descs2);

            vector<DMatch> matches;
            matcher.match(descs1, descs2, matches);

            vector<Point2f> pts1;
            vector<Point2f> pts2;
            for (auto m : matches) {
                pts1.push_back(kps1.at(m.queryIdx).pt);
                pts2.push_back(kps2.at(m.trainIdx).pt);
            }

            vector<char> isinlier;
            Mat h = findHomography(pts1, pts2, RANSAC, 1, isinlier);
            if (h.empty()) {
                throw runtime_error("Failed to compute homography\n");
            }

            vector<Point2f> inlier_pts1;
            vector<Point2f> inlier_pts2;
            for (size_t i = 0; i < isinlier.size(); ++i) {
                if (isinlier.at(i)) {
                    inlier_pts1.push_back(pts1.at(i));
                    inlier_pts2.push_back(pts2.at(i));
                }
            }

            // Calculate homography not normalized
            Matx33d homo;
            // homo = h;
            CalculateHomography(inlier_pts1, inlier_pts2, homo);

            homos.push_back(homo);
            img_vertices_pairs.push_back(
                make_pair(all_vertices.at(i), all_vertices.at(j)));
        }
    }

#if DEBUG_SELF_CALIBRATOR
    // Visualize triangle
    for (size_t i = 0; i < all_vertices.size(); ++i) {
        auto vs = all_vertices.at(i);
        Mat img;
        cvtColor(imgs.at(i), img, COLOR_GRAY2BGR);

        line(img, vs.at(2), vs.at(0), Scalar(0, 0, 255));
        line(img, vs.at(0), vs.at(1), Scalar(0, 0, 255));
        line(img, vs.at(1), vs.at(2), Scalar(0, 0, 255));

        putText(img, "A", vs.at(0), FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(0, 0, 255));
        putText(img, "B", vs.at(1), FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(0, 0, 255));
        putText(img, "C", vs.at(2), FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(0, 0, 255));

        stringstream title;
        title << "triangle " << i;
        namedWindow(title.str());
        imshow(title.str(), img);
        waitKey(0);
    }
#endif
}

void SelfCalibrator::ExtractTriangle(const vector<Point2f>& inlier_pts,
                                     tuple<int, int, int>& tri) const {
    /*
    auto found = false;
    // Extract largest triangle
    auto max_area = 0.0f;
    tuple<int, int, int> max_tri;
    for (auto i = 0; i < inlier_pts.size() - 2; ++i) {
        for (auto j = i + 1; j < inlier_pts.size() - 1; ++j) {
            for (auto k = j + 1; k < inlier_pts.size(); ++k) {
                auto p1 = inlier_pts.at(i);
                auto p2 = inlier_pts.at(j);
                auto p3 = inlier_pts.at(k);

                auto v1 = p1 - p2;
                auto v2 = p3 - p2;
                auto n1 = sqrt(v1.x * v1.x + v1.y * v1.y);
                auto n2 = sqrt(v2.x * v2.x + v2.y * v2.y);
                auto dot = v1.x * v2.x + v1.y * v2.y;
                auto costh = dot / (n1 * n2);
                auto th = acos(costh);
                auto sinth = sin(th);
                auto l1 = n1 / 2.0f;
                auto l2 = n2 / 2.0f;
                auto s1 = l1 - l2 * costh;
                auto s2 = l1 + l2 * costh;
                auto s3 = l2 * sinth;

                // Ignore invalid parallelogram
                if (M_PI > th && th > 0 && s1 > 0 && s2 > 0 && s3 > 0) {
                    // Check area
                    auto area = v1.x * v2.y - v1.y * v2.x;
                    if (max_area < area) {
                        max_area = area;
                        max_tri = tuple<int, int, int>(i, j, k);
                    }
                }
            }
            if (found) {
                break;
            }
        }
        if (found) {
            break;
        }
    }

    tri = max_tri;
    */

    // Extract nearest triangle
    vector<Point2f> ideal_pts = {
        Point2f(240 - 6 * 10, 320 + 6 * 10),
        Point2f(240 + 6 * 10, 320 + 6 * 10),
        Point2f(240 + 6 * 40, 320 - 6 * 10),
    };
    vector<int> tri_indices;
    for (auto i = 0; i < 3; ++i) {
        auto min_norm = DBL_MAX;
        auto min_index = -1;

        for (size_t j = 0; j < inlier_pts.size(); ++j) {
            auto n = norm(inlier_pts.at(j) - ideal_pts.at(i));
            if (min_norm > n) {
                min_norm = n;
                min_index = j;
            }
        }

        tri_indices.push_back(min_index);
    }

    tri = make_tuple(tri_indices.at(0), tri_indices.at(1), tri_indices.at(2));
}

void SelfCalibrator::CalculateHomography(const vector<Point2f>& inlier_pts1,
                                         const vector<Point2f>& inlier_pts2,
                                         Matx33d& homo) const {
    Mat A(2 * inlier_pts1.size(), 9, CV_64FC1);

    for (size_t i = 0; i < inlier_pts1.size(); ++i) {
        auto p1 = inlier_pts1.at(i);
        auto p2 = inlier_pts2.at(i);

        auto x = p1.x;
        auto y = p1.y;
        auto X = p2.x;
        auto Y = p2.y;

        double sub_A_data[] = {-x, -y, -1, 0,  0,  0,  x * X, y * X, X,
                               0,  0,  0,  -x, -y, -1, x * Y, y * Y, Y};
        Mat sub_A(2, 9, CV_64FC1, sub_A_data);
        sub_A.copyTo(A.rowRange(i * 2, (i + 1) * 2));
    }

    Mat x;
    SVD::solveZ(A, x);

    homo(0, 0) = x.at<double>(0);
    homo(0, 1) = x.at<double>(1);
    homo(0, 2) = x.at<double>(2);
    homo(1, 0) = x.at<double>(3);
    homo(1, 1) = x.at<double>(4);
    homo(1, 2) = x.at<double>(5);
    homo(2, 0) = x.at<double>(6);
    homo(2, 1) = x.at<double>(7);
    homo(2, 2) = x.at<double>(8);

    // debug
    // Normalize
    // homo *= 1.0 / homo(2, 2);
}

void SelfCalibrator::DecideParallelogramProjectedVertices(
    const vector<Point2f>& inlier_pts1, const vector<Point2f>& inlier_pts2,
    vector<Point2f>& img_vertices1, vector<Point2f>& img_vertices2) {
    // Extract random triangle
    vector<int> indices;
    for (size_t i = 0; i < inlier_pts1.size(); ++i) {
        indices.push_back(i);
    }

    random_device rd;
    mt19937 g(rd());

    tuple<int, int, int> max_tri;
    bool found = false;
    for (auto s = 0; s < 5; ++s) {
        shuffle(begin(indices), end(indices), g);

        for (size_t gi = 0; gi < indices.size() / 3; ++gi) {
            auto i = indices.at(gi * 3);
            auto j = indices.at(gi * 3 + 1);
            auto k = indices.at(gi * 3 + 2);

            auto p1 = inlier_pts1.at(i);
            auto p2 = inlier_pts1.at(j);
            auto p3 = inlier_pts1.at(k);

            auto v1 = p1 - p2;
            auto v2 = p3 - p2;
            auto n1 = sqrt(v1.x * v1.x + v1.y * v1.y);
            auto n2 = sqrt(v2.x * v2.x + v2.y * v2.y);
            auto dot = v1.x * v2.x + v1.y * v2.y;
            auto costh = dot / (n1 * n2);
            auto th = acos(costh);
            auto sinth = sin(th);
            auto l1 = n1 / 2.0f;
            auto l2 = n2 / 2.0f;
            auto s1 = l1 - l2 * costh;
            auto s2 = l1 + l2 * costh;
            auto s3 = l2 * sinth;

            // Ignore invalid parallelogram
            if (s1 > 0 && s2 > 0 && s3 > 0) {
                // Check area
                /*
                auto v1 = p2 - p1;
                auto v2 = p3 - p1;
                auto area = v1.x * v2.y - v1.y * v2.x;
                if (max_area < area) {
                    max_area = area;
                    max_tri = tuple<int, int, int>(i, j, k);
                    found = true;
                }
                */

                // Check area
                auto v1 = p2 - p1;
                auto v2 = p3 - p1;
                auto area = v1.x * v2.y - v1.y * v2.x;
                if (area > 50) {
                    max_tri = tuple<int, int, int>(i, j, k);
                    found = true;
                    break;
                }
            }

            if (found) {
                break;
            }
        }
    }

    if (!found) {
        throw runtime_error("Failed to discover triangle");
    }

    img_vertices1.push_back(inlier_pts1.at(get<0>(max_tri)));
    img_vertices1.push_back(inlier_pts1.at(get<1>(max_tri)));
    img_vertices1.push_back(inlier_pts1.at(get<2>(max_tri)));

    img_vertices2.push_back(inlier_pts2.at(get<0>(max_tri)));
    img_vertices2.push_back(inlier_pts2.at(get<1>(max_tri)));
    img_vertices2.push_back(inlier_pts2.at(get<2>(max_tri)));

    // Extract largest triangle
    /*
    auto max_area = 0.0f;
    tuple<int, int, int> max_tri;
    for (auto i = 0; i < inlier_pts1.size() - 2; ++i) {
        for (auto j = i + 1; j < inlier_pts1.size() - 1; ++j) {
            for (auto k = j + 1; k < inlier_pts1.size(); ++k) {
                auto p1 = inlier_pts1.at(i);
                auto p2 = inlier_pts1.at(j);
                auto p3 = inlier_pts1.at(k);

                // Ignore inverted parallelogram
                if (p1.x < p2.x && p1.x < p3.x && p1.y < p2.y && p1.y < p3.y &&
                    p2.y < p3.y && p3.x < p2.x) {
                    // Check area
                    auto v1 = p2 - p1;
                    auto v2 = p3 - p1;
                    auto area = v1.x * v2.y - v1.y * v2.x;
                    if (max_area < area) {
                        max_area = area;
                        max_tri = tuple<int, int, int>(i, j, k);
                    }
                }
            }
        }
    }

    img_vertices1.push_back(inlier_pts1.at(get<0>(max_tri)));
    img_vertices1.push_back(inlier_pts1.at(get<1>(max_tri)));
    img_vertices1.push_back(inlier_pts1.at(get<2>(max_tri)));

    img_vertices2.push_back(inlier_pts2.at(get<0>(max_tri)));
    img_vertices2.push_back(inlier_pts2.at(get<1>(max_tri)));
    img_vertices2.push_back(inlier_pts2.at(get<2>(max_tri)));
    */
}

void SelfCalibrator::CalculateParallelogramProjectionMat(
    const Matx33d& homo, const vector<Point2f>& img_vertices1,
    vector<Point2f>& img_vertices2, Matx33d& para_proj1,
    Matx33d& para_proj2) const {
    Mat A(4 * 3, 9, CV_64FC1);

    vector<Point2f> pts = {
        Point2f(-1, 1), Point2f(1, 1), Point2f(1, -1),
    };

    auto h11 = homo(0, 0);
    auto h12 = homo(0, 1);
    auto h13 = homo(0, 2);
    auto h21 = homo(1, 0);
    auto h22 = homo(1, 1);
    auto h23 = homo(1, 2);
    auto h31 = homo(2, 0);
    auto h32 = homo(2, 1);
    auto h33 = homo(2, 2);

    for (auto i = 0; i < 3; ++i) {
        auto x = pts.at(i).x;
        auto y = pts.at(i).y;

        auto uir = img_vertices1.at(i).x;
        auto vir = img_vertices1.at(i).y;

        auto ujr = img_vertices2.at(i).x;
        auto vjr = img_vertices2.at(i).y;

        double data_sub_A[] = {-x,
                               -y,
                               -1,
                               0,
                               0,
                               0,
                               uir * x,
                               uir * y,
                               uir,

                               0,
                               0,
                               0,
                               -x,
                               -y,
                               -1,
                               vir * x,
                               vir * y,
                               vir,

                               (h31 * ujr - h11) * x,
                               (h31 * ujr - h11) * y,
                               (h31 * ujr - h11),
                               (h32 * ujr - h12) * x,
                               (h32 * ujr - h12) * y,
                               (h32 * ujr - h12),
                               (h33 * ujr - h13) * x,
                               (h33 * ujr - h13) * y,
                               (h33 * ujr - h13),

                               (h31 * vjr - h21) * x,
                               (h31 * vjr - h21) * y,
                               (h31 * vjr - h21),
                               (h32 * vjr - h22) * x,
                               (h32 * vjr - h22) * y,
                               (h32 * vjr - h22),
                               (h33 * vjr - h23) * x,
                               (h33 * vjr - h23) * y,
                               (h33 * vjr - h23)};

        Mat sub_A(4, 9, CV_64FC1, data_sub_A);
        sub_A.copyTo(A.rowRange(i * 4, (i + 1) * 4));
    }

    Mat x;
    SVD::solveZ(A, x);

    para_proj1(0, 0) = x.at<double>(0);
    para_proj1(0, 1) = x.at<double>(1);
    para_proj1(0, 2) = x.at<double>(2);
    para_proj1(1, 0) = x.at<double>(3);
    para_proj1(1, 1) = x.at<double>(4);
    para_proj1(1, 2) = x.at<double>(5);
    para_proj1(2, 0) = x.at<double>(6);
    para_proj1(2, 1) = x.at<double>(7);
    para_proj1(2, 2) = x.at<double>(8);

    para_proj2 = homo * para_proj1;

    // debug
    // Normalize
    // para_proj1 *= 1.0f / para_proj1(2, 2);
    // para_proj2 *= 1.0f / para_proj2(2, 2);
}

void SelfCalibrator::ParamToProjAbsConic(const Mat& param,
                                         Matx33d& proj_abs_conic) const {
    /**
     * projection of the absolute conic is 5 elements
     *         [a, 0, b]
     * omega = [0, c, d]
     *         [b, d, e]
     */

    proj_abs_conic =
        Matx33d(param.at<double>(0), 0, param.at<double>(1), 0,
                param.at<double>(2), param.at<double>(3), param.at<double>(1),
                param.at<double>(3), param.at<double>(4));
}

void SelfCalibrator::ProjAbsConicToParam(const Matx33d& proj_abs_conic,
                                         Mat& param) const {
    param = Mat(5, 1, CV_64FC1);
    param.at<double>(0) = proj_abs_conic(0, 0);
    param.at<double>(1) = proj_abs_conic(0, 2);
    param.at<double>(2) = proj_abs_conic(1, 1);
    param.at<double>(3) = proj_abs_conic(1, 2);
    param.at<double>(4) = proj_abs_conic(2, 2);
}

void SelfCalibrator::ParamToIntrinsicMat(const Mat& param,
                                         Matx33d& int_mat) const {
    auto o11 = param.at<double>(0);
    auto o13 = param.at<double>(1);
    auto o22 = param.at<double>(2);
    auto o23 = param.at<double>(3);
    auto o33 = param.at<double>(4);

    auto u0 = -o13 / o11;
    auto v0 = -o23 / o22;
    auto epsilon = sqrt(o11 / o22);
    auto f = sqrt((o11 * o22 * o33 - o11 * pow(o23, 2) - o22 * pow(o13, 2)) /
                  (pow(o11, 2) * o22));

    int_mat(0, 0) = f;
    int_mat(0, 1) = 0;
    int_mat(0, 2) = u0;
    int_mat(1, 0) = 0;
    int_mat(1, 1) = epsilon * f;
    int_mat(1, 2) = v0;
    int_mat(2, 0) = 0;
    int_mat(2, 1) = 0;
    int_mat(2, 2) = 1;
}

void SelfCalibrator::CalculateError(
    const vector<pair<Matx33d, Matx33d>>& para_proj_pairs, const Mat& param,
    Mat& value) const {
    value.create(3 * para_proj_pairs.size(), 1, CV_64FC1);

    Matx33d proj_abs_conic;
    ParamToProjAbsConic(param, proj_abs_conic);

    auto i = 0;
    for (auto pair : para_proj_pairs) {
        auto para_proj1 = pair.first;
        auto para_proj2 = pair.second;

        auto nton1 = para_proj1.t() * proj_abs_conic * para_proj1;
        auto nton2 = para_proj2.t() * proj_abs_conic * para_proj2;

        auto e11 = nton1(0, 0);
        auto e21 = nton1(1, 1);
        auto e31 = nton1(0, 1);

        auto e12 = nton2(0, 0);
        auto e22 = nton2(1, 1);
        auto e32 = nton2(0, 1);

        auto f1 = e11 * e22 - e21 * e12;
        auto f2 = e31 * e12 - e11 * e32;
        auto f3 = e31 * e22 - e21 * e32;

        value.at<double>(i * 3) = f1;
        value.at<double>(i * 3 + 1) = f2;
        value.at<double>(i * 3 + 2) = f3;

        i++;
    }
}

void SelfCalibrator::CalculateJacobian(
    const vector<pair<Matx33d, Matx33d>>& para_proj_pairs, const Mat& param,
    Mat& jaco) const {
    jaco.create(3 * para_proj_pairs.size(), param.rows, CV_64F);

    auto o11 = param.at<double>(0);
    auto o13 = param.at<double>(1);
    auto o22 = param.at<double>(2);
    auto o23 = param.at<double>(3);
    auto o33 = param.at<double>(4);

    auto r = 0;
    for (auto pair : para_proj_pairs) {
        auto para_proj_i = pair.first;
        auto para_proj_j = pair.second;

        auto ni11 = para_proj_i(0, 0);
        auto ni12 = para_proj_i(0, 1);
        // auto ni13 = para_proj_i(0, 2);
        auto ni21 = para_proj_i(1, 0);
        auto ni22 = para_proj_i(1, 1);
        // auto ni23 = para_proj_i(1, 2);
        auto ni31 = para_proj_i(2, 0);
        auto ni32 = para_proj_i(2, 1);
        // auto ni33 = para_proj_i(2, 2);

        auto nj11 = para_proj_j(0, 0);
        auto nj12 = para_proj_j(0, 1);
        // auto nj13 = para_proj_j(0, 2);
        auto nj21 = para_proj_j(1, 0);
        auto nj22 = para_proj_j(1, 1);
        // auto nj23 = para_proj_j(1, 2);
        auto nj31 = para_proj_j(2, 0);
        auto nj32 = para_proj_j(2, 1);
        // auto nj33 = para_proj_j(2, 2);

        jaco.at<double>(r * 3, 0) =
            (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
             nj22 * (nj32 * o23 + nj22 * o22) +
             nj12 * (nj32 * o13 + nj12 * o11)) *
                pow(ni11, 2) +
            pow(nj12, 2) * (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                            ni11 * (o11 * ni11 + o13 * ni31) +
                            ni21 * (o22 * ni21 + o23 * ni31)) -
            (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj21 * (nj31 * o23 + nj21 * o22) +
             nj11 * (nj31 * o13 + nj11 * o11)) *
                pow(ni12, 2) -
            pow(nj11, 2) * (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                            ni12 * (o11 * ni12 + o13 * ni32) +
                            ni22 * (o22 * ni22 + o23 * ni32));

        jaco.at<double>(r * 3, 1) =
            2 * nj12 * nj32 * (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                               ni11 * (o11 * ni11 + o13 * ni31) +
                               ni21 * (o22 * ni21 + o23 * ni31)) +
            2 * (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
                 nj22 * (nj32 * o23 + nj22 * o22) +
                 nj12 * (nj32 * o13 + nj12 * o11)) *
                ni31 * ni11 -
            2 * nj11 * nj31 * (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                               ni12 * (o11 * ni12 + o13 * ni32) +
                               ni22 * (o22 * ni22 + o23 * ni32)) -
            2 * (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
                 nj21 * (nj31 * o23 + nj21 * o22) +
                 nj11 * (nj31 * o13 + nj11 * o11)) *
                ni32 * ni12;

        jaco.at<double>(r * 3, 2) =
            pow(nj22, 2) * (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                            ni11 * (o11 * ni11 + o13 * ni31) +
                            ni21 * (o22 * ni21 + o23 * ni31)) -
            pow(nj21, 2) * (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                            ni12 * (o11 * ni12 + o13 * ni32) +
                            ni22 * (o22 * ni22 + o23 * ni32)) +
            (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
             nj22 * (nj32 * o23 + nj22 * o22) +
             nj12 * (nj32 * o13 + nj12 * o11)) *
                pow(ni21, 2) -
            (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj21 * (nj31 * o23 + nj21 * o22) +
             nj11 * (nj31 * o13 + nj11 * o11)) *
                pow(ni22, 2);

        jaco.at<double>(r * 3, 3) =
            2 * nj22 * nj32 * (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                               ni11 * (o11 * ni11 + o13 * ni31) +
                               ni21 * (o22 * ni21 + o23 * ni31)) -
            2 * nj21 * nj31 * (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                               ni12 * (o11 * ni12 + o13 * ni32) +
                               ni22 * (o22 * ni22 + o23 * ni32)) +
            2 * (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
                 nj22 * (nj32 * o23 + nj22 * o22) +
                 nj12 * (nj32 * o13 + nj12 * o11)) *
                ni31 * ni21 -
            2 * (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
                 nj21 * (nj31 * o23 + nj21 * o22) +
                 nj11 * (nj31 * o13 + nj11 * o11)) *
                ni32 * ni22;

        jaco.at<double>(r * 3, 4) =
            pow(nj32, 2) * (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                            ni11 * (o11 * ni11 + o13 * ni31) +
                            ni21 * (o22 * ni21 + o23 * ni31)) -
            pow(nj31, 2) * (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                            ni12 * (o11 * ni12 + o13 * ni32) +
                            ni22 * (o22 * ni22 + o23 * ni32)) +
            (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
             nj22 * (nj32 * o23 + nj22 * o22) +
             nj12 * (nj32 * o13 + nj12 * o11)) *
                pow(ni31, 2) -
            (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj21 * (nj31 * o23 + nj21 * o22) +
             nj11 * (nj31 * o13 + nj11 * o11)) *
                pow(ni32, 2);

        jaco.at<double>(r * 3 + 1, 0) =
            (-(nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
               nj22 * (nj31 * o23 + nj21 * o22) +
               nj12 * (nj31 * o13 + nj11 * o11)) *
             pow(ni11, 2)) -
            nj11 * nj12 * (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                           ni11 * (o11 * ni11 + o13 * ni31) +
                           ni21 * (o22 * ni21 + o23 * ni31)) +
            pow(nj11, 2) * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                            ni12 * (o11 * ni11 + o13 * ni31) +
                            ni22 * (o22 * ni21 + o23 * ni31)) +
            (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj21 * (nj31 * o23 + nj21 * o22) +
             nj11 * (nj31 * o13 + nj11 * o11)) *
                ni12 * ni11;

        jaco.at<double>(r * 3 + 1, 1) =
            (-(nj11 * nj32 + nj12 * nj31) *
             (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
              ni11 * (o11 * ni11 + o13 * ni31) +
              ni21 * (o22 * ni21 + o23 * ni31))) +
            2 * nj11 * nj31 * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                               ni12 * (o11 * ni11 + o13 * ni31) +
                               ni22 * (o22 * ni21 + o23 * ni31)) +
            (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj21 * (nj31 * o23 + nj21 * o22) +
             nj11 * (nj31 * o13 + nj11 * o11)) *
                (ni32 * ni11 + ni31 * ni12) -
            2 * (nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
                 nj22 * (nj31 * o23 + nj21 * o22) +
                 nj12 * (nj31 * o13 + nj11 * o11)) *
                ni31 * ni11;

        jaco.at<double>(r * 3 + 1, 2) =
            (-nj21 * nj22 * (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                             ni11 * (o11 * ni11 + o13 * ni31) +
                             ni21 * (o22 * ni21 + o23 * ni31))) +
            pow(nj21, 2) * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                            ni12 * (o11 * ni11 + o13 * ni31) +
                            ni22 * (o22 * ni21 + o23 * ni31)) -
            (nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj22 * (nj31 * o23 + nj21 * o22) +
             nj12 * (nj31 * o13 + nj11 * o11)) *
                pow(ni21, 2) +
            (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj21 * (nj31 * o23 + nj21 * o22) +
             nj11 * (nj31 * o13 + nj11 * o11)) *
                ni22 * ni21;

        jaco.at<double>(r * 3 + 1, 3) =
            (-(nj21 * nj32 + nj22 * nj31) *
             (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
              ni11 * (o11 * ni11 + o13 * ni31) +
              ni21 * (o22 * ni21 + o23 * ni31))) +
            2 * nj21 * nj31 * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                               ni12 * (o11 * ni11 + o13 * ni31) +
                               ni22 * (o22 * ni21 + o23 * ni31)) +
            (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj21 * (nj31 * o23 + nj21 * o22) +
             nj11 * (nj31 * o13 + nj11 * o11)) *
                (ni32 * ni21 + ni31 * ni22) -
            2 * (nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
                 nj22 * (nj31 * o23 + nj21 * o22) +
                 nj12 * (nj31 * o13 + nj11 * o11)) *
                ni31 * ni21;

        jaco.at<double>(r * 3 + 1, 4) =
            (-nj31 * nj32 * (ni31 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                             ni11 * (o11 * ni11 + o13 * ni31) +
                             ni21 * (o22 * ni21 + o23 * ni31))) +
            pow(nj31, 2) * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                            ni12 * (o11 * ni11 + o13 * ni31) +
                            ni22 * (o22 * ni21 + o23 * ni31)) -
            (nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj22 * (nj31 * o23 + nj21 * o22) +
             nj12 * (nj31 * o13 + nj11 * o11)) *
                pow(ni31, 2) +
            (nj31 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj21 * (nj31 * o23 + nj21 * o22) +
             nj11 * (nj31 * o13 + nj11 * o11)) *
                ni32 * ni31;

        jaco.at<double>(r * 3 + 2, 0) =
            pow(nj12, 2) * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                            ni12 * (o11 * ni11 + o13 * ni31) +
                            ni22 * (o22 * ni21 + o23 * ni31)) +
            (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
             nj22 * (nj32 * o23 + nj22 * o22) +
             nj12 * (nj32 * o13 + nj12 * o11)) *
                ni12 * ni11 -
            (nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj22 * (nj31 * o23 + nj21 * o22) +
             nj12 * (nj31 * o13 + nj11 * o11)) *
                pow(ni12, 2) -
            nj11 * nj12 * (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                           ni12 * (o11 * ni12 + o13 * ni32) +
                           ni22 * (o22 * ni22 + o23 * ni32));

        jaco.at<double>(r * 3 + 2, 1) =
            2 * nj12 * nj32 * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                               ni12 * (o11 * ni11 + o13 * ni31) +
                               ni22 * (o22 * ni21 + o23 * ni31)) +
            (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
             nj22 * (nj32 * o23 + nj22 * o22) +
             nj12 * (nj32 * o13 + nj12 * o11)) *
                (ni32 * ni11 + ni31 * ni12) -
            (nj11 * nj32 + nj12 * nj31) *
                (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                 ni12 * (o11 * ni12 + o13 * ni32) +
                 ni22 * (o22 * ni22 + o23 * ni32)) -
            2 * (nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
                 nj22 * (nj31 * o23 + nj21 * o22) +
                 nj12 * (nj31 * o13 + nj11 * o11)) *
                ni32 * ni12;

        jaco.at<double>(r * 3 + 2, 2) =
            pow(nj22, 2) * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                            ni12 * (o11 * ni11 + o13 * ni31) +
                            ni22 * (o22 * ni21 + o23 * ni31)) -
            nj21 * nj22 * (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                           ni12 * (o11 * ni12 + o13 * ni32) +
                           ni22 * (o22 * ni22 + o23 * ni32)) +
            (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
             nj22 * (nj32 * o23 + nj22 * o22) +
             nj12 * (nj32 * o13 + nj12 * o11)) *
                ni22 * ni21 -
            (nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj22 * (nj31 * o23 + nj21 * o22) +
             nj12 * (nj31 * o13 + nj11 * o11)) *
                pow(ni22, 2);

        jaco.at<double>(r * 3 + 2, 3) =
            2 * nj22 * nj32 * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                               ni12 * (o11 * ni11 + o13 * ni31) +
                               ni22 * (o22 * ni21 + o23 * ni31)) -
            (nj21 * nj32 + nj22 * nj31) *
                (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                 ni12 * (o11 * ni12 + o13 * ni32) +
                 ni22 * (o22 * ni22 + o23 * ni32)) +
            (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
             nj22 * (nj32 * o23 + nj22 * o22) +
             nj12 * (nj32 * o13 + nj12 * o11)) *
                (ni32 * ni21 + ni31 * ni22) -
            2 * (nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
                 nj22 * (nj31 * o23 + nj21 * o22) +
                 nj12 * (nj31 * o13 + nj11 * o11)) *
                ni32 * ni22;

        jaco.at<double>(r * 3 + 2, 4) =
            pow(nj32, 2) * (ni32 * (o13 * ni11 + o23 * ni21 + o33 * ni31) +
                            ni12 * (o11 * ni11 + o13 * ni31) +
                            ni22 * (o22 * ni21 + o23 * ni31)) -
            nj31 * nj32 * (ni32 * (o13 * ni12 + o23 * ni22 + o33 * ni32) +
                           ni12 * (o11 * ni12 + o13 * ni32) +
                           ni22 * (o22 * ni22 + o23 * ni32)) +
            (nj32 * (nj32 * o33 + nj22 * o23 + nj12 * o13) +
             nj22 * (nj32 * o23 + nj22 * o22) +
             nj12 * (nj32 * o13 + nj12 * o11)) *
                ni32 * ni31 -
            (nj32 * (nj31 * o33 + nj21 * o23 + nj11 * o13) +
             nj22 * (nj31 * o23 + nj21 * o22) +
             nj12 * (nj31 * o13 + nj11 * o11)) *
                pow(ni32, 2);

        r++;
    }
}

void SelfCalibrator::CalculateInitialIntrinsicMat(
    const vector<pair<Matx33d, Matx33d>>& para_proj_pairs, const Size& img_size,
    Matx33d& int_mat) const {
    auto u0 = img_size.width / 2.0;
    auto v0 = img_size.height / 2.0;
    // auto epsilon = 1.0;

    // Compute focal length

    Mat X = Mat::zeros(3 * para_proj_pairs.size(), 2, CV_64FC1);
    Mat Z = Mat::zeros(3 * para_proj_pairs.size(), 1, CV_64FC1);

    auto r = 0;
    for (auto pair : para_proj_pairs) {
        auto para_proj1 = pair.first;
        auto para_proj2 = pair.second;

        auto ni11 = para_proj1(0, 0);
        auto ni12 = para_proj1(0, 1);
        // auto ni13 = para_proj1(0, 2);
        auto ni21 = para_proj1(1, 0);
        auto ni22 = para_proj1(1, 1);
        // auto ni23 = para_proj1(1, 2);
        auto ni31 = para_proj1(2, 0);
        auto ni32 = para_proj1(2, 1);
        // auto ni33 = para_proj1(2, 2);

        auto nj11 = para_proj2(0, 0);
        auto nj12 = para_proj2(0, 1);
        // auto nj13 = para_proj2(0, 2);
        auto nj21 = para_proj2(1, 0);
        auto nj22 = para_proj2(1, 1);
        // auto nj23 = para_proj2(1, 2);
        auto nj31 = para_proj2(2, 0);
        auto nj32 = para_proj2(2, 1);
        // auto nj33 = para_proj2(2, 2);

        // Numerators of f^4
        vector<double> A = {

            pow(v0, 2) *
                    (pow(nj32, 2) * pow(ni11, 2) - pow(nj31, 2) * pow(ni12, 2) +
                     pow(nj32, 2) * pow(ni21, 2) +
                     4 * nj22 * nj32 * ni31 * ni21 -
                     pow(nj31, 2) * pow(ni22, 2) -
                     4 * nj21 * nj31 * ni32 * ni22 +
                     pow(nj22, 2) * pow(ni31, 2) + pow(nj12, 2) * pow(ni31, 2) -
                     pow(nj21, 2) * pow(ni32, 2) -
                     pow(nj11, 2) * pow(ni32, 2)) -
                2 * v0 *
                    (nj22 * nj32 * pow(ni11, 2) - nj21 * nj31 * pow(ni12, 2) +
                     nj22 * nj32 * pow(ni21, 2) + pow(nj22, 2) * ni31 * ni21 +
                     pow(nj12, 2) * ni31 * ni21 - nj21 * nj31 * pow(ni22, 2) -
                     pow(nj21, 2) * ni32 * ni22 - pow(nj11, 2) * ni32 * ni22) -
                2 * u0 *
                    (nj12 * nj32 * pow(ni11, 2) + pow(nj22, 2) * ni31 * ni11 +
                     pow(nj12, 2) * ni31 * ni11 - nj11 * nj31 * pow(ni12, 2) -
                     pow(nj21, 2) * ni32 * ni12 - pow(nj11, 2) * ni32 * ni12 +
                     nj12 * nj32 * pow(ni21, 2) - nj11 * nj31 * pow(ni22, 2)) +
                (pow(nj22, 2) + pow(nj12, 2)) * (pow(ni11, 2) + pow(ni21, 2)) +
                pow(u0, 2) * (nj32 * (ni11 * (nj32 * ni11 + 4 * nj12 * ni31) +
                                      nj32 * pow(ni21, 2)) -
                              nj31 * (ni12 * (nj31 * ni12 + 4 * nj11 * ni32) +
                                      nj31 * pow(ni22, 2)) +
                              (pow(nj22, 2) + pow(nj12, 2)) * pow(ni31, 2) -
                              (pow(nj21, 2) + pow(nj11, 2)) * pow(ni32, 2)) -
                2 * u0 * pow(v0, 2) *
                    (nj32 * ni31 * (nj32 * ni11 + nj12 * ni31) -
                     nj31 * ni32 * (nj31 * ni12 + nj11 * ni32)) -
                2 * pow(u0, 3) * (nj32 * ni31 * (nj32 * ni11 + nj12 * ni31) -
                                  nj31 * ni32 * (nj31 * ni12 + nj11 * ni32)) +
                4 * u0 * v0 * (nj32 * ni31 * (nj22 * ni11 + nj12 * ni21) -
                               nj31 * ni32 * (nj21 * ni12 + nj11 * ni22)) -
                (pow(nj21, 2) + pow(nj11, 2)) * (pow(ni12, 2) + pow(ni22, 2)) -
                2 * pow(v0, 3) * (nj32 * ni31 * (nj32 * ni21 + nj22 * ni31) -
                                  nj31 * ni32 * (nj31 * ni22 + nj21 * ni32)) -
                2 * pow(u0, 2) * v0 *
                    (nj32 * ni31 * (nj32 * ni21 + nj22 * ni31) -
                     nj31 * ni32 * (nj31 * ni22 + nj21 * ni32)) +
                pow(v0, 4) * (nj32 * ni31 - nj31 * ni32) *
                    (nj32 * ni31 + nj31 * ni32) +
                2 * pow(u0, 2) * pow(v0, 2) * (nj32 * ni31 - nj31 * ni32) *
                    (nj32 * ni31 + nj31 * ni32) +
                pow(u0, 4) * (nj32 * ni31 - nj31 * ni32) *
                    (nj32 * ni31 + nj31 * ni32),

            pow(u0, 2) *
                    ((-nj31 * nj32 * pow(ni11, 2)) +
                     pow(nj31, 2) * ni12 * ni11 -
                     2 * nj11 * nj32 * ni31 * ni11 -
                     2 * nj12 * nj31 * ni31 * ni11 +
                     2 * nj11 * nj31 * ni32 * ni11 +
                     2 * nj11 * nj31 * ni31 * ni12 -
                     nj31 * nj32 * pow(ni21, 2) + pow(nj31, 2) * ni22 * ni21 -
                     ni31 * (nj21 * nj22 * ni31 + nj11 * nj12 * ni31 -
                             pow(nj21, 2) * ni32 - pow(nj11, 2) * ni32)) +
                pow(v0, 2) *
                    ((-nj31 * nj32 * pow(ni11, 2)) +
                     pow(nj31, 2) * ni12 * ni11 - nj31 * nj32 * pow(ni21, 2) +
                     pow(nj31, 2) * ni22 * ni21 -
                     2 * nj21 * nj32 * ni31 * ni21 -
                     2 * nj22 * nj31 * ni31 * ni21 +
                     2 * nj21 * nj31 * ni32 * ni21 +
                     2 * nj21 * nj31 * ni31 * ni22 -
                     nj21 * nj22 * pow(ni31, 2) - nj11 * nj12 * pow(ni31, 2) +
                     pow(nj21, 2) * ni32 * ni31 + pow(nj11, 2) * ni32 * ni31) +
                v0 * (nj21 * nj32 * pow(ni11, 2) + nj22 * nj31 * pow(ni11, 2) -
                      2 * nj21 * nj31 * ni12 * ni11 +
                      nj21 * nj32 * pow(ni21, 2) + nj22 * nj31 * pow(ni21, 2) -
                      2 * nj21 * nj31 * ni22 * ni21 +
                      2 * nj21 * nj22 * ni31 * ni21 +
                      2 * nj11 * nj12 * ni31 * ni21 -
                      pow(nj21, 2) * ni32 * ni21 - pow(nj11, 2) * ni32 * ni21 -
                      pow(nj21, 2) * ni31 * ni22 - pow(nj11, 2) * ni31 * ni22) +
                u0 * (nj11 * nj32 * pow(ni11, 2) + nj12 * nj31 * pow(ni11, 2) -
                      2 * nj11 * nj31 * ni12 * ni11 +
                      2 * nj21 * nj22 * ni31 * ni11 +
                      2 * nj11 * nj12 * ni31 * ni11 -
                      pow(nj21, 2) * ni32 * ni11 - pow(nj11, 2) * ni32 * ni11 -
                      pow(nj21, 2) * ni31 * ni12 - pow(nj11, 2) * ni31 * ni12 +
                      nj11 * nj32 * pow(ni21, 2) + nj12 * nj31 * pow(ni21, 2) -
                      2 * nj11 * nj31 * ni22 * ni21) -
                nj21 * nj22 * pow(ni11, 2) - nj11 * nj12 * pow(ni11, 2) +
                u0 * pow(v0, 2) *
                    (2 * nj31 * nj32 * ni31 * ni11 -
                     pow(nj31, 2) * ni32 * ni11 - pow(nj31, 2) * ni31 * ni12 +
                     nj11 * nj32 * pow(ni31, 2) + nj12 * nj31 * pow(ni31, 2) -
                     2 * nj11 * nj31 * ni32 * ni31) +
                pow(u0, 3) *
                    (2 * nj31 * nj32 * ni31 * ni11 -
                     pow(nj31, 2) * ni32 * ni11 - pow(nj31, 2) * ni31 * ni12 +
                     nj11 * nj32 * pow(ni31, 2) + nj12 * nj31 * pow(ni31, 2) -
                     2 * nj11 * nj31 * ni32 * ni31) -
                2 * u0 * v0 *
                    (nj21 * nj32 * ni31 * ni11 + nj22 * nj31 * ni31 * ni11 -
                     nj21 * nj31 * ni32 * ni11 - nj21 * nj31 * ni31 * ni12 +
                     nj11 * nj32 * ni31 * ni21 + nj12 * nj31 * ni31 * ni21 -
                     nj11 * nj31 * ni32 * ni21 - nj11 * nj31 * ni31 * ni22) +
                pow(nj21, 2) * ni12 * ni11 + pow(nj11, 2) * ni12 * ni11 -
                nj21 * nj22 * pow(ni21, 2) - nj11 * nj12 * pow(ni21, 2) +
                pow(v0, 3) *
                    (2 * nj31 * nj32 * ni31 * ni21 -
                     pow(nj31, 2) * ni32 * ni21 - pow(nj31, 2) * ni31 * ni22 +
                     nj21 * nj32 * pow(ni31, 2) + nj22 * nj31 * pow(ni31, 2) -
                     2 * nj21 * nj31 * ni32 * ni31) +
                pow(u0, 2) * v0 *
                    (2 * nj31 * nj32 * ni31 * ni21 -
                     pow(nj31, 2) * ni32 * ni21 - pow(nj31, 2) * ni31 * ni22 +
                     nj21 * nj32 * pow(ni31, 2) + nj22 * nj31 * pow(ni31, 2) -
                     2 * nj21 * nj31 * ni32 * ni31) +
                pow(nj21, 2) * ni22 * ni21 + pow(nj11, 2) * ni22 * ni21 -
                nj31 * pow(v0, 4) * ni31 * (nj32 * ni31 - nj31 * ni32) -
                2 * nj31 * pow(u0, 2) * pow(v0, 2) * ni31 *
                    (nj32 * ni31 - nj31 * ni32) -
                nj31 * pow(u0, 4) * ni31 * (nj32 * ni31 - nj31 * ni32),

            pow(u0, 2) *
                    (pow(nj32, 2) * ni12 * ni11 +
                     2 * nj12 * nj32 * ni32 * ni11 -
                     nj31 * nj32 * pow(ni12, 2) +
                     2 * nj12 * nj32 * ni31 * ni12 -
                     2 * nj11 * nj32 * ni32 * ni12 -
                     2 * nj12 * nj31 * ni32 * ni12 +
                     pow(nj32, 2) * ni22 * ni21 - nj31 * nj32 * pow(ni22, 2) +
                     ni32 * (pow(nj22, 2) * ni31 + pow(nj12, 2) * ni31 -
                             nj21 * nj22 * ni32 - nj11 * nj12 * ni32)) +
                pow(v0, 2) *
                    (pow(nj32, 2) * ni12 * ni11 - nj31 * nj32 * pow(ni12, 2) +
                     pow(nj32, 2) * ni22 * ni21 +
                     2 * nj22 * nj32 * ni32 * ni21 -
                     nj31 * nj32 * pow(ni22, 2) +
                     2 * nj22 * nj32 * ni31 * ni22 -
                     2 * nj21 * nj32 * ni32 * ni22 -
                     2 * nj22 * nj31 * ni32 * ni22 +
                     pow(nj22, 2) * ni32 * ni31 + pow(nj12, 2) * ni32 * ni31 -
                     nj21 * nj22 * pow(ni32, 2) - nj11 * nj12 * pow(ni32, 2)) +
                v0 * ((-2 * nj22 * nj32 * ni12 * ni11) +
                      nj21 * nj32 * pow(ni12, 2) + nj22 * nj31 * pow(ni12, 2) -
                      2 * nj22 * nj32 * ni22 * ni21 -
                      pow(nj22, 2) * ni32 * ni21 - pow(nj12, 2) * ni32 * ni21 +
                      nj21 * nj32 * pow(ni22, 2) + nj22 * nj31 * pow(ni22, 2) -
                      pow(nj22, 2) * ni31 * ni22 - pow(nj12, 2) * ni31 * ni22 +
                      2 * nj21 * nj22 * ni32 * ni22 +
                      2 * nj11 * nj12 * ni32 * ni22) +
                u0 * ((-2 * nj12 * nj32 * ni12 * ni11) -
                      pow(nj22, 2) * ni32 * ni11 - pow(nj12, 2) * ni32 * ni11 +
                      nj11 * nj32 * pow(ni12, 2) + nj12 * nj31 * pow(ni12, 2) -
                      pow(nj22, 2) * ni31 * ni12 - pow(nj12, 2) * ni31 * ni12 +
                      2 * nj21 * nj22 * ni32 * ni12 +
                      2 * nj11 * nj12 * ni32 * ni12 -
                      2 * nj12 * nj32 * ni22 * ni21 +
                      nj11 * nj32 * pow(ni22, 2) + nj12 * nj31 * pow(ni22, 2)) +
                u0 * pow(v0, 2) *
                    ((-pow(nj32, 2) * ni32 * ni11) -
                     pow(nj32, 2) * ni31 * ni12 +
                     2 * nj31 * nj32 * ni32 * ni12 -
                     2 * nj12 * nj32 * ni32 * ni31 +
                     nj11 * nj32 * pow(ni32, 2) + nj12 * nj31 * pow(ni32, 2)) +
                pow(u0, 3) *
                    ((-pow(nj32, 2) * ni32 * ni11) -
                     pow(nj32, 2) * ni31 * ni12 +
                     2 * nj31 * nj32 * ni32 * ni12 -
                     2 * nj12 * nj32 * ni32 * ni31 +
                     nj11 * nj32 * pow(ni32, 2) + nj12 * nj31 * pow(ni32, 2)) +
                2 * u0 * v0 *
                    (nj22 * nj32 * ni32 * ni11 + nj22 * nj32 * ni31 * ni12 -
                     nj21 * nj32 * ni32 * ni12 - nj22 * nj31 * ni32 * ni12 +
                     nj12 * nj32 * ni32 * ni21 + nj12 * nj32 * ni31 * ni22 -
                     nj11 * nj32 * ni32 * ni22 - nj12 * nj31 * ni32 * ni22) +
                pow(nj22, 2) * ni12 * ni11 + pow(nj12, 2) * ni12 * ni11 -
                nj21 * nj22 * pow(ni12, 2) - nj11 * nj12 * pow(ni12, 2) +
                pow(v0, 3) *
                    ((-pow(nj32, 2) * ni32 * ni21) -
                     pow(nj32, 2) * ni31 * ni22 +
                     2 * nj31 * nj32 * ni32 * ni22 -
                     2 * nj22 * nj32 * ni32 * ni31 +
                     nj21 * nj32 * pow(ni32, 2) + nj22 * nj31 * pow(ni32, 2)) +
                pow(u0, 2) * v0 *
                    ((-pow(nj32, 2) * ni32 * ni21) -
                     pow(nj32, 2) * ni31 * ni22 +
                     2 * nj31 * nj32 * ni32 * ni22 -
                     2 * nj22 * nj32 * ni32 * ni31 +
                     nj21 * nj32 * pow(ni32, 2) + nj22 * nj31 * pow(ni32, 2)) +
                pow(nj22, 2) * ni22 * ni21 + pow(nj12, 2) * ni22 * ni21 -
                nj21 * nj22 * pow(ni22, 2) - nj11 * nj12 * pow(ni22, 2) +
                nj32 * pow(v0, 4) * ni32 * (nj32 * ni31 - nj31 * ni32) +
                2 * nj32 * pow(u0, 2) * pow(v0, 2) * ni32 *
                    (nj32 * ni31 - nj31 * ni32) +
                nj32 * pow(u0, 4) * ni32 * (nj32 * ni31 - nj31 * ni32),

        };

        // Numerators of f^2
        vector<double> B = {
            pow(nj32, 2) * (pow(ni11, 2) + pow(ni21, 2)) -
                2 * u0 * (nj32 * ni31 * (nj32 * ni11 + nj12 * ni31) -
                          nj31 * ni32 * (nj31 * ni12 + nj11 * ni32)) -
                pow(nj31, 2) * (pow(ni12, 2) + pow(ni22, 2)) -
                2 * v0 * (nj32 * ni31 * (nj32 * ni21 + nj22 * ni31) -
                          nj31 * ni32 * (nj31 * ni22 + nj21 * ni32)) +
                (pow(nj22, 2) + pow(nj12, 2)) * pow(ni31, 2) +
                2 * pow(v0, 2) * (nj32 * ni31 - nj31 * ni32) *
                    (nj32 * ni31 + nj31 * ni32) +
                2 * pow(u0, 2) * (nj32 * ni31 - nj31 * ni32) *
                    (nj32 * ni31 + nj31 * ni32) -
                (pow(nj21, 2) + pow(nj11, 2)) * pow(ni32, 2),

            (-nj31 * (nj32 * pow(ni11, 2) - nj31 * ni12 * ni11 +
                      nj32 * pow(ni21, 2) - nj31 * ni22 * ni21)) +
                u0 * (2 * nj31 * nj32 * ni31 * ni11 -
                      pow(nj31, 2) * ni32 * ni11 - pow(nj31, 2) * ni31 * ni12 +
                      nj11 * nj32 * pow(ni31, 2) + nj12 * nj31 * pow(ni31, 2) -
                      2 * nj11 * nj31 * ni32 * ni31) +
                v0 * (2 * nj31 * nj32 * ni31 * ni21 -
                      pow(nj31, 2) * ni32 * ni21 - pow(nj31, 2) * ni31 * ni22 +
                      nj21 * nj32 * pow(ni31, 2) + nj22 * nj31 * pow(ni31, 2) -
                      2 * nj21 * nj31 * ni32 * ni31) -
                2 * nj31 * pow(v0, 2) * ni31 * (nj32 * ni31 - nj31 * ni32) -
                2 * nj31 * pow(u0, 2) * ni31 * (nj32 * ni31 - nj31 * ni32) -
                ni31 * (nj21 * nj22 * ni31 + nj11 * nj12 * ni31 -
                        pow(nj21, 2) * ni32 - pow(nj11, 2) * ni32),

            nj32 * (nj32 * ni12 * ni11 - nj31 * pow(ni12, 2) +
                    nj32 * ni22 * ni21 - nj31 * pow(ni22, 2)) +
                u0 * ((-pow(nj32, 2) * ni32 * ni11) -
                      pow(nj32, 2) * ni31 * ni12 +
                      2 * nj31 * nj32 * ni32 * ni12 -
                      2 * nj12 * nj32 * ni32 * ni31 +
                      nj11 * nj32 * pow(ni32, 2) + nj12 * nj31 * pow(ni32, 2)) +
                v0 * ((-pow(nj32, 2) * ni32 * ni21) -
                      pow(nj32, 2) * ni31 * ni22 +
                      2 * nj31 * nj32 * ni32 * ni22 -
                      2 * nj22 * nj32 * ni32 * ni31 +
                      nj21 * nj32 * pow(ni32, 2) + nj22 * nj31 * pow(ni32, 2)) +
                2 * nj32 * pow(v0, 2) * ni32 * (nj32 * ni31 - nj31 * ni32) +
                2 * nj32 * pow(u0, 2) * ni32 * (nj32 * ni31 - nj31 * ni32) +
                ni32 * (pow(nj22, 2) * ni31 + pow(nj12, 2) * ni31 -
                        nj21 * nj22 * ni32 - nj11 * nj12 * ni32),

        };

        // Constants
        vector<double> C = {
            pow(nj32, 2) * pow(ni31, 2) - pow(nj31, 2) * pow(ni32, 2),

            pow(nj31, 2) * ni32 * ni31 - nj31 * nj32 * pow(ni31, 2),

            pow(nj32, 2) * ni32 * ni31 - nj31 * nj32 * pow(ni32, 2),

        };

        // Make coeffecients matrix
        X.at<double>(3 * r, 0) = C.at(0);
        X.at<double>(3 * r, 1) = B.at(0);
        X.at<double>(3 * r + 1, 0) = C.at(1);
        X.at<double>(3 * r + 1, 1) = B.at(1);
        X.at<double>(3 * r + 2, 0) = C.at(2);
        X.at<double>(3 * r + 2, 1) = B.at(2);

        // Make constants vector
        Z.at<double>(3 * r, 0) = -A.at(0);
        Z.at<double>(3 * r + 1, 0) = -A.at(1);
        Z.at<double>(3 * r + 2, 0) = -A.at(2);

        r++;
    }

    Mat Y = (X.t() * X).inv() * X.t() * Z;

#if DEBUG_SELF_CALIBRATOR
    cout << "X:" << X << endl;
    cout << "Z:" << Z << endl;
    cout << "Y:" << Y << endl;
    cout << "f(0):" << sqrt(sqrt(Y.at<double>(0, 0))) << endl;
    cout << "f(1):" << sqrt(Y.at<double>(1, 0)) << endl;
#endif

    auto f = (sqrt(sqrt(Y.at<double>(0, 0))) + sqrt(Y.at<double>(1, 0))) / 2.0;

    int_mat(0, 0) = f;
    int_mat(0, 1) = 0;
    int_mat(0, 2) = u0;
    int_mat(1, 0) = 0;
    int_mat(1, 1) = f;
    int_mat(1, 2) = v0;
    int_mat(2, 0) = 0;
    int_mat(2, 1) = 0;
    int_mat(2, 2) = 1;
}
void SelfCalibrator::Optimize(
    const vector<pair<Matx33d, Matx33d>>& para_proj_pairs, Mat& param) const {
    auto nparams = 5;
    auto nerrs = 3 * para_proj_pairs.size();
    auto criteria =
        cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 1000, DBL_EPSILON);
    auto complete_symm_flag = false;
    auto solver = CvLevMarq(nparams, nerrs, criteria, complete_symm_flag);

    CvMat p = param;

    Mat err, jac;
    cvCopy(&p, solver.param);

    int iter = 0;
    for (;;) {
        const CvMat* _param = 0;
        CvMat* _jac = 0;
        CvMat* _err = 0;

        bool proceed = solver.update(_param, _jac, _err);

        cvCopy(_param, &p);

#if DEBUG_SELF_CALIBRATOR
        cout << "iter=" << iter << " state=" << solver.state
             << " errNorm=" << solver.errNorm << endl;
        cout << "p=" << Mat(p.rows, p.cols, CV_64FC1, p.data.db) << endl;
#endif

        if (!proceed || !_err) break;

        if (_jac) {
            CalculateJacobian(para_proj_pairs,
                              Mat(p.rows, p.cols, CV_64FC1, p.data.db), jac);
            CvMat tmp = jac;
            cvCopy(&tmp, _jac);
        }

        if (_err) {
            CalculateError(para_proj_pairs,
                           Mat(p.rows, p.cols, CV_64FC1, p.data.db), err);
            iter++;
            CvMat tmp = err;
            cvCopy(&tmp, _err);
        }
    }
}
