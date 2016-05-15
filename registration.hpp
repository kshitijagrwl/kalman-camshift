#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int init2d(int ***arr, int n)
{

  int i;
  int w = 4;

  *arr = (int **)malloc(sizeof(int *)*n);
  for (i = 0; i < n; i++) {
    (*arr)[i] = (int *)malloc(sizeof(int) * w);
  }

  return 0;
}

int free2d(int ***arr, int n)
{

  int i;
  for (i = 0; i < n; i++) {
    free((*arr)[i]);
  }
  free((*arr));

  return 0;
}

// A basic symmetry test
void symmetryTest(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, std::vector<cv::DMatch> &symMatches)
{
  symMatches.clear();
  for (vector<DMatch>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1) {
    for (vector<DMatch>::const_iterator matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2) {
      if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx && (*matchIterator2).queryIdx == (*matchIterator1).trainIdx) {
        symMatches.push_back(DMatch((*matchIterator1).queryIdx, (*matchIterator1).trainIdx, (*matchIterator1).distance));
        break;
      }
    }
  }
}

int bffKnn(const cv::Mat descriptors_1, const cv::Mat descriptors_2, std::vector<cv::DMatch> &good_matches)
{

  BFMatcher matcher;
  std::vector< vector<DMatch> > matches_1, matches_2;
  matcher.knnMatch(descriptors_1, descriptors_2, matches_1, 2, noArray(), false);

  matcher.knnMatch(descriptors_2, descriptors_1, matches_2, 2, noArray(), false);

  std::vector< DMatch > filtered_matches_1, filtered_matches_2;

  // Keep matches <0.8 (Lowe's paper), discard rest
  float ratio = 0.81;
  for (int i = 0; i < matches_1.size(); ++i) {
    if (matches_1[i][0].distance < (ratio * matches_1[i][1].distance)) {
      // printf("Dist 1 - %f vs Dist 2 - %f , Ratio - %f\n", matches_1[i][0].distance,
      // matches_1[i][1].distance, matches_1[i][0].distance / matches_1[i][1].distance);
      filtered_matches_1.push_back(matches_1[i][0]);
    }
  }
  printf("-----------------------------------------------------\n");
  ratio = 0.8;
  for (int i = 0; i < matches_2.size(); ++i) {
    if (matches_2[i][0].distance < (ratio * matches_2[i][1].distance)) {
      // printf("Dist 1 - %f vs Dist 2 - %f , Ratio - %f\n", matches_2[i][0].distance,
      // matches_2[i][1].distance, matches_2[i][0].distance / matches_2[i][1].distance);
      filtered_matches_2.push_back(matches_2[i][0]);
    }
  }
  printf("Set 1 %d Set 2 %d \n", filtered_matches_1.size(), filtered_matches_2.size());
  symmetryTest(filtered_matches_1, filtered_matches_2, good_matches);
}

// Matching descriptor vectors using FLANN matcher & min distance test
int flannMatcher(const cv::Mat descriptors_1, const cv::Mat descriptors_2, std::vector<cv::DMatch> &good_matches)
{
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches, final_1, final_2;

  matcher.match(descriptors_1, descriptors_2, matches);

  // -- Quick calculation of max and min distances between keypoints
  double max_dist = 0; double min_dist = 100;
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // -- Select good matches, distance <2*min_dist,
  // -- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  // -- small

  for (int i = 0; i < descriptors_1.rows; i++) {

    if (matches[i].distance <= max(1.2 * min_dist, 0.02)) {
      final_1.push_back(matches[i]);
    }
  }

  matcher.match(descriptors_2, descriptors_1, matches);

  max_dist = 0; min_dist = 100;
  for (int i = 0; i < descriptors_2.rows; i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptors_2.rows; i++) {

    if (matches[i].distance <= max(2 * min_dist, 0.02)) {
      final_2.push_back(matches[i]);
    }
  }

  symmetryTest(final_1, final_2, good_matches);

}
//Match descriptors based on knn FLANN and Lowe's ratio test
int flannKnn(const cv::Mat descriptors_1, const cv::Mat descriptors_2, std::vector<cv::DMatch> &good_matches)
{

  FlannBasedMatcher matcher;
  std::vector< vector<DMatch> > matches_1, matches_2;
  matcher.knnMatch(descriptors_1, descriptors_2, matches_1, 2, noArray(), false);


  std::vector< DMatch > filtered_matches_1, filtered_matches_2;

  // Keep matches <0.8 (Lowe's paper), discard rest
  const float ratio = 0.8;
  for (int i = 0; i < matches_1.size(); ++i) {
    if (matches_1[i][0].distance < (ratio * matches_1[i][1].distance)) {
      // printf("Dist 1 %f vs Dist 2 %f , Ratio %f\nn", matches_1[i][0].distance,
      //        matches_1[i][1].distance, matches_1[i][0].distance / matches_1[i][1].distance);
      filtered_matches_1.push_back(matches_1[i][0]);
    }
  }

  // matcher.knnMatch(descriptors_2, descriptors_1, matches_2, 2, noArray(), false);
  // printf("--------------------------------------------------\n");
  // for (int i = 0; i < matches_2.size(); ++i) {
  //   if (matches_2[i][0].distance < (ratio * matches_2[i][1].distance)) {
  //     // printf("Dist 1 %f vs Dist 2 %f , Ratio %f\nn", matches_2[i][0].distance,
  //     // matches_2[i][1].distance, matches_2[i][0].distance / matches_2[i][1].distance);
  //     filtered_matches_2.push_back(matches_2[i][0]);
  //   }
  // }
  // printf("flanKNnn: Set 1 %d Set 2 %d \n", filtered_matches_1.size(), filtered_matches_2.size());
  good_matches = filtered_matches_1;
}


int alphablending(const cv::Mat &ref, const cv::Mat &trans)
{

  // check size is same
  cv::Mat output;
  cv::Mat tmp1, tmp2;

  if (!ref.data) {
    fprintf(stderr, "Missing ref");
    return 1;
  }

  if (!trans.data) {
    fprintf(stderr, "Missing trans");
    return 1;
  }

  double alpha = 0.3;
  double beta = 1 - alpha;

  // Color space handling
  if (ref.type() > 6 && trans.type() > 6) {

    cvtColor(ref, tmp1, CV_BGR2GRAY, 0);
    cvtColor(trans, tmp2, CV_BGR2GRAY, 0);

    cv::Mat yuv;
    cvtColor(ref, yuv, CV_BGR2YUV, 3);
    vector <Mat> channels_ref;

    split(yuv, channels_ref);
    addWeighted(tmp1, alpha, tmp2, beta, 0.0, channels_ref[0]);
    Mat merge[] = {channels_ref[0], channels_ref[1], channels_ref[2]};
    cv::merge(merge, 3, output);

    cvtColor(output, output, CV_YUV2BGR);

  }
  else {

    addWeighted(ref, alpha, trans, beta, 0.0, output);
  }

  imshow("Linear Blend", output);


  waitKey(0);

  return 0;

}
