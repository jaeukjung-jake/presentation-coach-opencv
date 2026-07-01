#pragma once
#ifndef LDMARKMODEL_H_
#define LDMARKMODEL_H_

#include <iostream>
#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "cereal/cereal.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal_extension/mat_cerealisation.hpp"

#include "helper.h"
#include "feature_descriptor.h"




#define SDM_NO_ERROR        0       //�޴���
#define SDM_ERROR_FACEDET   200     //����ͨ��CascadeClassifier��⵽����
#define SDM_ERROR_FACEPOS   201     //����λ�ñ仯�ϴ󣬿���
#define SDM_ERROR_FACESIZE  202     //������С�仯�ϴ󣬿���
#define SDM_ERROR_FACENO    203     //�Ҳ�������
#define SDM_ERROR_IMAGE     204     //ͼ�����

#define SDM_ERROR_ARGS      400     //�������ݴ���
#define SDM_ERROR_MODEL     401     //ģ�ͼ��ش���



//�ع�����
class LinearRegressor{

public:
    LinearRegressor();

    bool learn(cv::Mat &data, cv::Mat &labels, bool isPCA=false);

    double test(cv::Mat data, cv::Mat labels);

    cv::Mat predict(cv::Mat values);

    void convert(std::vector<int> &tar_LandmarkIndex);
private:
    cv::Mat weights;
    cv::Mat eigenvectors;
    cv::Mat meanvalue;
    cv::Mat x;
    bool isPCA;

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template<class Archive>
    void serialize(Archive& ar)
    {
        ar(weights, meanvalue, x, isPCA);
        if(isPCA){
            ar(eigenvectors);
        }
    }
};


class ldmarkmodel{

public:
    ldmarkmodel();

    ldmarkmodel(std::vector<std::vector<int>> LandmarkIndexs, std::vector<int> eyes_index, cv::Mat meanShape, std::vector<HoGParam> HoGParams, std::vector<LinearRegressor> LinearRegressors);

    void loadFaceDetModelFile(std::string filePath = "src/haar_roboman_ff_alt2.xml");

    void train(std::vector<ImageLabel> &mImageLabels);

    cv::Mat predict(const cv::Mat& src);

    int  track(const cv::Mat& src, cv::Mat& current_shape, bool isDetFace=false);

    void printmodel();

    void convert(std::vector<int> &full_eyes_Indexs);

    cv::Mat EstimateHeadPose(cv::Mat &current_shape);

    void  EstimateHeadPose(cv::Mat &current_shape, cv::Vec3d &eav);

    void drawPose(cv::Mat& img, const cv::Mat& current_shape, float lineL=50);
private:
    cv::Rect faceBox;


    std::vector<std::vector<int>> LandmarkIndexs;
    std::vector<int> eyes_index;
    cv::Mat meanShape;
    std::vector<HoGParam> HoGParams;
    bool isNormal;
    std::vector<LinearRegressor> LinearRegressors;
    cv::CascadeClassifier face_cascade;

    cv::Mat estimateHeadPoseMat;
    cv::Mat estimateHeadPoseMat2;
    int *estimateHeadPosePointIndexs;

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template<class Archive>
    void serialize(Archive& ar)
    {
        ar(LandmarkIndexs, eyes_index, meanShape, HoGParams, isNormal, LinearRegressors);
    }
};

//����ģ��
bool load_ldmarkmodel(std::string filename, ldmarkmodel &model);

//����ģ��
void save_ldmarkmodel(ldmarkmodel model, std::string filename);



#include "ldmarkmodel.cpp"

#endif


