﻿#pragma once
#include "opencv_includes.h"
#include "torch_lib_includes.h"

const std::string haar_file_name("Resource_Depo/face/haarcascade_frontalface_alt2.xml");
const std::string torch_file_name("Resource_Depo/face/Face_sematic_seg_model.pt");
const cv::String model_bin = "Resource_Depo/face/opencv_face_detector_uint8.pb";
const cv::String config_text = "Resource_Depo/face/opencv_face_detector.pbtxt";
const cv::String genderProto = "Resource_Depo/gender/gender_deploy.prototxt";
const cv::String genderModel = "Resource_Depo/gender/gender_net.caffemodel";


class FaceGrabber
{
public:
	inline FaceGrabber()
	{
		//读入各个模型
		if (!face_cascade_.load(haar_file_name))
		{
			std::cout << "error loading haar_file !" << std::endl;
		}
		//torch模型
		sematic_module_ = torch::jit::load(torch_file_name);
		//opencv脸模型
		face_net_ = cv::dnn::readNetFromTensorflow(model_bin, config_text);
		face_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		face_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		//性别模型
		gender_net_ = cv::dnn::readNet(genderModel, genderProto);

	}
	~FaceGrabber() {}
	bool StarGrab();
	bool FaceDetectHaar();
	bool GetFace();
	void GetFrame();

	void FaceBeautify();

	void ShowSrc();
	void ShowDstTorch();
	void ShowROIFace();

	enum TypeIndex
	{
		BACKGROUND = 0,
		FACE = 127,
		HAIR = 254
	};

private:
	//方法
	bool FaceDetectTorch(const cv::Mat& input);
	bool GetSegments();
	void GetGender(const cv::Mat& input);

	void FaceGrinding(cv::Mat& input, cv::Mat& output);//磨皮
	void AdjustSaturation(cv::Mat& input, cv::Mat& output);
	void AdjustBrightness(cv::Mat& input, cv::Mat& output);

private:
	//分类器
	cv::CascadeClassifier face_cascade_;

	torch::jit::Module sematic_module_;

	cv::dnn::Net face_net_;
	cv::dnn::Net gender_net_;
	//视频控制器
	cv::VideoCapture cap_;
	//Mat
	cv::Mat src_;
	cv::Mat dst_;
	cv::Mat dst_torch_;
	//未经过裁切
	cv::Mat roi_face_all_;
	//只有脸
	cv::Mat roi_face_only_;
	//脸加头发
	cv::Mat roi_face_hair_;

	cv::Rect rect_face_;

};