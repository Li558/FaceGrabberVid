﻿#pragma once
#include "opencv_includes.h"
#include "torch_lib_includes.h"
#include "SimpleMath.h"
#include "dlib_includes.h"
#include "../PS_Mixer.h"


const std::string haar_file_name("Resource_Depo/face/haarcascade_eye_tree_eyeglasses.xml");
const std::string torch_file_name("Resource_Depo/face/Face_sematic_seg_model.pt");
const cv::String model_bin = "Resource_Depo/face/opencv_face_detector_uint8.pb";
const cv::String config_text = "Resource_Depo/face/opencv_face_detector.pbtxt";
const cv::String genderProto = "Resource_Depo/gender/gender_deploy.prototxt";
const cv::String genderModel = "Resource_Depo/gender/gender_net.caffemodel";
const cv::String DlibModel = "Resource_Depo/dlib/shape_predictor_68_face_landmarks.dat";

const cv::Scalar BODY_COLOR{143, 172, 229};

class FaceGrabber
{
public:
	//默认构造函数
	inline FaceGrabber()
	{
		//读入各个模型
		if (!haar_detector.load(haar_file_name))
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
		//五官模型
		dlib::deserialize(DlibModel) >> pose_model;
		dlib_detector_ = dlib::get_frontal_face_detector();

	}
	~FaceGrabber() {}
	//相机控制
	bool StarGrab();
	void GetFrame();
	//总特征识别
	bool ProcesseFace();


	//输出调试结果
	void ShowSrc();
	void ShowDstTorch();
	void ShowROIFace();
	void ShowBaldHead();
	void ShowDebug();

	void CleanDisk();
	void WritePic2Disk();

	//存储3个部分颜色的结构体
	enum TypeIndex
	{
		BACKGROUND = 0,
		FACE = 127,
		HAIR = 254,
		LIPS = 129,
		EYES = 128
	};

private:
	//方法

	//特征识别 
	//哈尔特征识别
	bool ObjectDetectHaar(const cv::Mat& input, std::vector< cv::Rect >& objects_rects, size_t min_target_nums);
	//深度学习获取人脸区域
	bool GetFace();
	//使用torch库对人脸进行深度学习分割
	bool FaceDetectTorch(const cv::Mat& input);
	//获得嘴唇区域掩膜, 掩膜为单通道图像
	bool GetLip(const cv::Mat& input);

	//根据torch库的分割效果，对图像进行切割
	bool GetSegments();
	//识别性别
	void GetGender(const cv::Mat& ipenput);

	/////////////////////////////////////////////////后处理
	//填充闭合轮廓
	void FillContour(cv::Mat& input, cv::Mat& output, const uchar mask_val = 255);
	//去除背景
	void RemoveBackground(cv::Mat& img);
	//总美颜参数
	void FaceBeautify(cv::Mat& input, cv::Mat& output);

	/*
	dx ,fc 磨皮程度与细节程度的确定 双边滤波参数
	transparency 透明度
	*/
	void FaceGrinding(cv::Mat& input, cv::Mat& output, int value1 = 3, int value2 = 1);//磨皮
	//saturation    max_increment
	void AdjustSaturation(cv::Mat& input, cv::Mat& output, int saturation = 0, const int max_increment = 200);
	//alpha 调整对比度				beta 调整亮度
	void AdjustBrightness(cv::Mat& input, cv::Mat& output, float alpha = 1.1, float beta = 40);
	//覆盖图层函数
	void ApplyMask(const std::string& mask_type, const cv::Mat& input, const cv::Mat& mask, cv::Mat& dst);
	
	//补全光头
	void GetBaldHead(cv::Mat& input, std::vector<cv::Rect>& eyes);
	//闭操作
	void MorphologyClose(cv::Mat& img, const int& kernel_size);

	//获取脸部平均肤色值
	cv::Scalar GetSkinColor(const cv::Mat& input);

	//根据图片size放大rect
	void ZoomRect(cv::Rect& rect, const int x, const int y, cv::Size pic_size);

	//获取五官信息
	void GetFacialFeatures(cv::Mat &input,cv::Mat &output);


private:
	//分类器
	cv::CascadeClassifier haar_detector;

	torch::jit::Module sematic_module_;

	cv::dnn::Net face_net_;
	cv::dnn::Net gender_net_;

	dlib::shape_predictor pose_model;
	dlib::frontal_face_detector dlib_detector_;

	//视频控制器
	cv::VideoCapture cap_;
	//Mat
	cv::Mat src_;
	cv::Mat dst_;
	cv::Mat dst_torch_;
	cv::Mat mask_eye_;
	cv::Mat mask_lip_;

	//face_beautified
	cv::Mat face_beautified_;
	//只有脸
	cv::Mat roi_face_only_;
	//只有脸去除眼睛和嘴
	cv::Mat delect_roi_face_only_;
	//person's gender
	std::string cur_gender_;


	//未经过裁切
	cv::Mat roi_face_all_;
	//脸加头发
	cv::Mat roi_face_hair_;
	//只有头发
	cv::Mat roi_hair_only_;
	//只有嘴唇
	cv::Mat roi_lips_only_;


	//秃头
	cv::Mat bald_head_;

	//脸部方框
	cv::Rect rect_face_;

	//眼部roi
	std::vector<cv::Rect> objects_eyes;

	//皮肤色彩
	cv::Scalar skin_color_;

};
