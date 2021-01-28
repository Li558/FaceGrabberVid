#pragma once
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
	//默认构造函数
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

	void FaceBeautify(cv::Mat& input, cv::Mat& output);

	void ShowSrc();
	void ShowDstTorch();
	void ShowROIFace();
	void Remove_background(cv::Mat& img);
	void MorphologyEx(cv::Mat& img);

	void CleanDisk();
	void WritePic2Disk();

	//存储3个部分颜色的结构体
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


	/*
	dx ,fc 磨皮程度与细节程度的确定 双边滤波参数
	transparency 透明度
	*/
	void FaceGrinding(cv::Mat& input, cv::Mat& output, int value1 = 3, int value2 = 1);//磨皮
	//saturation    max_increment
	void AdjustSaturation(cv::Mat& input, cv::Mat& output, int saturation = 0, const int max_increment = 200);
	//alpha 调整对比度				beta 调整亮度
	void AdjustBrightness(cv::Mat& input, cv::Mat& output, float alpha = 1.1, float beta = 40);

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

	//face_beautified
	cv::Mat face_beautified_;
	//只有脸
	cv::Mat roi_face_only_;
	//person's gender
	std::string cur_gender_;


	//未经过裁切
	cv::Mat roi_face_all_;
	//脸加头发
	cv::Mat roi_face_hair_;

	cv::Rect rect_face_;



};
