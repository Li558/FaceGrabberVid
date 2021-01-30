#pragma once
#include "opencv_includes.h"
#include "torch_lib_includes.h"
#include "SimpleMath.h"

const std::string haar_file_name("Resource_Depo/face/haarcascade_eye_tree_eyeglasses.xml");
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

	}
	~FaceGrabber() {}
	//相机控制
	bool StarGrab();
	void GetFrame();
	//特征识别
	bool GetFace();

	//输出调试结果
	void ShowSrc();
	void ShowDstTorch();
	void ShowROIFace();

	//存储3个部分颜色的结构体
	enum TypeIndex
	{
		BACKGROUND = 0,
		FACE = 127,
		HAIR = 254
	};

private:
	//方法

	//特征识别
	bool ObjectDetectHaar(const cv::Mat& input, std::vector< cv::Rect >& objects_rects, size_t min_target_nums);
	bool FaceDetectTorch(const cv::Mat& input);
	bool GetSegments();
	void GetGender(const cv::Mat& input);

	/////////////////////////////////////////////////后处理
	//填充闭合轮廓
	void FillContour(cv::Mat& input, cv::Mat& output);
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

	//补全光头
	void GetBaldHead(cv::Mat& input, std::vector<cv::Rect>& eyes);
	//闭操作
	void MorphologyClose(cv::Mat& img, const int& kernel_size);

	//获取脸部平均肤色值
	cv::Scalar GetSkinColor(const cv::Mat& input);

	//根据图片size放大rect
	void ZoomRect(cv::Rect& rect, const int x, const int y, cv::Size pic_size);


private:
	//分类器
	cv::CascadeClassifier haar_detector;

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
	//只有头发
	cv::Mat roi_hair_only_;
	//秃头
	cv::Mat bald_head_;

	//脸部方框
	cv::Rect rect_face_;

	//眼部roi
	std::vector<cv::Rect> objects_eyes;


	//皮肤色彩
	cv::Scalar skin_color_;
};
