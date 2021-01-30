#pragma once
#include <opencv.hpp>

#define DIR_VEC_X 0
#define DIR_VEC_Y 1

struct LS
{
	cv::Point2d start;
	cv::Point2d end;
};


namespace SimpleMath
{

	////////////////////////////////////////////////////生成类函数，以Get字开头///////////////////////////////////////////////////////////////////
	std::pair<double, double> GetLineParam(cv::Point2d l1Start, cv::Point2d l1End);
	//获得交点， 不建议使用该版本
	cv::Point2d GetCrossPoint(double x1, double y1, double x2, double y2,
		double x3, double y3, double x4, double y4);
	//获得两个直线的交点，直线位置由线段确定
	cv::Point2d GetCrossPoint(cv::Point2d l1Start, cv::Point2d l1End, cv::Point2d l2Start, cv::Point2d l2End);
	//以下这个角度是从向量(x1, y1)到向量(x2, y2)之间的角度，逆时针为正，顺时针为负
	//获得两个向量之间角度的弧度制
	double GetVectorAngleRad(double x1, double y1, double x2, double y2);
	//获得两个向量之间的角度
	double GetVectorAngleDegree(double x1, double y1, double x2, double y2);

	double GetLineLen(const cv::Point2d& p1, const cv::Point2d& p2);
	//获得该直线的放大正矩形
	cv::Rect GetLineBoundingRectRatio(const LS& input, const cv::Size& imgSize, const double Ratio = 1.2);
	//获得直线中点	
	cv::Point2d GetMidpt(const cv::Point& p1, const cv::Point& p2);
	cv::Point2d GetMidpt(const LS& input);
	//生成以该点开始的,以某个角度倾斜的矩形阵列
	//旋转阵列函数
	//该函数生成一个阵列，该阵列与我们在solidwork上画图基本上没有任何区别
	//指定阵列起点， 点之间的距离（即步长）， 以及旋转的角度，指定x方向的个数，y方向的个数；
	//start: 阵列起点， rotation_rad, 旋转角度， step_len, 步长，点之间的距离
	bool GeneratePointArray(cv::Point2d start, std::vector<cv::Point2d>& output_set,
		const double rotation_rad, const double x_step_len, const double y_step_len,
		const size_t row_num = 8, const size_t col_num = 11);
	//生成以当前向量为基准的，往左旋转Rad个弧度的向量
	cv::Point2d GetRotatedVecRad(const cv::Point2d& start, cv::Point2d& end, double Rad, double expand_coff);

	/////////////////////////////////////////////////////////////判断类函数，以Is开头/////////////////////////////////////////////////////////////////////////

	bool IsSameLine(const LS& l1, const LS& l2, const double k_threshold = 20, const double b_threshold = 1000);
	//判断两直线中点距离是否接近
	bool IsMidClose(const LS& line1, const LS& line2, const double ratio = 20);
	//判断两个向量是否垂直
	bool IsRightAngle(const LS& line1, const LS& line2, const double thres);
	bool IsRightAngle(const cv::Vec4f& line1, const cv::Vec4f& line2, const double thres);
	//判断两直线角度是否接近
	bool IsSameAngle(const LS& line1, const LS& line2, const double thres);
	//判断两直线中点向量与原向量角度偏移量
	bool IsMidptdeviated(const LS& line1, const LS& line2, const double thres = 30);



}
