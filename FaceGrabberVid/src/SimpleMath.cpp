#include "SimpleMath.h"

using namespace std;
using namespace cv;

static const double M_PI = 3.1415926535;

std::pair<double, double> SimpleMath::GetLineParam(cv::Point2d l1Start, cv::Point2d l1End)
{
	const double eps = 0.001;
	double m = 0;

	double k = 0, b = 0;// 斜率， 截距

	double x1 = l1Start.x, y1 = l1Start.y;
	double x2 = l1End.x, y2 = l1End.y;

	// 计算分子  
	m = x2 - x1;

	if (fabs(m - 0.0) < eps)
	{
		k = 10000.0;
		b = y1 - k * x1;
	}
	else
	{
		k = (y2 - y1) / (x2 - x1);
		b = y1 - k * x1;
	}


	return { k, b };
}

bool SimpleMath::IsSameLine(const LS& l1, const LS& l2, const double k_threshold, const double b_threshold)
{
	auto lineParam1 = GetLineParam(l1.start, l1.end);
	auto lineParam2 = GetLineParam(l2.start, l2.end);

	double kError = fabs(lineParam1.first - lineParam2.first);
	double bError = fabs(lineParam1.second - lineParam2.second);

	if (kError < k_threshold && bError < b_threshold)
		return true;
	else
		return false;
}

cv::Point2d SimpleMath::GetCrossPoint(double x1, double y1, double x2, double y2,
	double x3, double y3, double x4, double y4)
{
	/*定义直线参数方程如下
		x = x1 + t1*(x2 - x1);
		y = y1 + t1*(y2 - y1);

		解t为
		t1_1 = x3*(y4 - y3) + y1*(x4 - x3) - y3*(x4 - x3) - x1*(y4 - y3);
		t1_2 = (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1)
		t = t1_1 / t1_2;

		return Point2d(x, y);
	*/

	double x = 0.0, y = 0.0;
	double t1_1 = x3 * (y4 - y3) + y1 * (x4 - x3) - y3 * (x4 - x3) - x1 * (y4 - y3);
	double t1_2 = (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1);

	double t = t1_1 / t1_2;

	x = x1 + t * (x2 - x1);
	y = y1 + t * (y2 - y1);

	return cv::Point2d(x, y);

}

cv::Point2d SimpleMath::GetCrossPoint(cv::Point2d l1Start, cv::Point2d l1End, cv::Point2d l2Start, cv::Point2d l2End)
{
	return SimpleMath::GetCrossPoint(l1Start.x, l1Start.y, l1End.x, l1End.y, l2Start.x, l2Start.y, l2End.x, l2End.y);
}
//double SimpleMath::GetVectorAngle(double x1, double y1, double x2, double y2)
//{
//    /*double t = ((vector1.x * vector2.x) + (vector1.y * vector2.y)) / (sqrt(pow(vector1.x, 2) + pow(vector1.y, 2)) * sqrt(pow(vector2.x, 2) + pow(vector2.y, 2)));
//        cout << "这两个向量的夹角为:" << acos(t) * (180 / PI) << "度" << endl;*/
//    double t = ((x1 * x2) + (y1 * y2)) / (sqrt(pow(x1, 2) + pow(y1, 2)) * sqrt(pow(x2, 2) + pow(y2, 2)));
//    //t = acos(t) * (180 / M_PI);
//    t = acos(t);
//    return t;
//}

//返回向量角度(弧度制）
double SimpleMath::GetVectorAngleRad(double x1, double y1, double x2, double y2)
{
	double t = ((x1 * x2) + (y1 * y2)) / (sqrt(pow(x1, 2) + pow(y1, 2)) * sqrt(pow(x2, 2) + pow(y2, 2)));
	//t = acos(t) * (180 / M_PI);
	t = acos(t);
	return t;
}
//返回向量角度（角度制）
double SimpleMath::GetVectorAngleDegree(double x1, double y1, double x2, double y2)
{
	double result = GetVectorAngleRad(x1, y1, x2, y2) / M_PI * 180.0;
	return result;
}


double SimpleMath::GetLineLen(const cv::Point2d& p1, const cv::Point2d& p2)
{
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

cv::Rect SimpleMath::GetLineBoundingRectRatio(const LS& inputLine, const cv::Size& imgSize, const double Ratio)
{
	std::vector<cv::Point> pointVec = { inputLine.start, inputLine.end };
	//获得刚好的这个Rect
	const cv::Rect before = cv::boundingRect(pointVec);
	//计算放大后的Rect
	double midx = before.x + 0.5 * before.width;
	double midy = before.y;


	double afterWidth = before.width * Ratio;
	double afterHeight = before.height * Ratio;
	double afterX = midx - 0.5 * afterWidth;
	double afterY = midy - 0.5 * afterHeight + 0.5 * before.height;
	/*if (afterX < 0)
	{
		afterWidth - afterX;
	}
	if(afterY < 0)
	{
		afterHeight - afterY;
	}
	if (afterWidth > imgSize.width);
	{
		afterWidth = imgSize.width;
	}
	if (afterHeight > imgSize.height)
	{
		afterHeight = imgSize.height;
	}*/

	cv::Rect after(afterX, afterY, afterWidth, afterHeight);
	return after;
}

cv::Point2d SimpleMath::GetMidpt(const LS& input)
{
	return cv::Point2d((input.start.x + input.end.x) / 2, (input.start.y + input.end.y) / 2);
}

cv::Point2d SimpleMath::GetMidpt(const Point& p1, const Point& p2)
{
	return Point2d(((double)p1.x + (double)p2.x) / 2, ((double)p2.y + (double)p1.y) / 2);
}

bool SimpleMath::GeneratePointArray(cv::Point2d start, std::vector<cv::Point2d>& output_set, const double rotation_rad, const double x_step_len, const double y_step_len, const size_t row_num, const size_t col_num)
{
	//清空输出数组
	output_set.clear();
	//检测弧度范围，如果不是在-pi~pi之间，报错
	if (rotation_rad < -M_PI || rotation_rad > M_PI)
	{
		cout << "wrong rotation_rad input" << endl;
		return false;
	}
	if (row_num == 0 || col_num == 0)
	{
		cout << "row_num or col_num cant be zero!!!" << endl;
		return false;
	}
	//初始化数组

	//转换矩阵
	const size_t x_sub = 3, y_sub = 7;//转换矩阵中x, y平移的数组下标

	double transform_matrix_dat[] =
	{
		cos(rotation_rad), -sin(rotation_rad),
		sin(rotation_rad), cos(rotation_rad)
	};
	Mat transform_mat(2, 2, CV_64FC1, transform_matrix_dat);


	for (size_t row = 0; row < row_num; ++row)
	{
		for (size_t col = 0; col < col_num; ++col)
		{
			Point2d tmp(col * x_step_len, row * y_step_len);

			Mat result = transform_mat * Mat(tmp);
			result.at<double>(0) += start.x; result.at<double>(1) += start.y;

			cout << "result" << result << endl;
			output_set.emplace_back(result.at<double>(0), result.at<double>(1));
		}
	}

	return true;
}

cv::Point2d SimpleMath::GetRotatedVecRad(const cv::Point2d& start, cv::Point2d& end, double rotation_rad, double expand_coff)
{
	Point2d dir_vec((end.x - start.x), (end.y - start.y));

	if (rotation_rad < -M_PI || rotation_rad > M_PI)
	{
		cout << "wrong rotation_rad input" << endl;
		return false;
	}

	double len = GetLineLen(start, end) * expand_coff;
	double rad = tan(dir_vec.y / dir_vec.x) + rotation_rad;

	Point2d result(len * cos(rad) + start.x, len * sin(rad) + start.y);

	return result;
}


bool SimpleMath::IsMidClose(const LS& line1, const LS& line2, const double ratio)
{
	auto mid_1 = GetMidpt(line1);
	auto mid_2 = GetMidpt(line2);

	if (GetLineLen(mid_1, mid_2) < ratio)
	{
		return true;
	}
	return false;
}


bool SimpleMath::IsRightAngle(const LS& line1, const LS& line2, const double thres)
{
	Point2d relativeVec1 = line1.end - line1.start;
	Point2d relativeVec2 = line2.end - line2.start;
	double angle = GetVectorAngleDegree(relativeVec1.x, relativeVec1.y, relativeVec2.x, relativeVec2.y);

	if (fabs(angle - 90.0) < thres)
	{
		cout << "found vertical" << endl;
		return true;
	}
	return false;
}

bool SimpleMath::IsRightAngle(const Vec4f& line1, const Vec4f& line2, const double thres)
{
	Point2d relativeVec1(line1[DIR_VEC_X], line1[DIR_VEC_Y]);
	Point2d relativeVec2(line2[DIR_VEC_X], line2[DIR_VEC_Y]);
	double angle = GetVectorAngleDegree(relativeVec1.x, relativeVec1.y, relativeVec2.x, relativeVec2.y);

	if (fabs(angle - 90.0) < thres)
	{
		cout << "found vertical" << endl;
		return true;
	}
	return false;
	return false;
}

bool SimpleMath::IsSameAngle(const LS& line1, const LS& line2, const double thres)
{
	Point2d relativeVec1 = line1.end - line1.start;
	Point2d relativeVec2 = line2.end - line2.start;
	double angle = GetVectorAngleDegree(relativeVec1.x, relativeVec1.y, relativeVec2.x, relativeVec2.y);

	if (angle < 0)
		angle += 180;

	if (angle <= thres || angle >= 180 - thres)
		return true;
	return false;
}

bool SimpleMath::IsMidptdeviated(const LS& line1, const LS& line2, const double thres)
{
	//获得两条直线的中点

	const auto Mid_pt1 = GetMidpt(line1);
	const auto Mid_pt2 = GetMidpt(line2);

	const auto midVec = Mid_pt2 - Mid_pt1;
	const auto relativeVec = line1.end - line1.start;

	double angle = GetVectorAngleDegree(midVec.x, midVec.y, relativeVec.x, relativeVec.y);
	//不算方向
	if (angle < 0)
		angle += 180;

	if (angle <= thres || angle >= 180 - thres)
		return false;
	if (angle >= 80 || angle <= 100)//两直线只有与自身方向向量垂直的偏移
	{
		//计算中点距离，小的话也可以合并
		if (GetLineLen(Mid_pt1, Mid_pt2) < 20.0)
			return false;
	}
	return true;
}
