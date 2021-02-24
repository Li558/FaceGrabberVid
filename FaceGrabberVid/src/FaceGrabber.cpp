#include "FaceGrabber.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;



//打开相机
bool FaceGrabber::StarGrab()
{

	cap_.open(0, CAP_DSHOW);

	if (!cap_.isOpened())
	{
		cap_.open(1, CAP_DSHOW);
		if (!cap_.isOpened())
		{
			cout << "error cam open failed" << endl;
			return false;
		}
	}

	return true;

}

//读入一帧帧图片
void FaceGrabber::GetFrame()
{
	cap_ >> src_;
}


//得到人脸, 总处理函数
bool FaceGrabber::GetFace()
{
	//src判空
	bool face_dectected = false;
	if (src_.empty())
		return false;

	//整体像素值减去平均值（mean）通过缩放系数（scalefactor）对图片像素值进行缩放
	cv::Mat blob_image = blobFromImage(src_, 1.0,
		cv::Size(300, 300),
		cv::Scalar(104.0, 177.0, 123.0), false, false);

	face_net_.setInput(blob_image, "data");
	cv::Mat detection = face_net_.forward("detection_out");

	const int x_padding = 40;
	const int y_padding = 80;
	cv::Mat detection_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	//阈值为0.5 超过0.5才会显示
	float confidence_threshold = 0.5;
	for (int i = 0; i < detection_mat.rows; i++) {
		float confidence = detection_mat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detection_mat.at<float>(i, 1));
			float tl_x = detection_mat.at<float>(i, 3) * src_.cols;
			float tl_y = detection_mat.at<float>(i, 4) * src_.rows;
			float br_x = detection_mat.at<float>(i, 5) * src_.cols;
			float br_y = detection_mat.at<float>(i, 6) * src_.rows;
			//原始ROI
			rect_face_ = cv::Rect((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			if (rect_face_.area() < 50)
				return false;
			//由于有时候会产生十分奇怪的坐标，故对坐标进行规范化
			if (rect_face_.x > src_.cols || rect_face_.x < 0 || rect_face_.y > src_.rows || rect_face_.x < 0)
			{
				return false;
			}
			//放大后的ROI
			cv::Rect roi;
			roi.x = max(0, rect_face_.x - x_padding);
			roi.y = max(0, rect_face_.y - y_padding);

			roi.width = rect_face_.width + 2 * x_padding;
			if (roi.width + roi.x > src_.cols - 1)
				roi.width = src_.cols - 1 - roi.x;

			roi.height = rect_face_.height + 2 * y_padding;
			if (roi.height + roi.y > src_.rows - 1)
				roi.height = src_.rows - 1 - roi.y;

			roi_face_all_ = src_(roi);
			//识别性别
			GetGender(roi_face_all_);

			//美化图片
			FaceBeautify(roi_face_all_, roi_face_all_);

			//像素语义分割
			FaceDetectTorch(roi_face_all_);
			//根据掩膜信息，得到各个部位图
			GetSegments();

			//获取唇部区域
			GetLip(roi_face_only_);
			//获取输入眼睛位置
			ObjectDetectHaar(roi_face_only_, objects_eyes, 2);
			if (objects_eyes.empty())
				return false;

			//获取肤色平均值
			//通过对标准值进行图层叠加完成实验

			Mat mask(roi_face_only_.rows, roi_face_only_.cols, CV_8UC3, BODY_COLOR);
			Mat dst;
			ApplyMask(MIX_TYPE::COLOR, roi_face_only_, mask, dst);

			skin_color_ = BODY_COLOR;
			GetBaldHead(dst, objects_eyes);


			return true;
		}

	}
	return false;
}

//采用haar特征提取目标特征位置
/*
* input: 输入图像
* objects_rects：识别出来的目标位置，以方框表示
* min_target_nums: 最少需要识别出来的特征数量，少于这个数，程序将直接跳出
*/
bool FaceGrabber::ObjectDetectHaar(const Mat& input, vector<Rect>& objects_rects, size_t min_target_nums)
{
	objects_rects.clear();
	if (input.empty())
		return false;
	vector<Rect> parts;
	haar_detector.detectMultiScale(input, parts, 1.2, 6, 0, cv::Size(30, 30));
	if (parts.size() != min_target_nums)
	{
		cout << "cant detect any objects_rects! " << endl;
		return false;
	}
	Mat tmp = input.clone();
	for (int i = 0; i < parts.size(); i++)
	{
		Rect ROI_haar_;
		//添加偏置
		ROI_haar_.x = max(parts[static_cast<int>(i)].x + 10, 0);
		ROI_haar_.y = max(parts[static_cast<int>(i)].y + 10, 0);
		ROI_haar_.width = min(parts[static_cast<int>(i)].width - 20, src_.cols);
		ROI_haar_.height = min(parts[static_cast<int>(i)].height - 20, src_.rows);
		cv::rectangle(tmp, ROI_haar_, cv::Scalar(0, 255, 0), 1, 8, 0);
		objects_rects.push_back(ROI_haar_);
	}
	//imshow("eyes", tmp);

	return false;
}

//语义分割
bool FaceGrabber::FaceDetectTorch(const Mat& input)
{//判空
	if (input.empty())
		return false;
	Mat image_transformed;
	const int set_size = 224;//网络需要的固定图片长宽大小
	const int multiple = 127; // 转换的倍数大小
	//重设尺寸
	resize(input, image_transformed, Size(set_size, set_size));
	cvtColor(image_transformed, image_transformed, COLOR_BGR2RGB);

	// 3.图像转换为Tensor
	torch::Tensor tensor_image = torch::from_blob(image_transformed.data, { image_transformed.rows, image_transformed.cols,3 }, torch::kByte);
	tensor_image = tensor_image.permute({ 2,0,1 });
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);
	tensor_image = tensor_image.unsqueeze(0);


	//网络前向计算
	torch::Tensor out_tensor_all = sematic_module_.forward({ tensor_image }).toTensor();
	torch::Tensor out_tensor = out_tensor_all.argmax(1);
	out_tensor = out_tensor.squeeze();

	//mul函数，表示张量中每个元素乘与一个数，clamp表示夹紧，限制在一个范围内输出
	//由于一共就三种标签0 1 2， 所以最终mat输出应该是 0 127 254
	out_tensor = out_tensor.mul(multiple).to(torch::kU8);
	out_tensor = out_tensor.to(torch::kCPU);

	dst_torch_.create(set_size, set_size, CV_8U);
	memcpy((void*)dst_torch_.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());

	//resize回原来的大小
	resize(dst_torch_, dst_torch_, Size(input.cols, input.rows), 0.0, 0.0, INTER_NEAREST);

	return true;
}
bool FaceGrabber::GetLip(const cv::Mat & input)
{
	if (roi_face_only_.empty())
	{
		cout << "error in GetLip: roi_face_only empty" << endl;
		return false;
	}
	roi_lips_only_.create(Size(input.cols, input.rows), CV_8UC1);
	for (int y = 0; y < input.rows; ++y)
	{
		for (int x = 0; x < input.cols; ++x)
		{
			const Vec3b& origin_pixel = input.at<Vec3b>(y, x);
			//转换为YIQ空间
			const auto& b = (double)origin_pixel[0], g = (double)origin_pixel[1], r = (double)origin_pixel[2];
			const auto  Y = 0.299 * r + 0.587 * g + 0.114 * b;
			const auto  I = 0.596 * r - 0.275 * g - 0.321 * b;
			const auto  Q = 0.212 * r - 0.523 * g + 0.311 * b;
			//进行阈值判断
			if ((Y >= 80 && Y <= 220 && I >= 12 && I <= 78 && Q >= 7 && Q <= 25))
			{
				roi_lips_only_.at<uchar>(y, x) = 255;
			}
			else
			{
				roi_lips_only_.at<uchar>(y, x) = 0;
			}
		}
	}
	Mat dst;
	//对图像进行闭操作
	Mat element = getStructuringElement(MORPH_RECT, Size(10, 15));
	//闭操作
	morphologyEx(roi_lips_only_, roi_lips_only_, MORPH_CLOSE, element);

	cvtColor(roi_lips_only_, dst, COLOR_GRAY2BGR);

	bitwise_and(dst, roi_face_only_, dst);
	imshow("dst", dst);

}
//获得一张有脸和头发的和一张只有脸的 并去除背景
bool FaceGrabber::GetSegments()
{
	if (roi_face_all_.empty() || dst_torch_.empty())
		return false;
	//创建一个图像矩阵的矩阵体，之后该图像只有脸
	roi_face_only_.create(Size(roi_face_all_.cols, roi_face_all_.rows), CV_8UC3);
	//创建一个图像矩阵的矩阵体，之后该图像只有头发和脸
	roi_face_hair_.create(Size(roi_face_all_.cols, roi_face_all_.rows), CV_8UC3);
	//创建一个图像，之后该图像只有头发
	roi_hair_only_.create(Size(roi_face_all_.cols, roi_face_all_.rows), CV_8UC3);
	//设置背景为黑色
	const Vec3b background = { 0, 0, 0 };
	//循环 遍历每个像素
	for (int i = 0; i < dst_torch_.rows; ++i)
	{
		for (int j = 0; j < dst_torch_.cols; ++j)
		{
			auto cur_pixel = dst_torch_.at<uchar>(i, j);
			//如果监测到头发的颜色，有头及脸的图像不做改动，另一张去除头发保持只有脸
			if (cur_pixel == TypeIndex::HAIR)
			{
				roi_face_only_.at<Vec3b>(i, j) = background;
				roi_face_hair_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				roi_hair_only_.at<Vec3b>(i, j) = Vec3b(178, 178, 158);
			}
			//如果监测到脸的颜色，两张图像都保存脸的部分
			else if (cur_pixel == TypeIndex::FACE)
			{
				roi_face_only_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				roi_face_hair_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				roi_hair_only_.at<Vec3b>(i, j) = background;
			}
			//如果是其他地方，通通变为黑色背景
			else
			{
				roi_face_only_.at<Vec3b>(i, j) = background;
				roi_face_hair_.at<Vec3b>(i, j) = background;
				roi_hair_only_.at<Vec3b>(i, j) = background;
			}
		}
	}
	//MorphologyEx(roi_face_hair_);


	return true;
}

//性别识别
void FaceGrabber::GetGender(const cv::Mat& input)
{
	cv::String gender_list[] = { "m", "f" };
	//整体像素值减去平均值（mean）通过缩放系数（scalefactor）对图片像素值进行缩放
	Mat face_blob = blobFromImage(input, 1.0, cv::Size(227, 227), cv::Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
	gender_net_.setInput(face_blob);

	Mat gender_preds = gender_net_.forward();
	Mat prob_mat = gender_preds.reshape(1, 1);
	//复制
	Mat output = src_.clone();

	Point class_number;
	double class_prob;
	//寻找矩阵(一维数组当作向量, 用Mat定义) 中最小值和最大值的位置.单通道图像
	minMaxLoc(prob_mat, NULL, &class_prob, NULL, &class_number);

	int classidx = class_number.x;
	cv::String gender = gender_list[classidx];
	//在图像上绘制文字
	putText(output, cv::format("gender:%s", gender.c_str()), rect_face_.tl(),
		cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 1, 8);
	cur_gender_ = gender;
}

//美颜处理
void FaceGrabber::FaceBeautify(Mat& input, Mat& output)
{


	Mat dst_grinded;
	FaceGrinding(input, dst_grinded);
	Mat dst_Saturated(input.size(), input.type());
	AdjustSaturation(dst_grinded, dst_Saturated);
	Mat dst_brighted(input.size(), input.type());
	AdjustBrightness(dst_Saturated, dst_brighted);
	output = dst_brighted.clone();
}



//填充闭合轮廓，输出为闭合的掩膜
void FaceGrabber::FillContour(cv::Mat& input, cv::Mat& output)
{
	//对轮廓图进行填充
	if (input.type() != CV_8UC1)
		return;
	if (output.empty())
		output.create(input.size(), input.type());

	Mat tmp = input.clone();

	vector<vector<Point>> contour;
	findContours(input, contour, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	vector<Point> longest_contour = contour[0];
	//选最长的contours
	for (const auto& v : contour)
	{
		if (v.size() > longest_contour.size())
		{
			longest_contour = v;
		}
	}
	contour = { longest_contour };

	Rect b_rect = boundingRect(longest_contour);
	b_rect.x = max(0, b_rect.x - 1);
	b_rect.y = max(0, b_rect.y - 1);
	b_rect.width += 2;
	if (b_rect.width + b_rect.x > input.cols)
		b_rect.width = input.cols - 1 - b_rect.x;
	if (b_rect.height + b_rect.y > input.rows)
		b_rect.height = input.rows - 1 - b_rect.y;
	auto fun_in_rect = [&b_rect](int x, int y)
	{
		return (x >= b_rect.x && x <= b_rect.x + b_rect.width && y >= b_rect.y && y <= b_rect.y + b_rect.height);
	};
	queue<Point> neighbor_queue;
	neighbor_queue.emplace(b_rect.x, b_rect.y);
	tmp.at<uchar>(b_rect.y, b_rect.x) = 128;

	while (!neighbor_queue.empty())
	{
		//从队列取出种子点，获取其4邻域坐标点
		auto seed = neighbor_queue.front();
		neighbor_queue.pop();

		std::vector<Point> pts;
		pts.emplace_back(seed.x, (seed.y - 1));
		pts.emplace_back(seed.x, (seed.y + 1));
		pts.emplace_back((seed.x - 1), seed.y);
		pts.emplace_back((seed.x + 1), seed.y);

		for (auto& pt : pts)
		{
			if (fun_in_rect(pt.x, pt.y) && tmp.at<uchar>(pt.y, pt.x) == 0)
			{
				//将矩形范围内且灰度值为0的可连通坐标点添加到队列
				neighbor_queue.push(pt);
				tmp.at<uchar>(pt.y, pt.x) = 128;
			}
		}

	}

	for (int i = b_rect.y; i < b_rect.y + b_rect.height; i++)
	{
		for (int j = b_rect.x; j < b_rect.x + b_rect.width; j++)
		{
			if (tmp.at<uchar>(i, j) == 0)
			{
				output.at<uchar>(i, j) = 255;
			}
		}
	}
	//imshow("output", output);


	return;
}

void FaceGrabber::GetBaldHead(cv::Mat& input, std::vector<cv::Rect>& eyes)
{
	try
	{

		if (input.empty() || eyes.empty())
			return;

		//确定左右眼的中心
		Point lefteye_center, righteye_center;
		if (eyes[0].x < eyes[1].x)
		{
			lefteye_center = SimpleMath::GetMidpt(eyes[0].tl(), eyes[0].br());
			righteye_center = SimpleMath::GetMidpt(eyes[1].tl(), eyes[1].br());
		}
		else
		{
			lefteye_center = SimpleMath::GetMidpt(eyes[1].tl(), eyes[1].br());
			righteye_center = SimpleMath::GetMidpt(eyes[0].tl(), eyes[0].br());
		}
		//生成一幅
		Mat tmp(input.clone());

		Point2d center = SimpleMath::GetMidpt(lefteye_center, righteye_center);
		Point2d virtual_top = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteye_center.x, (double)righteye_center.y), -M_PI / 2, 1.2);
		Point2d virtual_bottom = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteye_center.x, (double)righteye_center.y), M_PI / 2, 1.2);
		Point2d virtual_left = SimpleMath::GetRotatedVecRad(center, Point2d((double)lefteye_center.x, (double)lefteye_center.y), 0.01, 1.2);
		Point2d virtual_right = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteye_center.x, (double)righteye_center.y), 0.01, 1.2);

		vector<Point2d> virtual_pts = { virtual_top, virtual_bottom, virtual_left, virtual_right };

		/*circle(tmp, center, 3, Scalar(255, 0, 0), -1);
		circle(tmp, virtual_top, 3, Scalar(0, 255, 255), -1);
		circle(tmp, lefteye_center, 3, Scalar(0, 255, 255), -1);
		circle(tmp, righteye_center, 3, Scalar(0, 255, 255), -1);*/


		double short_axis = SimpleMath::GetLineLen(center, virtual_top) * 2.0;
		double long_axis = SimpleMath::GetLineLen(center, virtual_left) * 2.2;

		vector<Point> ellipes_verti;
		ellipse2Poly((Point)center, Size(long_axis, short_axis), 0, 180 + 15, 180 + 165, 1, ellipes_verti);


		Mat mask(input.size(), CV_8UC1, Scalar(0));

		for (int i = 0; i < ellipes_verti.size() - 1; ++i)
		{
			//line(tmp, ellipes_verti[i], ellipes_verti[i + 1], Scalar(123, 45, 78), 2);
			line(mask, ellipes_verti[i], ellipes_verti[i + 1], Scalar(255), 2);
		}

		Point2d br_padding = ellipes_verti.front(); br_padding.y += input.rows - br_padding.y - 1;
		Point2d bl_padding = ellipes_verti.back(); bl_padding.y += input.rows - bl_padding.y - 1;

		/*line(tmp, ellipes_verti.front(), br_padding, Scalar(123, 45, 78), 2);
		line(tmp, ellipes_verti.back(), bl_padding, Scalar(123, 45, 78), 2);
		line(tmp, bl_padding, br_padding, Scalar(123, 45, 78), 2);*/

		line(mask, ellipes_verti.front(), br_padding, Scalar(255), 2);
		line(mask, ellipes_verti.back(), bl_padding, Scalar(255), 2);
		line(mask, bl_padding, br_padding, Scalar(255), 2);

		//染发
		for (int x = 0; x < roi_hair_only_.cols; ++x)
		{
			for (int y = 0; y < roi_hair_only_.rows; ++y)
			{
				Vec3b& pixel_color = roi_hair_only_.at<Vec3b>(y, x);
				if (pixel_color[0] != 0 && pixel_color[1] != 0 && pixel_color[2] != 0)
				{
					pixel_color[0] = skin_color_[0];
					pixel_color[1] = skin_color_[1];
					pixel_color[2] = skin_color_[2];
				}
			}
		}

		tmp += roi_hair_only_;

		FillContour(mask, mask);

		//将图片转换为三通道执行与运算
		cvtColor(mask, mask, COLOR_GRAY2BGR);
		bitwise_and(tmp, mask, bald_head_);


		//imshow("mask", mask);
		//imshow("bald", bald_head_);


	}
	catch (...)
	{
		//do nothing
	}

}


//滤波
void FaceGrabber::FaceGrinding(Mat& input, Mat& output, int value1, int value2)
{
	int dx = value1 * 5;    //双边滤波参数之一  
	double fc = value1 * 12.5; //双边滤波参数之一  
	int transparency = 50; //透明度  
	cv::Mat dst;
	//双边滤波  
	bilateralFilter(input, dst, dx, fc, fc);
	dst = (dst - input + 128);
	//高斯模糊  
	GaussianBlur(dst, dst, cv::Size(2 - 1, 2 - 1), 0, 0);
	dst = input + 2 * dst - 255;
	dst = (input * (100 - transparency) + dst * transparency) / 100;
	dst.copyTo(output);
}
//调节对比度和亮度
void FaceGrabber::AdjustSaturation(cv::Mat& input, cv::Mat& output, int saturation, const int max_increment)
{

	float increment = (saturation - 80) * 1.0 / max_increment;


	for (int col = 0; col < input.cols; col++)
	{
		for (int row = 0; row < input.rows; row++)
		{
			// R,G,B 分别对应数组中下标的 2,1,0
			uchar r = input.at<Vec3b>(row, col)[2];
			uchar g = input.at<Vec3b>(row, col)[1];
			uchar b = input.at<Vec3b>(row, col)[0];

			float maxn = max(r, max(g, b));
			float minn = min(r, min(g, b));

			float delta, value;
			delta = (maxn - minn) / 255;
			value = (maxn + minn) / 255;

			float new_r, new_g, new_b;

			if (delta == 0)		 // 差为 0 不做操作，保存原像素点
			{
				output.at<Vec3b>(row, col)[0] = b;
				output.at<Vec3b>(row, col)[1] = g;
				output.at<Vec3b>(row, col)[2] = r;
				continue;
			}

			float light, sat, alpha;
			light = value / 2;

			if (light < 0.5)
				sat = delta / value;
			else
				sat = delta / (2 - value);

			if (increment >= 0)
			{
				if ((increment + sat) >= 1)
					alpha = sat;
				else
				{
					alpha = 1 - increment;
				}
				alpha = 1 / alpha - 1;
				new_r = r + (r - light * 255) * alpha;
				new_g = g + (g - light * 255) * alpha;
				new_b = b + (b - light * 255) * alpha;
			}
			else
			{
				alpha = increment;
				new_r = light * 255 + (r - light * 255) * (1 + alpha);
				new_g = light * 255 + (g - light * 255) * (1 + alpha);
				new_b = light * 255 + (b - light * 255) * (1 + alpha);
			}
			output.at<Vec3b>(row, col)[0] = new_b;
			output.at<Vec3b>(row, col)[1] = new_g;
			output.at<Vec3b>(row, col)[2] = new_r;
		}
	}
}
//调节色调
void FaceGrabber::AdjustBrightness(cv::Mat& input, cv::Mat& output, float alpha, float beta)
{
	int height = input.rows;//求出src的高
	int width = input.cols;//求出input的宽
	output = cv::Mat::zeros(input.size(), input.type());  //这句很重要，创建一个与原图一样大小的空白图片              
	//循环操作，遍历每一列，每一行的元素
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (input.channels() == 3)//判断是否为3通道图片
			{
				//将遍历得到的原图像素值，返回给变量b,g,r
				float b = input.at<Vec3b>(row, col)[0];//nlue
				float g = input.at<Vec3b>(row, col)[1];//green
				float r = input.at<Vec3b>(row, col)[2];//red
				//开始操作像素，对变量b,g,r做改变后再返回到新的图片。
				output.at<Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b * alpha + beta);
				output.at<Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g * alpha + beta);
				output.at<Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r * alpha + beta);
			}
			else if (input.channels() == 1)//判断是否为单通道的图片
			{

				float v = input.at<uchar>(row, col);
				output.at<uchar>(row, col) = cv::saturate_cast<uchar>(v * alpha + beta);
			}
		}
	}
}

void FaceGrabber::ApplyMask(const std::string & mask_type, const cv::Mat& input, const cv::Mat& mask, cv::Mat& dst)
{
	MixerFactory m_factory;
	auto mixer = m_factory.GetMixer(mask_type);
	mixer->Mix(input, mask, dst);
	imshow("dst", dst);
}




//去除背景，使背景变透明
void FaceGrabber::RemoveBackground(cv::Mat& img)
{
	if (img.channels() != 4)
	{
		cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);


		for (int y = 0; y < img.rows; ++y)
		{
			for (int x = 0; x < img.cols; ++x)
			{
				cv::Vec4b& pixel = img.at<cv::Vec4b>(y, x);
				if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				{
					pixel[0] = 0;
					pixel[1] = 0;
					pixel[2] = 0;
					pixel[3] = 0;
				}

			}
		}
	}
	else
	{
		img = img.clone();
		for (int y = 0; y < img.rows; ++y)
		{
			for (int x = 0; x < img.cols; ++x)
			{
				cv::Vec4b& pixel = img.at<cv::Vec4b>(y, x);
				if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				{
					pixel[0] = 0;
					pixel[1] = 0;
					pixel[2] = 0;
					pixel[3] = 0;
				}

			}
		}
	}
}
//对闭运算  消除黑线
void FaceGrabber::MorphologyClose(cv::Mat& img, const int& kernel_size)
{
	Mat kernel = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));
	morphologyEx(img, img, MORPH_CLOSE, kernel);
	//medianBlur(img, img, 3);
}

//const string path("D:/CPP_Projects/FaceGrabberVid/");
const string path("F:/Beauty/Beauty/Assets/Resources/");

void FaceGrabber::CleanDisk()
{
	const string suffix(".png");
	remove((path + string("head-2") + suffix).c_str());
	remove((path + string("face-2") + suffix).c_str());
	remove((path + string("head-1") + suffix).c_str());
}



void FaceGrabber::WritePic2Disk()
{

	RemoveBackground(roi_face_hair_);
	resize(roi_face_hair_, roi_face_hair_, Size(150, 200), 0.0, 0.0, 0);

	RemoveBackground(bald_head_);
	resize(bald_head_, bald_head_, Size(128, 168));

	resize(roi_face_all_, roi_face_all_, Size(400, 400), 0.0, 0.0, 0);
	Mat mask(roi_face_all_.size(), roi_face_all_.type(), Scalar(0, 0, 0, 0));
	circle(mask, Point(200, 200), 200, Scalar(255, 255, 255, 255), -1);
	bitwise_and(roi_face_all_, mask, roi_face_all_);

	if (!cur_gender_.empty())
	{
		const string suffix(".png");
		imwrite(path + string("head-2") + suffix, roi_face_hair_);
		imwrite(path + string("face-2") + suffix, bald_head_);
		imwrite(path + string("head-1") + suffix, roi_face_all_);
	}
}

Scalar FaceGrabber::GetSkinColor(const cv::Mat& input)
{
	if (input.channels() != 3)
		return Scalar();

	Mat dst = input.clone();

	for (auto& rect : objects_eyes)
	{
		ZoomRect(rect, 10, 10, input.size());
		rectangle(dst, rect, Scalar(0, 0, 0), -1);
	}
	imshow("dst", dst);

	size_t color_r = 0, color_g = 0, color_b = 0, pix_size = 0;

	for (int x = 0; x < dst.cols; ++x)
	{
		for (int y = 0; y < dst.rows; ++y)
		{
			const Vec3b& pixel_color = dst.at<Vec3b>(y, x);
			if (pixel_color[0] != 0 && pixel_color[1] != 0 && pixel_color[2] != 0)
			{
				color_r += pixel_color[0];
				color_b += pixel_color[1];
				color_b += pixel_color[2];
				++pix_size;
			}
		}
	}
	return Scalar(color_r / pix_size, color_g / pix_size, color_b / pix_size);
}

void FaceGrabber::ZoomRect(cv::Rect& rect, const int x, const int y, cv::Size pic_size)
{
	rect.x = max(0, rect.x - x);
	rect.y = max(0, rect.y - y);
	rect.width = rect.x + 2 * x;
	if (rect.width > pic_size.width)
		rect.width = pic_size.width - 1 - rect.x;
	if (rect.height > pic_size.height)
		rect.height = pic_size.height - 1 - rect.y;
}



//显示相机一帧帧图像
void FaceGrabber::ShowSrc()
{
	imshow("src", src_);
	waitKey(1);
}
//显示语义分割的图像
void FaceGrabber::ShowDstTorch()
{
	imshow("dst_torched", dst_torch_);


	return;
}
//显示一张有脸和头发的和一张只有脸的
void FaceGrabber::ShowROIFace()
{
	imshow("roi_face", roi_face_all_);
	imshow("roi_face_hair_", roi_face_hair_);
	imshow("roi_face_only_", roi_face_only_);

}

void FaceGrabber::ShowBaldHead()
{
	imshow("bald_head", bald_head_);
	waitKey(1);
}

void FaceGrabber::ShowDebug()
{
	imshow("lips", roi_lips_only_);
	waitKey(1);
}
