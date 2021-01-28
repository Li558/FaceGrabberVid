#include "FaceGrabber.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;



//判空
bool FaceGrabber::StarGrab()
{
	cap_.open(0);
	if (!cap_.isOpened())
	{
		cout << "error cam open failed" << endl;
		return false;
	}



}
//采用haar特征提取人脸
bool FaceGrabber::FaceDetectHaar()
{
	if (src_.empty())
		return false;
	std::vector<cv::Rect> faces;
	face_cascade_.detectMultiScale(src_, faces, 1.2, 6, 0, cv::Size(120, 120));
	if (faces.empty())
	{
		cout << "cant detect any faces! " << endl;
		return false;
	}
	for (int i = 0; i < faces.size(); i++)
	{
		Rect ROI_haar_;
		//添加偏置
		ROI_haar_.x = max(faces[static_cast<int>(i)].x - 10, 0);
		ROI_haar_.y = max(faces[static_cast<int>(i)].y - 80, 0);
		ROI_haar_.width = min(faces[static_cast<int>(i)].width + 10, src_.cols);
		ROI_haar_.height = min(faces[static_cast<int>(i)].height + 80, src_.rows);
		cv::rectangle(src_, ROI_haar_, cv::Scalar(0, 255, 0), 1, 8, 0);
		roi_face_all_ = src_(ROI_haar_);

		FaceDetectTorch(roi_face_all_);
		GetSegments();
	}

	return false;
}
//读入一帧帧图片
void FaceGrabber::GetFrame()
{

	cap_ >> src_;
	//cv::imshow("原图", src_);



}
//美颜处理
void FaceGrabber::FaceBeautify(Mat& input, Mat& output)
{
	Mat dst_grinded(input.size(), input.type());
	FaceGrinding(input, dst_grinded);
	Mat dst_Saturated(input.size(), input.type());
	AdjustSaturation(dst_grinded, dst_Saturated);
	Mat dst_brighted(input.size(), input.type());
	AdjustBrightness(dst_Saturated, dst_brighted);
	output = dst_brighted.clone();
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
	resize(dst_torch_, dst_torch_, Size(input.cols, input.rows));

	return true;
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
			}
			//如果监测到脸的颜色，两张图像都保存脸的部分
			else if (cur_pixel == TypeIndex::FACE)
			{
				roi_face_only_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				roi_face_hair_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
			}
			//如果是其他地方，通通变为黑色背景
			else

			{
				roi_face_only_.at<Vec3b>(i, j) = background;
				roi_face_hair_.at<Vec3b>(i, j) = background;
			}
		}
	}
	MorphologyEx(roi_face_hair_);
	Remove_background(roi_face_hair_);
	Remove_background(roi_face_only_);

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
				output.at<Vec3b>(row, col)[0] = new_b;
				output.at<Vec3b>(row, col)[1] = new_g;
				output.at<Vec3b>(row, col)[2] = new_r;
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
//得到人脸
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

			roi.height = rect_face_.height + y_padding;
			if (roi.height + roi.y > src_.rows - 1)
				roi.height = src_.rows - 1 - roi.y;

			roi_face_all_ = src_(roi).clone();
			GetGender(roi_face_all_);
			FaceBeautify(roi_face_all_, roi_face_all_);
			FaceDetectTorch(roi_face_all_);
			GetSegments();
			return true;
		}

	}
	return false;
}
//去除背景，使背景变透明
void FaceGrabber::Remove_background(cv::Mat& img)
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
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
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
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
					pixel[3] = 0;
				}

			}
		}
	}
}
//对闭运算  消除黑线
void FaceGrabber::MorphologyEx(cv::Mat& img)
{
	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	morphologyEx(img, img, MORPH_CLOSE, kernel);
	//medianBlur(img, img, 3);
}

void FaceGrabber::CleanDisk()
{
	const string suffix(".bmp");
	for (int i = 0; i < 3; ++i)
	{

		string fi_filename(string("f") + to_string(i) + suffix);
		string mi_filename(string("m") + to_string(i) + suffix);
		remove(fi_filename.c_str());
		remove(mi_filename.c_str());

	}
}

void FaceGrabber::WritePic2Disk()
{
	if (!cur_gender_.empty())
	{
		const string suffix(".bmp");
		imwrite(cur_gender_ + string("0") + suffix, roi_face_all_);
		imwrite(cur_gender_ + string("1") + suffix, roi_face_hair_);
		imwrite(cur_gender_ + string("2") + suffix, roi_face_only_);
	}
}
