#include "PS_Mixer.h"

using namespace std;
using namespace cv;

bool CheckInput(const cv::Mat & src, const cv::Mat & mask)
{
	//检查源图像的通道数
	if (src.channels() < 3)
	{
		cout << "error in color_mixer, you must input a 3 channel pic" << endl;
		return false;
	}
	else if (src.empty() || mask.empty())
	{
		cout << "input or mask empty" << endl;
		return false;
	}
	return true;
}

bool ColorMixer::Mix(const cv::Mat & src, const cv::Mat& mask, cv::Mat & output)
{
	//检查输入
	if (!CheckInput(src, mask))
		return false;
	//转换源图像和mask为HSV
	Mat src_hsv, mask_hsv;
	cvtColor(src, src_hsv, COLOR_BGR2HSV);
	cvtColor(mask, mask_hsv, COLOR_BGR2HSV);
	//给output图像内存
	output.create(Size(src.size()), src.type());

	//遍历图像进行赋值
	for (int y = 0; y < src.rows; ++y)
	{
		for (int x = 0; x < src.cols; ++x)
		{
			//HcScBc = HaSaBb
			const auto& pixel_a = mask_hsv.at<Vec3b>(y, x);
			const auto& pixel_b = src_hsv.at<Vec3b>(y, x);
			output.at<Vec3b>(y, x) = { pixel_a[0], pixel_a[1], pixel_b[2] };
		}
	}
	cvtColor(output, output, COLOR_HSV2BGR);

	return true;
}

bool ColorMixer::Release()
{
	delete this;
	return true;
}

MixerPtr MixerFactory::GetMixer(const std::string & mix_mode)
{
	if (mix_mode == MIX_TYPE::COLOR)
	{
		return MixerPtr(new ColorMixer, mem_fun(&ColorMixer::Release));
	}
}
