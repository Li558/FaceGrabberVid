#pragma once
#include "src/opencv_includes.h"
#include "src/std_includes.h"

/*
第一次写工厂类
我们尝试对不同的图层混合算法进行测试，直到找到我们需要的类
*/

namespace MIX_TYPE
{
	const std::string COLOR("COLOR");
}

//定义抽象类
struct PS_Mixer
{
	virtual bool Mix(const cv::Mat& src, const cv::Mat& mask, cv::Mat& output) = 0;
	virtual bool Release() = 0;
};
//定义抽象类智能指针
typedef std::shared_ptr<PS_Mixer> MixerPtr;

//公共非类方法
bool CheckInput(const cv::Mat& src, const cv::Mat& mask);

//定义简单工厂类
struct MixerFactory
{
	MixerPtr GetMixer(const std::string& mix_mode);
};

//定义各个实现类
//颜色混合模式
class ColorMixer : public PS_Mixer
{
public:
	bool Mix(const cv::Mat& src, const cv::Mat& mask, cv::Mat& output);
	bool Release();
};

