#pragma once
#include "src/opencv_includes.h"
#include "src/std_includes.h"

/*
��һ��д������
���ǳ��ԶԲ�ͬ��ͼ�����㷨���в��ԣ�ֱ���ҵ�������Ҫ����
*/

namespace MIX_TYPE
{
	const std::string COLOR("COLOR");
}

//���������
struct PS_Mixer
{
	virtual bool Mix(const cv::Mat& src, const cv::Mat& mask, cv::Mat& output) = 0;
	virtual bool Release() = 0;
};
//�������������ָ��
typedef std::shared_ptr<PS_Mixer> MixerPtr;

//�������෽��
bool CheckInput(const cv::Mat& src, const cv::Mat& mask);

//����򵥹�����
struct MixerFactory
{
	MixerPtr GetMixer(const std::string& mix_mode);
};

//�������ʵ����
//��ɫ���ģʽ
class ColorMixer : public PS_Mixer
{
public:
	bool Mix(const cv::Mat& src, const cv::Mat& mask, cv::Mat& output);
	bool Release();
};

