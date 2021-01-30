#include <iostream>
#include "src/std_includes.h"
#include "src/opencv_includes.h"
#include "src/FaceGrabber.h"

using namespace std;
using namespace cv;

int main()
{
	FaceGrabber fg;

	//如果false 退出程序
	if (!fg.StarGrab())
	{
		return 1;
	}
	//执行死循环
	while (true)
	{
		fg.CleanDisk();
		//读入一帧帧图片
		fg.GetFrame();
		if (fg.GetFace())
		{

			/*fg.ShowDstTorch();
			fg.ShowROIFace();*/
			fg.WritePic2Disk();
			return 1;
		}

	}

	return 1;
}
