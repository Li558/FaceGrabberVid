#include <iostream>
#include "src/std_includes.h"
#include "src/opencv_includes.h"
#include "src/FaceGrabber.h"
#include "src/dlib_includes.h"
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
		//读入一帧帧图片
		fg.GetFrame();
		if (fg.ProcesseFace())
		{
			
			/*fg.ShowDstTorch();
			fg.ShowROIFace();*/
			fg.ShowBaldHead();
			//fg.ShowDebug();
		} 

	}

	return 1;
}
