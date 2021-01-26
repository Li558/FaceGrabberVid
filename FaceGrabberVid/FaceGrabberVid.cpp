#include <iostream>
#include "src/std_includes.h"
#include "src/opencv_includes.h"
#include "src/FaceGrabber.h"

using namespace std;
using namespace cv;

int main()
{
	FaceGrabber fg;
	fg.StarGrab();

	while (true)
	{
		fg.GetFrame();
		if (fg.GetFace())
		{
			fg.ShowDstTorch();
			fg.ShowROIFace();
			fg.FaceBeautify();
			if (waitKey(1) == 'q')
				return 1;
		}

	}

	return 1;
}
