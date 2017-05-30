#include "CaffeClassifier.h"
#include "ManageDirectory.h"
#include "StringParse.h"
#include "ValidateModel.h"
#include <omp.h>
#include "ConvertString.h"

using namespace std;
using namespace cv;

DEFINE_int32(gpu, 0,
	"Optional; run in GPU mode on given device IDs separated by ','."
	"Use '-gpu all' to run on all available GPUs. The effective training "
	"batch size is multiplied by the number of devices.");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		printf("arg number error\n");
		printf("intel modelroot testlistfile imgRoot TrainedFilePrefix\n");
		getchar();
		return 0;
	}

	if (string(argv[1]) == "intel" )
	{
		string strModelRoot = argv[2];
		strModelRoot += "/";
		string strTestListFile = argv[3];
		string strImgRoot = argv[4];
		strImgRoot += "/";
		string strTrainedPreFix = argv[5];

		string strModelFile = strTrainedPreFix + "*.caffemodel";
		string strScoreFile = strTrainedPreFix + "*_PrdResult.csv";

		const int nGpu = std::max<int>(0, FLAGS_gpu);;
		const int nWidth = std::max<int>(0, FLAGS_resize_width);
		const int nHeight = std::max<int>(0, FLAGS_resize_height);
		while (1)
		{
			_sleep(1000);
			vector<string> vStrTrainedFile = CManageDirectory::get_files_inDirectory(strModelRoot, strModelFile);
			vector<string> vStrScoreFile = CManageDirectory::get_files_inDirectory(strModelRoot, strScoreFile);
			vector<string> vStrIter;
			for (int i = 0; i < vStrTrainedFile.size(); i++)
			{
				vector<string> vStrTemp = CStringParse::splitString(vStrTrainedFile[i], '.');
				vStrIter.push_back(vStrTemp[0]);
			}

			for (int i = 0; i < vStrIter.size(); i++)
			{
				string strTemp = vStrIter[i] + "_PrdResult.csv";
				bool bSkip = false;
				for (int j = 0; j < vStrScoreFile.size(); j++)
				{
					if (vStrScoreFile[j] == strTemp)
					{
						bSkip = true;
						continue;
					}

				}

				if (bSkip)
					continue;

				CValidateModel val(strModelRoot, vStrTrainedFile[i], strImgRoot, strTestListFile, nGpu, true, nWidth, nHeight);
			}
		}
		return 0;
	}
	return 1;
}
