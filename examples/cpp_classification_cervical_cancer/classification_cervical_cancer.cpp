#include "CaffeClassifier.h"
#include "ManageDirectory.h"
#include "StringParse.h"
#include "ValidateModel.h"
#include <omp.h>
#include "ConvertString.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	if (argc < 5)
	{
		printf("arg number error\n");
		printf("modelroot testlistfile imgRoot TrainedFilePrefix gpu_id \n");
		getchar();
		return 0;
	}


	string strModelRoot = argv[1];
	strModelRoot += "/";
	string strTestListFile = argv[2];
	string strImgRoot = argv[3];
	strImgRoot += "/";
	string strTrainedPreFix = argv[4];
	string strGpuid = argv[5];

	string strModelFile = strTrainedPreFix + "*.caffemodel";
	string strScoreFile = strTrainedPreFix + "*_PrdResult.csv";

	const int nGpu = atoi(strGpuid.c_str());
	const int nWidth = 360;
	const int nHeight = 270;
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
