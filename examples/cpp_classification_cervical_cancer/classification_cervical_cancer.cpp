#include "CaffeClassifier.h"
#include "ManageDirectory.h"
#include "StringParse.h"
#include "ValidateModel.h"
#include <omp.h>
#include "ConvertString.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		printf("필요한 인자가 부족합니다.\n");
		printf("1.모델 score계산 : CalScore modelroot testlistfile imgRoot TrainedFilePrefix 사용gpu MultiNum nWidth nHeight\n");
		printf("2.불량분류 : classifyListDir ResultOutDir 사용gpu");
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
		string strGpu = argv[6];

		string strWidth;
		string strHeight;
		bool bFreeSize = false;
		if (argc > 8)
		{
			strWidth = argv[7];
			strHeight = argv[8];
		}
		else
		{
			bFreeSize = true;
		}

		string strModelFile = strTrainedPreFix + "*.caffemodel";
		string strScoreFile = strTrainedPreFix + "*_PrdResult.txt";

		int nGpu = atoi(strGpu.c_str());
		int nWidth = atoi(strWidth.c_str());
		int nHeight = atoi(strHeight.c_str());
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
				string strTemp = vStrIter[i] + "_PrdResult.txt";
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
