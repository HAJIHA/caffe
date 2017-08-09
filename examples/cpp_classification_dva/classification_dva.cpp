#include "CaffeClassifier.h"
#include "ManageDirectory.h"
#include "StringParse.h"
#include "ValidateModel.h"
#include <omp.h>
#include "ConvertString.h"

using namespace std;
using namespace cv;

typedef struct fileinfo
{
	enum {
		DEPLOY,
		MODEL,
		MEAN,
		SYNC,
		LINE,
		CELLID,
		DEVICEID,
		STEP,
		CLIPIMAGEPATH,
		ORIGINPATH,
		CLIPX,
		CLIPY,
		WIDTH,
		HEIGHT,
		E_TIME,
		MODULEID,
		BATCH_ID,
		GRADE,
		LAYER,
		COORDX,
		COORDY,
		GLASSID,
		STEPID,
		PRODTYPE,
		STACK_FLAG,
		MODEL_TIME,
		MODEL_PATH,
		GATE_LINE,
		DATA_LINE,
		INSP_BDATE,
		LAST,
	};
}INFO_IDX;

int main(int argc, char** argv)
{
	//[deploy.txt 경로] [학습된 모델 파일 경로] [학습된 모델의 mean 파일 경로] [싱크셋 파일 경로] [예측할 이미지 파일 경로]
	//[cellid]	[deviceid] [step]	[originpath]	[clipx]	[clipy]
	//CalScore modelroot, testlistfile, imgRoot, TrainedFilePrefix
	//I:\DME\ClassfyListDir\GATE_ACI I : \DME\ResultOutDir 0
	//classifyListDir ResultOutDir 사용gpu

	//classifyListDir1,classifyListDir2 ResultOutDir  gpu사용list
	//classifyListDir1,classifyListDir2 ResultOutDir  0,0,0,1,1,1  : 0번 3개 모델 ,1번 3개 모델 6개 Thread가 동시에 사용.

	if (argc < 3)
	{
		printf("필요한 인자가 부족합니다.\n");
		printf("1.모델 score계산 : CalScore modelroot testlistfile imgRoot TrainedFilePrefix 사용gpu MultiNum\n");
		printf("2.불량분류 : classifyListDir ResultOutDir 사용gpu");
		getchar();
		return 0;
	}


	if (string(argv[1]) == "CalScore")
	{
		string strModelRoot = argv[2];
		strModelRoot += "/";
		string strTestListFile = argv[3];
		string strImgRoot = argv[4];
		strImgRoot += "/";
		string strTrainedPreFix = argv[5];
		string strGpu = argv[6];
		//string strMultiNum = argv[7];

		string strModelFile = strTrainedPreFix + "*.caffemodel";
		string strScoreFile = strTrainedPreFix + "*_Score.txt";

		int nGpu = atoi(strGpu.c_str());
		//int nMulti = atoi(strMultiNum.c_str());
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
				string strTemp = vStrIter[i] + "_Score.txt";
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

				CValidateModel val(strModelRoot, vStrTrainedFile[i], strImgRoot, strTestListFile, nGpu);
			}


		}
		return 0;
	}


	string strpath = argv[1];
	vector<string> vStrPath = CStringParse::splitString(strpath, ',');
	string strAiResultPath = argv[2];
	string strGpuNum = argv[3];
	vector<string> vStrGpuNum = CStringParse::splitString(strGpuNum, ',');
	//int nGpuNum = atoi(strGpuNum.c_str());
	vector<int> vGpuNum;
	for (int i = 0; i < vStrGpuNum.size(); i++)
	{
		vGpuNum.push_back(atoi(vStrGpuNum[i].c_str()));
	}

	CaffeClassifier classifier[16];

	//for (int i = 0; i < vStrPath.size(); i++)
	//{
	//vStrPath[i] =  CManageDirectory::checkNAddSlush(vStrPath[i]);
	//}
	vector<string> vStrPath_Prior, vStrPath_All;
	int nFlagDepart = 2; // SDACI,BIZO_ADI
	for (int i = nFlagDepart; i < vStrPath.size(); i++) //except STEP  Priority
	{
		vStrPath_All.push_back(CManageDirectory::checkNAddSlush(vStrPath[i]));
	}
	//strpath += "/";
	strAiResultPath = CManageDirectory::checkNAddSlush(strAiResultPath);

	for (int i = 0; i < nFlagDepart; i++) //STEP  Priority by now SDACI,BIZO_ADI
	{
		vStrPath_Prior.push_back(CManageDirectory::checkNAddSlush(vStrPath[i]));
	}

	const int nBatchSize = 10;
	const int nOverSample = 10;
	while (1)
	{
		_sleep(1000);
		//vector<string> vClassifyList = CManageDirectory::get_files_inDirectory(strpath, "*.txt",6);
		//vector<string> vClassifyList = CManageDirectory::get_files_inDirectory(vStrPath, "*.txt", 6);
		vector<string> vClassifyList, vClassifyList_tmpPRIOR, vClassifyList_tmpALL;
		vClassifyList_tmpPRIOR = CManageDirectory::get_files_inDirectory(vStrPath_Prior, "*.txt", 18);
		_sleep(5);
		vClassifyList_tmpALL = CManageDirectory::get_files_inDirectory(vStrPath_All, "*.txt", 18);

		if (vClassifyList_tmpPRIOR.size() > 10)
		{
			vClassifyList = vClassifyList_tmpPRIOR;
			std::cout << "---------STEP Priority Process-----------" << std::endl;
		}
		else if (vClassifyList_tmpALL.size() > 3000)
		{
			int sizePorition = vClassifyList_tmpALL.size()*0.2;
			std::cout << "---------Portion Count " + to_string(sizePorition) + "-----------" << std::endl;
			for (int i = 0; i< sizePorition; i++)
			{
				vClassifyList.push_back(vClassifyList_tmpALL[i]);
			}
		}
		else
			vClassifyList = vClassifyList_tmpALL;

		vClassifyList_tmpPRIOR.clear();
		vClassifyList_tmpALL.clear();

		int nImgBatch = nBatchSize / nOverSample;
		int nCnt = 0;
		int nThread = vGpuNum.size();// 3;
		omp_set_dynamic(0);     // Explicitly disable dynamic teams
		omp_set_num_threads(nThread); // Use 2 threads for all consecutive parallel regions
#pragma omp parallel for ordered schedule(static,1)
		for (int i = 0; i < vClassifyList.size(); i++)
		{
			int clIdx = i % nThread;
			int nGpuNum = vGpuNum[clIdx];
			CString strCurTime = CConvertString::getCurTimeCString();

			std::ifstream inFile;
			//string filename = strpath + vClassifyList[i];
			string resulstFileName = strAiResultPath + CManageDirectory::getFileName(vClassifyList[i]);

			inFile.open(vClassifyList[i]);
			if (inFile.is_open())
			{
				char buf[1024];
				while (!inFile.getline(buf, sizeof(buf)).eof()) {
				}
				string line = string(buf);

				vector<string> para = CStringParse::splitString(line, ',');
				string model_file = para[INFO_IDX::DEPLOY];
				string trained_file = para[INFO_IDX::MODEL];
				string mean_file = para[INFO_IDX::MEAN];
				string label_file = para[INFO_IDX::SYNC];
				string file = para[INFO_IDX::CLIPIMAGEPATH];

				//test
				int nFcn = model_file.rfind("_fcn.prototxt");
				if (nFcn > 0)
				{
					//remove(vClassifyList[i].c_str());
					//continue;
					classifier[clIdx].loadModel(model_file, trained_file, mean_file, label_file, true, 1, nGpuNum);
					classifier[clIdx].setFcn(true);
				}
				else
				{
					classifier[clIdx].loadModel(model_file, trained_file, mean_file, label_file, true, nBatchSize, nGpuNum);
					classifier[clIdx].setFcn(false);
				}

				clock_t st;
				clock_t et;
				string strDisp;
				strDisp = CConvertString::CStringToString(strCurTime) + file + " : ";

				struct tm *t;
				time_t timer;

				timer = time(NULL);    // 현재 시각을 초 단위로 얻기
				t = localtime(&timer); // 초 단위의 시간을 분리하여 구조체에 넣기

				char s[15];

				sprintf(s, "%04d%02d%02d%02d%02d%02d",
					t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
					t->tm_hour, t->tm_min, t->tm_sec
				);
				para[INFO_IDX::E_TIME] = string(s);
				//std::cout << "---------- Prediction for " << file << " ----------" << std::endl;

				st = clock();
				cv::Mat img = cv::imread(file, -1);

				if (img.empty())
				{
					inFile.close();
					remove(vClassifyList[i].c_str());
					std::cout << "---------- Not Exist Image " << file << " ----------" << std::endl;
				}
				else
				{
					et = clock();
					double dAllTime = 0.0;
					double dTime = (double)(et - st) / CLOCKS_PER_SEC;
					dAllTime += dTime;
					strDisp += "read(" + to_string(dTime) + "),";

					//full Image;

					st = clock();
					cv::resize(img, img, cv::Size(800, 600));
					et = clock();
					dTime = (double)(et - st) / CLOCKS_PER_SEC;
					dAllTime += dTime;
					strDisp += "resize(" + to_string(dTime) + "),";

					st = clock();

					vector< vector<Prediction> >  predictions;
					vector<Mat> vImg;
					vImg.push_back(img);
					if (classifier[clIdx].isFcn())
					{
						predictions = classifier[clIdx].ClassifyFcnBatch(vImg, 1);
					}
					else
					{
						vector<Prediction> prediction = classifier[clIdx].ClassifyOverSample(vImg[0], 1, nOverSample);
						predictions.push_back(prediction);
					}
					et = clock();
					dTime = (double)(et - st) / CLOCKS_PER_SEC;
					dAllTime += dTime;
					strDisp += "predict(" + to_string(dTime) + "),";


					st = clock();
					ofstream outFile(resulstFileName, ios::out); //파일을 연다, 없으면 생성
					vector<string> vResult;
					for (int pi = INFO_IDX::LINE; pi < INFO_IDX::MODEL_TIME; pi++)
						vResult.push_back(para[pi]);

					vResult.push_back(predictions[0][0].first);
					vResult.push_back(to_string(predictions[0][0].second));

					for (int pi = INFO_IDX::MODEL_TIME; pi < INFO_IDX::LAST; pi++)
						vResult.push_back(para[pi]);

					for (int pi = 0; pi < vResult.size(); pi++)
					{
						outFile << vResult[pi] << ',';
					}

					inFile.close();
					outFile.close();
					int ret = remove(vClassifyList[i].c_str());

					et = clock();
					dTime = (double)(et - st) / CLOCKS_PER_SEC;
					dAllTime += dTime;
					strDisp += "write(" + to_string(dTime) + "),";
					strDisp += "All(" + to_string(dAllTime) + "),";

					std::cout << strDisp << std::endl;
				}
			}

		}
	}
}
