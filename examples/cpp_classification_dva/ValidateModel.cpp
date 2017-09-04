#include "caffeClassifier.h"
#include "StringParse.h"
#include "ValidateModel.h"
#include "ManageDirectory.h"


void CValidateModel::InitModel(string strModelRoot, string strTrainedFile, string strImgRoot, string strTestListFile, int gpuNum, bool bIntel)
{
	if (bIntel)
	{
		m_nBatchSize = 1;
		m_nOverSample = 1;
	}
	else
	{
		m_nBatchSize = 10;
		m_nOverSample = 10;
	}
	string strRoot = CManageDirectory::checkNAddSlush(strModelRoot);

	m_trained_file = strRoot + strTrainedFile;
	m_mean_file = strRoot + "train.binaryproto";
	m_label_file = strRoot + "synset_words.txt";
	m_ValidateListFile = strRoot + strTestListFile;
	vector<string> vTraindFile = CStringParse::splitString(strTrainedFile, '.');
	string strPredictResultFile = vTraindFile[0] + "_PrdResult.txt";
	string strScoreResultFile = vTraindFile[0] + "_Score.txt";
	// is fcn train model
	int nFcn = vTraindFile[0].rfind("_fcn");
	if (nFcn > 0)
	{
		m_model_file = strRoot + "deploy_fcn.prototxt";
		m_classifier.loadModel(m_model_file, m_trained_file, m_mean_file, m_label_file, true, 1, gpuNum);
		m_classifier.setFcn(true);
	}
	else
	{
		m_model_file = strRoot + "deploy.prototxt";
		m_classifier.loadModel(m_model_file, m_trained_file, m_mean_file, m_label_file, true, m_nBatchSize, gpuNum);
		m_classifier.setFcn(false);
	}

	m_strSavePredictResult = strRoot + strPredictResultFile;
	m_strSaveValidate = strRoot + strScoreResultFile;
	m_ImgRoot = strImgRoot;
	m_label = m_classifier.getLabelList();
}

void CValidateModel::testValidateSetIntel()
{
	std::ifstream inFile;
	inFile.open(m_ValidateListFile);
	if (!inFile.is_open())
		return;

	char buf[1024];
	bool bTestSet = false;

	while (!inFile.getline(buf, sizeof(buf)).eof())
	{
		string line = string(buf);
		vector<string> vstrline = CStringParse::splitString(line, ' ');

		stValSetIntel OneSet;
		OneSet.strPath = vstrline[0];
		if (vstrline.size() == 1)
		{
			bTestSet = true;
			OneSet.strOrginCode = "NotCal";
		}
		else
		{
			int nSolLabel = atoi(vstrline[1].c_str());
			OneSet.strOrginCode = m_label[nSolLabel];
		}
		m_vValidateIntelList.push_back(OneSet);
	}

	string strHeader = "image_name\tType_1\tType_2\tType_3";
	ofstream outFile(m_strSavePredictResult, ios::out); //파일을 연다, 없으면 생성
	ofstream outFileCal(m_strSaveValidate, ios::out);
	outFile << strHeader + "\n";

	struct stValCal
	{
		int nSolTotal;
		int nSolCorrect;
		int nPrdTotal;
		int nPrdCorrect;
	};
	map<string, stValCal>  mapValCal;
	map<string, stValCal>::iterator iterValCal;

	const int nImgSize = m_nBatchSize / m_nOverSample;
	vector<cv::Mat> vImg;
	int nCnt = 0;
	for (int i = 0; i < m_vValidateIntelList.size(); i++)
	{
		clock_t st;
		clock_t et;
		st = clock();
		m_vValidateIntelList[i].strPath = m_ImgRoot + m_vValidateIntelList[i].strPath;
		if (m_vValidateIntelList[i].strPredictCode == "")
		{
			cv::Mat img = cv::imread(m_vValidateIntelList[i].strPath);
			if (img.empty())
			{
				inFile.close();
				//remove(m_vValidateIntelList[i].strPath.c_str());
				std::cout << "---------- Not Exist Image " << m_vValidateIntelList[i].strPath << " ----------" << std::endl;
				continue;
			}

			if (img.cols < img.rows)
				cv::transpose(img, img);
			cv::resize(img, img, cv::Size(800, 600));

			vImg.push_back(img);
			nCnt++;
			if (nCnt < nImgSize)
			{
				et = clock();
				double dTime = (double)(et - st) / CLOCKS_PER_SEC;
				cout << "process time one image : " + to_string(dTime) << endl;
				continue;
			}

			///////////////////////////////////////////////////

			vector< vector<Prediction> >  vPredictions;
			if (m_classifier.isFcn())
			{
				vPredictions = m_classifier.ClassifyFcnBatch(vImg, 1);
			}
			else
			{
				vector<Prediction> prediction = m_classifier.ClassifyOverSample(vImg[0], 3, 1);
				vPredictions.push_back(prediction);
			}
			vImg.clear();
			nCnt = 0;
			m_vValidateIntelList[i].strPredictCode = vPredictions[0][0].first;

			for (int labelidx = 0; labelidx < 3; labelidx++)
			{
				for (int prdidx = 0; prdidx < 3; prdidx++)
				{
					if (m_label[labelidx] == vPredictions[0][prdidx].first)
					{
						m_vValidateIntelList[i].fProb[labelidx] = vPredictions[0][prdidx].second;
						break;
					}
				}
			}
		}
		else
		{
			stValSetIntel tempSet;
			tempSet = m_vValidateIntelList[i];
			mapValCal[tempSet.strOrginCode].nSolTotal++;
			mapValCal[tempSet.strPredictCode].nPrdTotal++;
			if (tempSet.strPredictCode == tempSet.strOrginCode)
			{
				mapValCal[tempSet.strPredictCode].nSolCorrect++;
				mapValCal[tempSet.strPredictCode].nPrdCorrect++;
			}
			string strOneLineOut;
			strOneLineOut = tempSet.strPath + "\t";
			//strOneLineOut += tempSet.strOrginCode + "\t";
			//strOneLineOut += tempSet.strPredictCode + "\t";
			strOneLineOut += to_string(tempSet.fProb[0]) + "\t";
			strOneLineOut += to_string(tempSet.fProb[1]) + "\t";
			strOneLineOut += to_string(tempSet.fProb[2]) + "\n";
			outFile << strOneLineOut;
			continue;
		}

		for (int p = 0; p < nImgSize; p++)
		{
			stValSetIntel tempSet;
			tempSet = m_vValidateIntelList[i - (nImgSize - 1) + p];

			mapValCal[tempSet.strOrginCode].nSolTotal++;
			mapValCal[tempSet.strPredictCode].nPrdTotal++;
			if (tempSet.strPredictCode == tempSet.strOrginCode)
			{
				mapValCal[tempSet.strPredictCode].nSolCorrect++;
				mapValCal[tempSet.strPredictCode].nPrdCorrect++;
			}

			string strOneLineOut;
			strOneLineOut = tempSet.strPath + "\t";
			//strOneLineOut += tempSet.strOrginCode + "\t";
			//strOneLineOut += tempSet.strPredictCode + "\t";
			strOneLineOut += to_string(tempSet.fProb[0]) + "\t";
			strOneLineOut += to_string(tempSet.fProb[1]) + "\t";
			strOneLineOut += to_string(tempSet.fProb[2]) + "\n";
			outFile << strOneLineOut;
		}

		et = clock();
		double dTime = (double)(et - st) / CLOCKS_PER_SEC;
		cout << "process time one image : " + to_string(dTime) << endl;
	}

	string strValHeader = "model\tPrdOrg\t";
	for (iterValCal = mapValCal.begin(); iterValCal != mapValCal.end(); iterValCal++)
		strValHeader += iterValCal->first + "\t";

	strValHeader += "total\n";
	outFileCal << strValHeader;


	string strModel = CStringParse::getLastString(m_trained_file, '/');
	string strValOutOrg = strModel + "\tOrg\t";
	string strValOutPrd = strModel + "\tPrd\t";
	int nOrgSum = 0;
	int nPrdSum = 0;
	int nTotalSum = 0;
	for (iterValCal = mapValCal.begin(); iterValCal != mapValCal.end(); iterValCal++)
	{
		nTotalSum += iterValCal->second.nSolTotal;
		nOrgSum += iterValCal->second.nSolCorrect;
		nPrdSum += iterValCal->second.nPrdCorrect;
		float fOrgRatio = static_cast<float>(iterValCal->second.nSolCorrect) / zeroCheck(iterValCal->second.nSolTotal);
		float fPrdRatio = static_cast<float>(iterValCal->second.nPrdCorrect) / zeroCheck(iterValCal->second.nPrdTotal);
		strValOutOrg += to_string(fOrgRatio) + "\t";
		strValOutPrd += to_string(fPrdRatio) + "\t";
	}
	float fOrgTotalRatio = static_cast<float>(nOrgSum) / nTotalSum;
	float fPrdTotalRatio = static_cast<float> (nPrdSum) / nTotalSum;
	strValOutOrg += to_string(fOrgTotalRatio) + "\n";
	strValOutPrd += to_string(fPrdTotalRatio) + "\n";

	outFileCal << strValOutOrg;
	outFileCal << strValOutPrd;
}
void CValidateModel::testValidateSetMeasure()
{
	std::ifstream inFile;
	inFile.open(m_ValidateListFile);
	if (!inFile.is_open())
		return;

	char buf[1024];

	while (!inFile.getline(buf, sizeof(buf)).eof())
	{
		string line = string(buf);
		vector<string> vstrline = CStringParse::splitString(line, ',');

		stValSetMeasure OneSet;
		OneSet.strPath = vstrline[0];
		for (int i = 1; i < vstrline.size(); i++)
			OneSet.fOrigin.push_back(atof(vstrline[i].c_str()));

		m_vValidateMeasureList.push_back(OneSet);
	}

	string strHeader = "file_name,";
	for (int i = 0; i < m_label.size(); i++)
	{
		strHeader += m_label[i] + "_Org,";
	}
	for (int i = 0; i < m_label.size(); i++)
	{
		strHeader += m_label[i] + "_Prd,";
	}


	ofstream outFile(m_strSavePredictResult, ios::out); //파일을 연다, 없으면 생성
	ofstream outFileCal(m_strSaveValidate, ios::out);
	outFile << strHeader + "\n";


	const int nImgSize = m_nBatchSize / m_nOverSample;
	vector<cv::Mat> vImg;
	int nCnt = 0;
	for (int i = 0; i < m_vValidateMeasureList.size(); i++)
	{
		clock_t st;
		clock_t et;
		st = clock();
		m_vValidateMeasureList[i].strPath = m_ImgRoot + m_vValidateMeasureList[i].strPath;

		cv::Mat img = cv::imread(m_vValidateMeasureList[i].strPath);
		if (img.empty())
		{
			inFile.close();
			remove(m_vValidateMeasureList[i].strPath.c_str());
			std::cout << "---------- Not Exist Image " << m_vValidateMeasureList[i].strPath << " ----------" << std::endl;
			continue;
		}

		cv::Size mean_size;
		mean_size.width = m_classifier.getmean().cols;
		mean_size.height = m_classifier.getmean().rows;

		cv::resize(img, img, mean_size);

		vImg.push_back(img);
		nCnt++;
		if (nCnt < nImgSize)
		{
			et = clock();
			double dTime = (double)(et - st) / CLOCKS_PER_SEC;
			cout << "process time one image : " + to_string(dTime) << endl;
			continue;
		}

		///////////////////////////////////////////////////
		vector<float> prediction = m_classifier.MeasureOverSample(vImg[0],  10);
		vImg.clear();
		nCnt = 0;
		for (int m_num = 0; m_num < prediction.size(); m_num++)
			m_vValidateMeasureList[i].fMeasure.push_back(prediction[m_num]);

		string strOneLineOut;
		strOneLineOut = m_vValidateMeasureList[i].strPath + ",";
		for (int mnum = 0; mnum < m_label.size(); mnum++)
			strOneLineOut += to_string(m_vValidateMeasureList[i].fOrigin[mnum]) + ",";
		for (int mnum = 0; mnum < m_label.size(); mnum++)
			strOneLineOut += to_string(m_vValidateMeasureList[i].fMeasure[mnum]) + ",";

		outFile << strOneLineOut + "\n";
		et = clock();
		double dTime = (double)(et - st) / CLOCKS_PER_SEC;
		cout << "process time one image : " + to_string(dTime) << endl;
	}
}

void CValidateModel::testValidateSet()
{
	std::ifstream inFile;
	inFile.open(m_ValidateListFile);
	if (!inFile.is_open())
		return;

	char buf[1024];

	while (!inFile.getline(buf, sizeof(buf)).eof())
	{
		string line = string(buf);
		vector<string> vstrline = CStringParse::splitString(line, ',');

		stValSet OneSet;
		OneSet.strPath = vstrline[0];
		int nSolLabel = atoi(vstrline[1].c_str());
		OneSet.strOrginCode = m_label[nSolLabel];
		m_vValidateList.push_back(OneSet);
	}

	string strHeader = "file_name\treal_code\tpredict_code\tpercent";
	std::ifstream infilePrd;
	infilePrd.open(m_strSavePredictResult);
	if (infilePrd.is_open())
	{
		int idx = 0;
		while (!infilePrd.getline(buf, sizeof(buf)).eof())
		{
			string line = string(buf);
			if (line == strHeader) continue;

			vector<string> vstrline = CStringParse::splitString(line, '\t');
			string strImgPath = m_ImgRoot + m_vValidateList[idx].strPath;
			if (vstrline[0] == strImgPath&& vstrline.size() == 4)
			{
				m_vValidateList[idx].strOrginCode = vstrline[1];
				m_vValidateList[idx].strPredictCode = vstrline[2];
				m_vValidateList[idx].fProb = atof(vstrline[3].c_str());
			}
			idx++;
		}
		infilePrd.close();
	}

	ofstream outFile(m_strSavePredictResult, ios::out); //파일을 연다, 없으면 생성
	ofstream outFileCal(m_strSaveValidate, ios::out);
	outFile << strHeader + "\n";

	struct stValCal
	{
		int nSolTotal;
		int nSolCorrect;
		int nPrdTotal;
		int nPrdCorrect;
	};
	map<string, stValCal>  mapValCal;
	map<string, stValCal>::iterator iterValCal;

	const int nImgSize = m_nBatchSize / m_nOverSample;
	vector<cv::Mat> vImg;
	int nCnt = 0;
	for (int i = 0; i < m_vValidateList.size(); i++)
	{
		clock_t st;
		clock_t et;
		st = clock();
		m_vValidateList[i].strPath = m_ImgRoot + m_vValidateList[i].strPath;
		if (m_vValidateList[i].strPredictCode == "")
		{
			cv::Mat img = cv::imread(m_vValidateList[i].strPath);
			if (img.empty())
			{
				inFile.close();
				remove(m_vValidateList[i].strPath.c_str());
				std::cout << "---------- Not Exist Image " << m_vValidateList[i].strPath << " ----------" << std::endl;
				continue;
			}

			cv::Size mean_size;
			mean_size.width = m_classifier.getmean().cols;
			mean_size.height = m_classifier.getmean().rows;

			cv::resize(img, img, mean_size);

			vImg.push_back(img);
			nCnt++;
			if (nCnt < nImgSize)
			{
				et = clock();
				double dTime = (double)(et - st) / CLOCKS_PER_SEC;
				cout << "process time one image : " + to_string(dTime) << endl;
				continue;
			}

			///////////////////////////////////////////////////

			vector< vector<Prediction> >  vPredictions;
			if (m_classifier.isFcn())
			{
				vPredictions = m_classifier.ClassifyFcnBatch(vImg, 1);
			}
			else
			{
				vector<Prediction> prediction = m_classifier.ClassifyOverSample(vImg[0], 1, 10);
				vPredictions.push_back(prediction);
			}
			vImg.clear();
			nCnt = 0;
			m_vValidateList[i].strPredictCode = vPredictions[0][0].first;
			m_vValidateList[i].fProb = vPredictions[0][0].second;
		}
		else
		{
			stValSet tempSet;
			tempSet = m_vValidateList[i];
			mapValCal[tempSet.strOrginCode].nSolTotal++;
			mapValCal[tempSet.strPredictCode].nPrdTotal++;
			if (tempSet.strPredictCode == tempSet.strOrginCode)
			{
				mapValCal[tempSet.strPredictCode].nSolCorrect++;
				mapValCal[tempSet.strPredictCode].nPrdCorrect++;
			}
			string strOneLineOut;
			strOneLineOut = tempSet.strPath + "\t";
			strOneLineOut += tempSet.strOrginCode + "\t";
			strOneLineOut += tempSet.strPredictCode + "\t";
			strOneLineOut += to_string(tempSet.fProb) + "\n";
			outFile << strOneLineOut;
			continue;
		}

		for (int p = 0; p < nImgSize; p++)
		{
			stValSet tempSet;
			tempSet = m_vValidateList[i - (nImgSize - 1) + p];

			mapValCal[tempSet.strOrginCode].nSolTotal++;
			mapValCal[tempSet.strPredictCode].nPrdTotal++;
			if (tempSet.strPredictCode == tempSet.strOrginCode)
			{
				mapValCal[tempSet.strPredictCode].nSolCorrect++;
				mapValCal[tempSet.strPredictCode].nPrdCorrect++;
			}

			string strOneLineOut;
			strOneLineOut = tempSet.strPath + "\t";
			strOneLineOut += tempSet.strOrginCode + "\t";
			strOneLineOut += tempSet.strPredictCode + "\t";
			strOneLineOut += to_string(tempSet.fProb) + "\n";
			outFile << strOneLineOut;
		}

		et = clock();
		double dTime = (double)(et - st) / CLOCKS_PER_SEC;
		cout << "process time one image : " + to_string(dTime) << endl;
	}

	string strValHeader = "model\tPrdOrg\t";
	for (iterValCal = mapValCal.begin(); iterValCal != mapValCal.end(); iterValCal++)
		strValHeader += iterValCal->first + "\t";

	strValHeader += "total\n";
	outFileCal << strValHeader;


	string strModel = CStringParse::getLastString(m_trained_file, '/');
	string strValOutOrg = strModel + "\tOrg\t";
	string strValOutPrd = strModel + "\tPrd\t";
	int nOrgSum = 0;
	int nPrdSum = 0;
	int nTotalSum = 0;
	for (iterValCal = mapValCal.begin(); iterValCal != mapValCal.end(); iterValCal++)
	{
		nTotalSum += iterValCal->second.nSolTotal;
		nOrgSum += iterValCal->second.nSolCorrect;
		nPrdSum += iterValCal->second.nPrdCorrect;
		float fOrgRatio = static_cast<float>(iterValCal->second.nSolCorrect) / zeroCheck(iterValCal->second.nSolTotal);
		float fPrdRatio = static_cast<float>(iterValCal->second.nPrdCorrect) / zeroCheck(iterValCal->second.nPrdTotal);
		strValOutOrg += to_string(fOrgRatio) + "\t";
		strValOutPrd += to_string(fPrdRatio) + "\t";
	}
	float fOrgTotalRatio = static_cast<float>(nOrgSum) / nTotalSum;
	float fPrdTotalRatio = static_cast<float> (nPrdSum) / nTotalSum;
	strValOutOrg += to_string(fOrgTotalRatio) + "\n";
	strValOutPrd += to_string(fPrdTotalRatio) + "\n";

	outFileCal << strValOutOrg;
	outFileCal << strValOutPrd;
}

CValidateModel::CValidateModel(string strModelRoot, string strTrainedFile, string strImgRoot, string strTestListFile, int gpuNum, string strType)
{
	InitModel(strModelRoot, strTrainedFile, strImgRoot, strTestListFile, gpuNum, false);
	testValidateSetMeasure();
}

CValidateModel::CValidateModel(string strModelRoot, string strTrainedFile, string strImgRoot, string strTestListFile, int gpuNum)
{
	InitModel(strModelRoot, strTrainedFile, strImgRoot, strTestListFile, gpuNum, false);
	testValidateSet();
}

CValidateModel::CValidateModel(string strModelRoot, string strTrainedFile, string strImgRoot, string strTestListFile, int gpuNum, bool bIntel)
{
	InitModel(strModelRoot, strTrainedFile, strImgRoot, strTestListFile, gpuNum, bIntel);
	testValidateSetIntel();
}


CValidateModel::~CValidateModel()
{
}
