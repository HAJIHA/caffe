#include "caffeClassifier.h"
#include "../cpp_classification_cervical_cancer/StringParse.h"
#include "ValidateModel.h"
#include "../cpp_classification_cervical_cancer/ManageDirectory.h"

using namespace cv;

void CValidateModel::InitModel(string strModelRoot, string strTrainedFile, string strImgRoot, string strTestListFile, int gpuNum)
{
    m_nBatchSize = 10;
    m_nOverSample = 10;

  string strRoot = CManageDirectory::checkNAddSlush(strModelRoot);

  m_trained_file = strRoot + strTrainedFile;
  m_mean_file = strRoot + "test.binaryproto";
  m_label_file = strRoot + "synset_words.txt";
  m_ValidateListFile = strRoot + strTestListFile;
  vector<string> vTraindFile = CStringParse::splitString(strTrainedFile, '.');
  vector<string> vTrainedFileWord = CStringParse::splitString(vTraindFile[0], '_');
  string strPredictResultFile = vTraindFile[0] + "_PrdResult.txt";

  bool bFcn = false;
  string strDeploy;
  for (int i = 0; i < vTrainedFileWord.size(); i++)
  {
    if (vTrainedFileWord[i] == "iter")
    {
      strDeploy += "deploy.prototxt";
      break;
    }
    else
    {
      strDeploy += vTrainedFileWord[i];
      strDeploy += "_";
    }
  }
  m_model_file = strRoot + strDeploy;

  m_classifier.loadModel(m_model_file, m_trained_file, m_mean_file, m_label_file, true, m_nBatchSize, gpuNum);

  m_strSavePredictResult = strRoot + strPredictResultFile;
  m_ImgRoot = strImgRoot;
  m_label = m_classifier.getLabelList();
}


void CValidateModel::testValidateSet()
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
    vector<string> vstrline = CStringParse::splitString(line, '\t');

	vector<string> vstrLabel = CStringParse::splitString(vstrline[1], ' ');
    stValSetPlanet OneSet;
    OneSet.strPath = vstrline[0];
	OneSet.vstrCode = vstrLabel;
	m_vValidateList.push_back(OneSet);
  }

  string strHeader = "image_name,tags";
  ofstream outFile(m_strSavePredictResult, ios::out); 
  outFile << strHeader + "\n";

  const int nImgSize = m_nBatchSize / m_nOverSample;
  int nCnt = 0;
  for (int i = 0; i < m_vValidateList.size(); i++)
  {
    clock_t st;
    clock_t et;
    st = clock();
	//여기서부터 휴가 복귀후 코딩
		
	string strJpgPath;
	string strTifPath;
	string subFolder;
	if (m_vValidateList[i].strPath.substr(0, 4) == "test")
	{
		subFolder = "test";
	}
	else
	{
		subFolder = "train";
	}
	strJpgPath = m_ImgRoot + subFolder + "-jpg/" + m_vValidateList[i].strPath + ".jpg";
	strTifPath = m_ImgRoot + subFolder + "-tif/" + m_vValidateList[i].strPath + ".tif";

	vector<string> vFile;
	vFile.push_back(strJpgPath);
	vFile.push_back(strTifPath);

	  vector<cv::Mat> vImgMerge;
	  cv::Mat merge_img;
	  for (int subidx = 0; subidx < vFile.size(); subidx++)
	  {
		  vector<cv::Mat> vImgSplit;
		  cv::Mat cv_img = cv::imread(vFile[subidx], CV_LOAD_IMAGE_UNCHANGED);
		  cv::split(cv_img, vImgSplit);
		  for (int j = 0; j < vImgSplit.size(); j++)
		  {
			  cv::Mat fImg;
			  int d = vImgSplit[j].depth();
			  vImgSplit[j].convertTo(fImg, CV_32FC1);
			  vImgMerge.push_back(fImg);
		  }
	  }
	  if (vImgMerge.size() != 7)
	  {
		  LOG(WARNING) << " Not equal channels  : first img path " << vFile[0];
		  continue;
	  }

	  //for (int chidx = 0; chidx < vImgMerge.size(); chidx++)
	  //{
		 // Mat dispImg;
		 // double minVal, maxVal;
		 // minMaxLoc(vImgMerge[chidx], &minVal, &maxVal); //find minimum and maximum intensities
		 // vImgMerge[chidx].convertTo(dispImg, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
		 // imshow("cv_img_" + to_string(chidx), dispImg);
	  //}
	  //waitKey(0);

	  cv::merge(vImgMerge, merge_img);

      nCnt++;
      if (nCnt < nImgSize)
      {
        et = clock();
        double dTime = (double)(et - st) / CLOCKS_PER_SEC;
        cout << "process time one image : " + to_string(dTime) << endl;
        continue;
      }

      ///////////////////////////////////////////////////
      vector<Prediction> prediction = m_classifier.ClassifyOverSample(merge_img,17,10);
      nCnt = 0;


	  float fThresh = 0.0001;
	  string strOneLineOut;
	  strOneLineOut = m_vValidateList[i].strPath + "\t";

	  for (int prdidx = 0; prdidx < prediction.size(); prdidx++)
	  {
		  if (prediction[prdidx].second < fThresh)
		  {
			  strOneLineOut += "\n";
			  break;
		  }
		  else
		  {
			  strOneLineOut += prediction[prdidx].first + " ";
		  }
	  }
	  outFile << strOneLineOut;

    et = clock();
    double dTime = (double)(et - st) / CLOCKS_PER_SEC;
    cout << "process time one image : " + to_string(dTime) << endl;
  }
}

CValidateModel::CValidateModel(string strModelRoot, string strTrainedFile, string strImgRoot, string strTestListFile, int gpuNum)
{
  InitModel(strModelRoot, strTrainedFile, strImgRoot, strTestListFile, gpuNum);
  testValidateSet();
}

CValidateModel::~CValidateModel()
{
}
