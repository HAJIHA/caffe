#pragma once
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
class CaffeClassifier
{
public:
	CaffeClassifier();
	~CaffeClassifier();
	CaffeClassifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file, const bool use_GPU,
		const int batch_size,
		const int gpuNum);

	void loadModel(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file, const bool use_GPU,
		const int batch_size,
		const int gpuNum);
	vector< vector<Prediction> > ClassifyBatch(const vector< cv::Mat > imgs, int num_classes);
	vector<Prediction> ClassifyOverSample(const cv::Mat img, int num_classes, int num_overSample);
	vector< vector< Prediction > > ClassifyFcnBatch(const vector<cv::Mat> img, int num_classes);
	//vector<vector<Prediction>> ClassifyOverSample(const vector<cv::Mat> vImg, int num_classes, int num_overSample);
	cv::Mat getmean();
	std::vector<string> getLabelList();
	void setFcn(bool bFcn);
	bool isFcn();
	void setSelelctMinValueLabel(vector<string> strCode);
public:
	//	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
	void PreprocessBatch(const vector<cv::Mat> imgs, std::vector< std::vector<cv::Mat> >* input_batch);
	void PreprocessBatchNonSub(const vector<cv::Mat> imgs, std::vector< std::vector<cv::Mat> >* input_batch);
	void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);
	vector<cv::Mat> OverSample(const vector<cv::Mat> vImgs, int size);
	vector<cv::Mat> OverSample(const cv::Mat img, int size);
	vector<cv::Mat> OverSampleSub(const cv::Mat img, int nOverSample, cv::Mat mean);

	vector< float >  PredictBatch(const vector< cv::Mat > imgs);
	vector< float >  PredictBatchNonSub(const vector< cv::Mat > imgs);
	// num, channel, height, width
	vector< vector< vector< vector< float > > > > PredictFcnBatch(const vector< cv::Mat > imgs);
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);
	bool isMinValueCode(string code);


	vector<string> m_vStrMinValueCode;

private:
	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int batch_size_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;

	string m_modelfile;
	string m_trained_file;
	string m_mean_file;
	string m_label_file;
	int m_nUseGpuNum;
	bool m_bFcn;
};

