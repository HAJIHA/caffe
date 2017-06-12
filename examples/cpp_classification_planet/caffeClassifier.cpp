#include "caffeclassifier.h"
using namespace std;
using namespace cv;

CaffeClassifier::CaffeClassifier()
{

}

CaffeClassifier::~CaffeClassifier()
{

}

CaffeClassifier::CaffeClassifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file,
	const bool use_GPU,
	const int batch_size,
	const int gpuNum) {
	loadModel(model_file, trained_file, mean_file, label_file, use_GPU, batch_size, gpuNum);
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}


void CaffeClassifier::loadModel(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file, const bool use_GPU,const int batch_size, const int gpuNum)
{
	if (model_file.compare(m_modelfile) == 0 &&
		trained_file.compare(m_trained_file) == 0 &&
		mean_file.compare(m_mean_file) == 0 &&
		label_file.compare(m_label_file) == 0 &&
		m_nUseGpuNum == gpuNum
		)
	{
		return;
	}
	else
	{
		m_modelfile = model_file;
		m_trained_file = trained_file;
		m_mean_file = mean_file;
		m_label_file = label_file;
		m_nUseGpuNum = gpuNum;
		labels_.clear();
	}

	if (use_GPU)
	{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(m_nUseGpuNum);
	}
	else
	{
		Caffe::set_mode(Caffe::CPU);
	}

	/* Set batchsize */
	batch_size_ = batch_size;

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	//CHECK(num_channels_ == 3 || num_channels_ == 1)
	//	<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}


/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

std::vector<string> CaffeClassifier::getLabelList()
{
	return labels_;
}

vector<cv::Mat> CaffeClassifier::OverSamplePlanet(const cv::Mat img, int nOverSample, cv::Mat mean)
{
	cv::Mat resultImg = img;
	int srcW = resultImg.cols;
	int srcH = resultImg.rows;
	if (srcH > srcW)
	{
		cv::transpose(resultImg, resultImg);
		cv::flip(resultImg, resultImg, 1);
		srcW = resultImg.cols;
		srcH = resultImg.rows;
	}

	cv::Mat sample_float;
	if (num_channels_ == 3)
		resultImg.convertTo(sample_float, CV_32FC3);
	else if(num_channels_ == 7)
		resultImg.convertTo(sample_float, CV_32FC(7));
	else
		resultImg.convertTo(sample_float, CV_32FC1);


	Mat sample_normalized;
	subtract(sample_float, mean, sample_normalized);


	//vector<Mat> vImgSplit;
	//vector<Mat> vImgMean;
	//vector<Mat> vImgSrc;
	//split(img, vImgSrc);
	//split(mean, vImgMean);
	//split(sample_normalized, vImgSplit);

	//for (int chidx = 0; chidx < vImgSplit.size(); chidx++)
	//{
	//	Mat dispImg;
	//	double minVal, maxVal;
	//	minMaxLoc(vImgSrc[chidx], &minVal, &maxVal); //find minimum and maximum intensities
	//	vImgSrc[chidx].convertTo(dispImg, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
	//	imshow("src_" + to_string(chidx), dispImg);

	//	minMaxLoc(vImgMean[chidx], &minVal, &maxVal); //find minimum and maximum intensities
	//	vImgMean[chidx].convertTo(dispImg, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
	//	imshow("mean_" + to_string(chidx), dispImg);

	//	minMaxLoc(vImgSplit[chidx], &minVal, &maxVal); //find minimum and maximum intensities
	//	vImgSplit[chidx].convertTo(dispImg, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
	//	imshow("sample_normalized_" + to_string(chidx), dispImg);
	//}
	//waitKey(0);


	return OverSample(sample_normalized, nOverSample);
}

vector<cv::Mat> CaffeClassifier::OverSample(const cv::Mat img, int size)
{

	int srcW = img.cols;
	int srcH = img.rows;
	int tarW = input_geometry_.width;
	int tarH = input_geometry_.height;


	int cenHOffset = (srcH - tarH) / 2;
	int cenWOffset = (srcW - tarW) / 2;

	int rightWOffset = (srcW - tarW) - 1;
	int bottomHOffset = (srcH - tarH) - 1;

	cv::Mat flipImg;
	cv::flip(img, flipImg, 1);
    cv::Mat resizeImg;
    cv::resize(img, resizeImg, cv::Size(srcW*1.5, srcH*1.5));
    int ResizeCenHOffset = (resizeImg.rows - tarH) / 2;
    int ResizecenWOffset = (resizeImg.cols - tarW) / 2;

	vector<cv::Rect> vRect;
	vRect.push_back(cv::Rect(cenWOffset, cenHOffset, tarW, tarH));
	vRect.push_back(cv::Rect(0, 0, tarW, tarH));
	vRect.push_back(cv::Rect(rightWOffset, 0, tarW, tarH));
	vRect.push_back(cv::Rect(0, bottomHOffset, tarW, tarH));
	vRect.push_back(cv::Rect(rightWOffset, bottomHOffset, tarW, tarH));


    cv::Rect RectReizeCenter = cv::Rect(ResizecenWOffset, ResizeCenHOffset, tarW, tarH);

	vector<cv::Mat> vImgs;
    if (size == 1)
    {
      vImgs.push_back(img(vRect[0]));
      return vImgs;
    }

    // idea. centor crop multi scale add
    if (size == 6)
    {
      for (int i = 0; i < vRect.size(); i++)
      {
        vImgs.push_back(img(vRect[i]));
      }
      vImgs.push_back(resizeImg(RectReizeCenter));
      return vImgs;
    }

	for (int i = 0; i < vRect.size(); i++)
	{
		vImgs.push_back(img(vRect[i]));
		if (size == 10)
		{
			vImgs.push_back(flipImg(vRect[i]));
		}
	}

	return vImgs;
}


vector<cv::Mat> CaffeClassifier::OverSample(const vector<cv::Mat> vImgs, int size)
{
	int srcW = vImgs[0].cols;
	int srcH = vImgs[0].rows;
	int tarW = input_geometry_.width;
	int tarH = input_geometry_.height;


	int cenHOffset = (srcH - tarH) / 2;
	int cenWOffset = (srcW - tarW) / 2;

	int rightWOffset = (srcW - tarW) - 1;
	int bottomHOffset = (srcH - tarH) - 1;

	vector<cv::Mat> vRetImgs;
	vector<cv::Rect> vRect;
	vRect.push_back(cv::Rect(cenWOffset, cenHOffset, tarW, tarH));
	vRect.push_back(cv::Rect(0, 0, tarW, tarH));
	vRect.push_back(cv::Rect(rightWOffset, 0, tarW, tarH));
	vRect.push_back(cv::Rect(0, bottomHOffset, tarW, tarH));
	vRect.push_back(cv::Rect(rightWOffset, bottomHOffset, tarW, tarH));


	for (int i = 0; i < vImgs.size(); i++)
	{
		cv::Mat flipImg;
		cv::flip(vImgs[i], flipImg, 1);
		for (int r = 0; r < vRect.size(); r++)
		{
			vRetImgs.push_back(vImgs[i](vRect[r]));
			if (size == 10)
			{
				vRetImgs.push_back(flipImg(vRect[r]));
			}
		}
	}

	return vRetImgs;
}

vector<Prediction> CaffeClassifier::ClassifyOverSample(const cv::Mat img, int num_classes, int num_overSample)
{
	vector<cv::Mat> vImgs = OverSamplePlanet(img, num_overSample,mean_);

	std::vector<float> output_batch = PredictBatchNonSub(vImgs);
	std::vector<float> output;

	for (int i = 0; i < labels_.size(); i++)
		output.push_back(output_batch[i]);

	for (int i = labels_.size(); i < output_batch.size(); i++)
	{
		int idx = i% labels_.size();
		output[idx] += output_batch[i];

	}

	std::vector<Prediction> prediction_single;
	std::vector<int> maxN = Argmax(output, num_classes);
	for (int i = 0; i < num_classes; ++i)
	{
		int idx = maxN[i];
		prediction_single.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return prediction_single;
}

vector<Prediction> CaffeClassifier::ClassifyIntel(const cv::Mat img)
{
  int nOverSample = 72;
  const int maxBatch = 24;
  //vector<cv::Mat> vImgs = OverSampleIntel(img, nOverSample);
  vector<cv::Mat> vImgs = OverSamplePlanet(img, nOverSample,mean_);
  vector< vector<cv::Mat> > vvImg;
  for (int i = 0; i < nOverSample / maxBatch ; i++)
  {
	  vector<cv::Mat> vImg;
	  for (int j = 0; j < maxBatch; j++)
	  {
		  vImg.push_back(vImgs[i*maxBatch + j]);
	  }
	  vvImg.push_back(vImg);
  }

  vector< vector<float> > vOutput_batch;
  for (int i = 0; i < nOverSample / maxBatch; i++)
  {
	  //vector<float> output = PredictBatch(vvImg[i]);
	  vector<float> output = PredictBatchNonSub(vvImg[i]);
	  vOutput_batch.push_back(output);
  }

  std::vector<float> output_batch;
  for (int i = 0; i < nOverSample / maxBatch; i++)
  {
	  for (int j = 0; j < vOutput_batch[i].size(); j++)
	  {
		  output_batch.push_back(vOutput_batch[i][j]);
	  }
  }


  // idea. multi calssifier max prob value prior 
  std::vector<Prediction> prediction_single;
  //int maxidx = 0;
  //float maxprob = 0.0;
  //for (int i = 0; i < output_batch.size(); i++)
  //{
  //  if (maxprob < output_batch[i])
  //  {
  //    maxprob = output_batch[i];
  //    maxidx = i;
  //  }
  //}
  //int maxProbBatchS = maxidx / 3;
  //for (int i = maxProbBatchS*3; i < maxProbBatchS*3 + 3; ++i)
  //{
  //  prediction_single.push_back(std::make_pair(labels_[i % 3], output_batch[i]));
  //}

  // average prob
  std::vector<float> output;
  for (int i = 0; i < labels_.size(); i++)
    output.push_back(output_batch[i]);

  for (int i = labels_.size(); i < output_batch.size(); i++)
  {
    int idx = i% labels_.size();
    output[idx] += output_batch[i];
  }


  for (int i = 0; i < output.size(); i++)
  {
    prediction_single.push_back(std::make_pair(labels_[i], output[i] / nOverSample));
  }

  return prediction_single;
}

/* Load the mean file in binaryproto format. */
void CaffeClassifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);
	//cv::Mat cMean;
	//mean.convertTo(cMean, CV_8UC3);

	//cv::imshow("mean", cMean);
	//cv::waitKey(0);
	
	mean.copyTo(mean_);
	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	//cv::Scalar channel_mean = cv::mean(mean);
	//mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

vector< vector< vector< vector< float > > > >  CaffeClassifier::PredictFcnBatch(const vector< cv::Mat > imgs)
{
  Blob<float>* input_layer = net_->input_blobs()[0];

  input_layer->Reshape(batch_size_, num_channels_,
    input_geometry_.height,
    input_geometry_.width);

  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector< std::vector<cv::Mat> > input_batch;
  WrapBatchInputLayer(&input_batch);

  PreprocessBatch(imgs, &input_batch);
  net_->Forward();

  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();


  vector< vector< vector< vector< float > > > > outblobCopy;

  for (int b = 0; b < output_layer->num(); b++)
  {
    //label
    vector< vector< vector< float> > > vC;
    for (int c = 0; c < output_layer->channels(); c++)
    {
      vector< vector< float> > vH;
      for (int h = 0; h < output_layer->height(); h++)
      {
        vector< float> vW;
        for (int w = 0; w < output_layer->width(); w++)
        {
          vW.push_back(output_layer->data_at(b, c, h, w));
        }
        vH.push_back(vW);
      }
      vC.push_back(vH);
    }
    outblobCopy.push_back(vC);
  }
  return outblobCopy;
}

vector< float >  CaffeClassifier::PredictBatchNonSub(const vector< cv::Mat > imgs)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	input_layer->Reshape(batch_size_, num_channels_,
		input_geometry_.height,
		input_geometry_.width);

	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector< std::vector<cv::Mat> > input_batch;
	WrapBatchInputLayer(&input_batch);

	PreprocessBatchNonSub(imgs, &input_batch);
	net_->Forward();
	//net_->ForwardPrefilled();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels()*imgs.size();
	return std::vector<float>(begin, end);
}

std::vector< float >  CaffeClassifier::PredictBatch(const vector< cv::Mat > imgs) 
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	input_layer->Reshape(batch_size_, num_channels_,
		input_geometry_.height,
		input_geometry_.width);

	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector< std::vector<cv::Mat> > input_batch;
	WrapBatchInputLayer(&input_batch);

	PreprocessBatch(imgs, &input_batch);
	net_->Forward();
	//net_->ForwardPrefilled();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels()*imgs.size();
	return std::vector<float>(begin, end);
}


void CaffeClassifier::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	int num = input_layer->num();
	float* input_data = input_layer->mutable_cpu_data();
	for (int j = 0; j < num; j++){
		vector<cv::Mat> input_channels;
		for (int i = 0; i < input_layer->channels(); ++i){
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels.push_back(channel);
			input_data += width * height;
		}
		input_batch->push_back(vector<cv::Mat>(input_channels));
	}
	//cv::imshow("bla", input_batch->at(1).at(0));
	//cv::waitKey(1);
}

void CaffeClassifier::PreprocessBatchNonSub(const vector<cv::Mat> imgs, std::vector< std::vector<cv::Mat> >* input_batch)
{
	for (int i = 0; i < imgs.size(); i++)
	{
		cv::Mat img = imgs[i];
		std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

		/* Convert the input image to the input image format of the network. */
		cv::Mat sample;
		if (img.channels() == 3 && num_channels_ == 1)
			cv::cvtColor(img, sample, CV_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cv::cvtColor(img, sample, CV_BGRA2GRAY);
		else if (img.channels() == 4 && num_channels_ == 3)
			cv::cvtColor(img, sample, CV_BGRA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cv::cvtColor(img, sample, CV_GRAY2BGR);
		else
			sample = img;

		cv::Mat sample_resized;
		if (sample.size() != input_geometry_)
			cv::resize(sample, sample_resized, input_geometry_);
		else
			sample_resized = sample;

		cv::Mat sample_float;
		if (num_channels_ == 3)
			sample_resized.convertTo(sample_float, CV_32FC3);
		else if (num_channels_ == 1)
			sample_resized.convertTo(sample_float, CV_32FC1);
		else if (num_channels_ == 7)
			sample_resized.convertTo(sample_float, CV_32FC(7));

		/* This operation will write the separate BGR planes directly to the
		* input layer of the network because it is wrapped by the cv::Mat
		* objects in input_channels. */
		cv::split(sample_float, *input_channels);

		//        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		//              == net_->input_blobs()[0]->cpu_data())
		//          << "Input channels are not wrapping the input layer of the network.";
	}
}

void CaffeClassifier::PreprocessBatch(const vector<cv::Mat> imgs,	std::vector< std::vector<cv::Mat> >* input_batch)
{
	for (int i = 0; i < imgs.size(); i++)
	{
		cv::Mat img = imgs[i];
		std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

		/* Convert the input image to the input image format of the network. */
		cv::Mat sample;
		if (img.channels() == 3 && num_channels_ == 1)
			cv::cvtColor(img, sample, CV_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cv::cvtColor(img, sample, CV_BGRA2GRAY);
		else if (img.channels() == 4 && num_channels_ == 3)
			cv::cvtColor(img, sample, CV_BGRA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cv::cvtColor(img, sample, CV_GRAY2BGR);
		else
			sample = img;

		cv::Mat sample_resized;
		if (sample.size() != input_geometry_)
			cv::resize(sample, sample_resized, input_geometry_);
		else
			sample_resized = sample;

		cv::Mat sample_float;
		if (num_channels_ == 3)
			sample_resized.convertTo(sample_float, CV_32FC3);
		else if(num_channels_ == 1)
			sample_resized.convertTo(sample_float, CV_32FC1);
		else if (num_channels_ == 7)
			sample_resized.convertTo(sample_float, CV_32FC(7));

		cv::Mat sample_normalized;
		cv::subtract(sample_float, mean_, sample_normalized);

		/* This operation will write the separate BGR planes directly to the
		* input layer of the network because it is wrapped by the cv::Mat
		* objects in input_channels. */
		cv::split(sample_normalized, *input_channels);

		//        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		//              == net_->input_blobs()[0]->cpu_data())
		//          << "Input channels are not wrapping the input layer of the network.";
	}
}