#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <math.h>
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

inline double radian2degree(double radian) {
	return radian * (180.0 /M_PI);
}
inline double degree2radian(double degree) {
	return degree * (M_PI / 180);
}

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
	Blob<Dtype>* transformed_blob) {
	// If datum is encoded, decoded and transform the cv::image.
	if (datum.encoded()) {
		CHECK(!(param_.force_color() && param_.force_gray()))
			<< "cannot set both force_color and force_gray";
		cv::Mat cv_img;
		if (param_.force_color() || param_.force_gray()) {
			// If force_color then decode in color otherwise decode in gray.
			cv_img = DecodeDatumToCVMat(datum, param_.force_color());
		}
		else {
			cv_img = DecodeDatumToCVMatNative(datum);
		}
		// Transform the cv::image into blob.
		return Transform(cv_img, transformed_blob);
	}
	else {
		if (param_.force_color() || param_.force_gray()) {
			LOG(ERROR) << "force_color and force_gray only for encoded datum";
}
		}

	cv::Mat cv_img = DatumToCVMat(datum);
	Transform(cv_img, transformed_blob);

	//const int crop_size = param_.crop_size();
	//const int datum_channels = datum.channels();
	//const int datum_height = datum.height();
	//const int datum_width = datum.width();

	//// Check dimensions.
	//const int channels = transformed_blob->channels();
	//const int height = transformed_blob->height();
	//const int width = transformed_blob->width();
	//const int num = transformed_blob->num();

	//CHECK_EQ(channels, datum_channels);
	//CHECK_LE(height, datum_height);
	//CHECK_LE(width, datum_width);
	//CHECK_GE(num, 1);

	//if (crop_size) {
	//	CHECK_EQ(crop_size, height);
	//	CHECK_EQ(crop_size, width);
	//}
	//else {
	//	CHECK_EQ(datum_height, height);
	//	CHECK_EQ(datum_width, width);
	//}

	//Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	//Transform(datum, transformed_data);
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& img,
	Blob<Dtype>* transformed_blob) {

	const bool display = param_.display();
	if (display)
		cv::imshow("orgin", img);
	cv::Mat cv_img;
	//img.copyTo(cv_img);
	cv::Mat pre_img = img;
	// pre_transform
	const bool transpose_for_unify_hor = param_.transpose_for_unify_hor();
	if (transpose_for_unify_hor)
	{
		if (pre_img.cols < pre_img.rows)
			cv::transpose(pre_img, pre_img);
	}

	const bool use_pre_resize = param_.use_pre_resize();
	const int rand_omit_offset = param_.rand_omit_offset();
	const int pre_resize_width = param_.pre_resize_width();
	const int pre_resize_height = param_.pre_resize_height();

	if (use_pre_resize)
	{
		if (rand_omit_offset == 0)
		{
			cv::resize(pre_img, cv_img, cv::Size(pre_resize_width, pre_resize_height));
		}
		else
		{
			const int omit_rand = Rand(rand_omit_offset);
			const int dir = Rand(4);
			cv::Rect pre_roi;
			if (dir == 0)
			{
				pre_roi = cv::Rect(omit_rand, 0, pre_img.cols - omit_rand, pre_img.rows);
			}
			else if (dir == 1)
			{
				pre_roi = cv::Rect(0, omit_rand, pre_img.cols, pre_img.rows - omit_rand);
			}
			else if (dir == 2)
			{
				pre_roi = cv::Rect(0, 0, pre_img.cols, pre_img.rows - omit_rand);
			}
			else
			{
				pre_roi = cv::Rect(0, 0, pre_img.cols - omit_rand, pre_img.rows);
			}

			cv::Mat cv_resize_img = pre_img(pre_roi);
			const int method = Rand(4);
			cv::resize(cv_resize_img, cv_img, cv::Size(pre_resize_width, pre_resize_height));
			if (display)
				cv::imshow("PreResize", cv_img);
		}
	}
	else
	{
		cv_img = img;
		//img.copyTo(cv_img);
	}


	// transform
	const int crop_size = param_.crop_size();
	const bool contrast_adjustment = param_.contrast_adjustment();
	const bool smooth_filtering = param_.smooth_filtering();
	const bool jpeg_compression = param_.jpeg_compression();
	const bool bMirror = param_.mirror();
	const bool bFlipHor = param_.flip_hor();


	const int img_channels = cv_img.channels();
	const int img_height = cv_img.rows;
	const int img_width = cv_img.cols;

	// Check dimensions.
	const int channels = transformed_blob->channels();
	const int height = transformed_blob->height();
	const int width = transformed_blob->width();
	const int num = transformed_blob->num();

	CHECK_EQ(channels, img_channels);
	CHECK_LE(height, img_height);
	CHECK_LE(width, img_width);
	CHECK_GE(num, 1);

	CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

	const Dtype scale = param_.scale();
	const bool has_mean_file = param_.has_mean_file();
	const bool has_mean_values = mean_values_.size() > 0;

	CHECK_GT(img_channels, 0);
	CHECK_GE(img_height, crop_size);
	CHECK_GE(img_width, crop_size);

	int rIndex = Rand(0X0FFFFFFF);
	cv::RNG rng(rIndex);
	
	// param for rotation
	const float rotation_angle_interval = param_.rotation_angle_interval();
	// param for scaling
	const float min_scaling_factor = param_.min_scaling_factor();
	const float max_scaling_factor = param_.max_scaling_factor();
	bool bScale = false;
	if (min_scaling_factor<1.0 && max_scaling_factor > 1.0)
		bScale = true;

	if (display)
		cv::imshow("Source", cv_img);

	// scaling factor for height and width respectively
	rIndex = Rand(0X0FFFFFFF);
	rng = cv::RNG(rIndex);
	float sf_w = rng.uniform(min_scaling_factor, max_scaling_factor);
	rIndex = Rand(0X0FFFFFFF);
	rng = cv::RNG(rIndex);
	float sf_h = rng.uniform(min_scaling_factor, max_scaling_factor);
	int widthNew = static_cast<int>(cv_img.cols * sf_w);
	int heightNew = static_cast<int>(cv_img.rows * sf_h);
	if (widthNew < crop_size)
		widthNew = crop_size;
	if (heightNew < crop_size)
		heightNew = crop_size;


	bool doScale = Rand(2) && bScale;
	if (doScale)
	{
		cv::resize(cv_img, cv_img, cv::Size(widthNew, heightNew));
	}
	if (display)
		cv::imshow("Scale Image", cv_img);

	int imgWidthIn = cv_img.cols;
	int imgHeightIn = cv_img.rows;

	// Flipping and Reflection -----------------------------------------------------------------
	bool doFlipHor = Rand(2) && bFlipHor;
	bool doMirror = Rand(2) && bFlipHor;

	if (doMirror&&doFlipHor)
	{
		cv::flip(cv_img, cv_img, -1);
	}
	else if (doMirror && !doFlipHor)
	{
		cv::flip(cv_img, cv_img, 1);
	}
	else if (!doMirror&&doFlipHor)
	{
		cv::flip(cv_img, cv_img, 0);
	}
	if (display)
		cv::imshow("Flipping and Reflection", cv_img);

	//cv::Mat cv_cropped_img = cv_img;
	// Cropping -------------------------------------------------------------
	bool bCropOffsetUse = param_.crop_offset_rand();
	bool bLeftUpCropUse = param_.left_up_crop_based();

	int h_off = 0;
	int w_off = 0;
	int mean_h_off = 0;
	int mean_w_off = 0;

	if (crop_size!= imgHeightIn || crop_size!= imgHeightIn)
	{
		CHECK_EQ(crop_size, height);
		CHECK_EQ(crop_size, width);

		if (bCropOffsetUse) 
		{
			h_off = Rand(imgHeightIn - crop_size + 1);
			w_off = Rand(imgWidthIn - crop_size + 1);
		}
		else 
		{
			if (bLeftUpCropUse)
			{
				h_off = 0;
				w_off = 0;
			}
			else
			{
				h_off = (imgHeightIn - crop_size) / 2;
				w_off = (imgWidthIn - crop_size) / 2;
			}
		}
		cv::Rect roi(w_off, h_off, crop_size, crop_size);
		cv_img = cv_img(roi);

		if (display)
			cv::imshow("Cropping", cv_img);

		//cv_cropped_img.copyTo(cv_img);
	}
	else 
	{
		CHECK_EQ(imgHeightIn, height);
		CHECK_EQ(imgWidthIn, width);
	}

	// perspective ----------------------------------------------------------
	rIndex = Rand(0X0FFFFFFF);
	rng = cv::RNG(rIndex);
	float max_perpective_ratio = param_.max_perspective_ratio();
	float perpective_ratio = rng.uniform(float(1.0), max_perpective_ratio);
	if (perpective_ratio != 1.0)
	{
   		int offsetPerspective = static_cast<int>(abs(perpective_ratio * crop_size - crop_size));
		cv::Point2f fPtOrg[4] = { cv::Point2f(0,0), cv::Point2f(crop_size,0)
			,cv::Point2f(crop_size,crop_size), cv::Point2f(0,crop_size) };


		vector<cv::Point2f*> vfPtPers;
		cv::Point2f fPtPers_a[4] = { cv::Point2f(offsetPerspective,0), cv::Point2f(crop_size- offsetPerspective,0)
			,cv::Point2f(crop_size,crop_size), cv::Point2f(0,crop_size) };
		vfPtPers.push_back(fPtPers_a);

		cv::Point2f fPtPers_b[4] = { cv::Point2f(0,0), cv::Point2f(crop_size ,offsetPerspective)
			,cv::Point2f(crop_size,crop_size- offsetPerspective), cv::Point2f(0,crop_size) };
		vfPtPers.push_back(fPtPers_b);

		cv::Point2f fPtPers_c[4] = { cv::Point2f(0,0), cv::Point2f(crop_size ,0)
			,cv::Point2f(crop_size-offsetPerspective,crop_size), cv::Point2f(offsetPerspective,crop_size) };
		vfPtPers.push_back(fPtPers_c);

		cv::Point2f fPtPers_d[4] = { cv::Point2f(0,offsetPerspective), cv::Point2f(crop_size ,0)
			,cv::Point2f(crop_size ,crop_size), cv::Point2f(0,crop_size-offsetPerspective) };
		vfPtPers.push_back(fPtPers_d);

		rIndex = Rand(4);
		cv::Mat perM = cv::getPerspectiveTransform(vfPtPers[rIndex], fPtOrg);
		cv::Mat PersImg;
		cv::warpPerspective(cv_img, cv_img, perM, cv_img.size());
		if (display)
			cv::imshow("Perspecive", cv_img);
	}


	// Rotation -------------------------------------------------------------
	double rotation_degree = 0;
	if (rotation_angle_interval != 0) {
		cv::Mat dst;
		int interval = 360 / rotation_angle_interval;
		int apply_rotation = Rand(interval);
		if (apply_rotation != 0)
		{
			cv::Point2f pt(cv_img.cols / 2., cv_img.rows / 2.);
			rotation_degree = apply_rotation*rotation_angle_interval;
			double rotatedScale = 1.0;
			double sin_a = abs(sin(degree2radian(rotation_degree)));
			double cos_a = abs(cos(degree2radian(rotation_degree)));
			double cos_2a = cos_a*cos_a - sin_a*sin_a;
			if (cos_2a == 0.0)
			{
				rotatedScale = sqrt(2);
			}
			else
			{
				double wr = (crop_size*cos_a - crop_size*sin_a) / cos_2a;
				rotatedScale = crop_size / wr;
			}

			cv::Mat r = getRotationMatrix2D(pt, rotation_degree, rotatedScale);
			warpAffine(cv_img, dst, r, cv::Size(cv_img.cols, cv_img.rows));

			if (display)
			{
				cv::imshow("Rotation", dst);
				cv::waitKey(0);
			}
			dst.copyTo(cv_img);
		}
	}


	// Smooth Filtering -------------------------------------------------------------
	int smooth_param1 = 3;
	int apply_smooth = Rand(2);
	if (smooth_filtering && apply_smooth) {
		int smooth_type = Rand(4); // see opencv_util.hpp
		smooth_param1 = 3 + 2 * (Rand(1));
		switch (smooth_type) {
		case 0:
			//cv::Smooth(cv_img, cv_img, smooth_type, smooth_param1);
			cv::GaussianBlur(cv_img, cv_img, cv::Size(smooth_param1, smooth_param1), 0);
			break;
		case 1:
			cv::blur(cv_img, cv_img, cv::Size(smooth_param1, smooth_param1));
			break;
		case 2:
			cv::medianBlur(cv_img, cv_img, smooth_param1);
			break;
		case 3:
			cv::boxFilter(cv_img, cv_img, -1, cv::Size(smooth_param1 * 2, smooth_param1 * 2));
			break;
		}
		if (display)
			cv::imshow("Smooth Filtering", cv_img);
	}

	// Contrast and Brightness Adjuestment ----------------------------------------
	float alpha = 1, beta = 0;
	int apply_contrast = Rand(2);
	if (contrast_adjustment && apply_contrast) {
		float min_alpha = 0.8, max_alpha = 1.2;
		rIndex = Rand(0X0FFFFFFF);
		rng = cv::RNG(rIndex);
		alpha = rng.uniform(min_alpha, max_alpha);
		beta = (float)(Rand(6));
		// flip sign
		if (Rand(2)) beta = -beta;
		cv_img.convertTo(cv_img, -1, alpha, beta);
		if (display)
			cv::imshow("Contrast Adjustment", cv_img);
	}

	// JPEG Compression -------------------------------------------------------------
	// DO NOT use the following code as there is some memory leak which I cann't figure out
	int QF = 100;
	int apply_JPEG = Rand(2);
	if (jpeg_compression && apply_JPEG) {
		// JPEG quality factor
		QF = 95 + 1 * (Rand(6));
		int cp[] = { 1, QF };
		vector<int> compression_params(cp, cp + 2);
		vector<unsigned char> img_jpeg;
		//cv::imencode(".jpg", cv_img, img_jpeg);
		cv::imencode(".jpg", cv_img, img_jpeg, compression_params);
		cv::Mat temp = cv::imdecode(img_jpeg, 1);
		temp.copyTo(cv_img);
		if (display)
			cv::imshow("JPEG Compression", cv_img);
	}

	if (display)
		cv::imshow("Final", cv_img);




	//--------------------!! for debug only !!-------------------
	if (display) {
		LOG(INFO) << "----------------------------------------";
		LOG(INFO) << "src width: " << width << ", src height: " << height;
		LOG(INFO) << "dest width: " << crop_size << ", dest height: " << crop_size;
		if (doMirror) {
			LOG(INFO) << "Mirroring";
		}
		if (doFlipHor) {
			LOG(INFO) << "Horizontal Flipping";
		}
		if (smooth_filtering && apply_smooth) {
			LOG(INFO) << "* parameter for smooth filtering: ";
			//LOG(INFO) << "  smooth type: " << smooth_type << ", smooth param1: " << smooth_param1;
		}
		if (contrast_adjustment && apply_contrast) {
			LOG(INFO) << "* parameter for contrast adjustment: ";
			LOG(INFO) << "  alpha: " << alpha << ", beta: " << beta;
		}
		if (jpeg_compression && apply_JPEG) {
			LOG(INFO) << "* parameter for JPEG compression: ";
			LOG(INFO) << "  QF: " << QF;
		}
		LOG(INFO) << "* parameter for cropping: ";
		LOG(INFO) << "  w: " << w_off << ", h: " << h_off;
		LOG(INFO) << "  roi_width: " << crop_size << ", roi_height: " << crop_size;
		LOG(INFO) << "* parameter for rotation: ";
		LOG(INFO) << "  angle_interval: " << rotation_angle_interval;
		LOG(INFO) << "  angle: " << rotation_degree;
		cvWaitKey(0);
	}

	Dtype* mean = NULL;
	if (has_mean_file) {
		CHECK_EQ(img_channels, data_mean_.channels());
		CHECK_EQ(img_height, data_mean_.height());
		CHECK_EQ(img_width, data_mean_.width());
		mean = data_mean_.mutable_cpu_data();
	}

	if (has_mean_values) {
		CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
			"Specify either 1 mean_value or as many as channels: " << img_channels;
		if (img_channels > 1 && mean_values_.size() == 1) {
			// Replicate the mean_value for simplicity
			for (int c = 1; c < img_channels; ++c) {
				mean_values_.push_back(mean_values_[0]);
			}
		}
	}

	Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	int top_index;
	for (int h = 0; h < height; ++h) {
		const uchar* ptr = cv_img.ptr<uchar>(h); // here!!
		int img_index = 0;
		for (int w = 0; w < width; ++w) {
			for (int c = 0; c < img_channels; ++c) {
				//if (do_mirror) {
				//  top_index = (c * height + h) * width + (width - 1 - w);
				//} else {
				top_index = (c * height + h) * width + w;
				//}
				// int top_index = (c * height + h) * width + w;
				Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
				if (has_mean_file) {
					int mean_index = (c * img_height + h) * img_width + w;
					transformed_data[top_index] =
						(pixel - mean[mean_index]) * scale;
				}
				else {
					if (has_mean_values) {
						transformed_data[top_index] =
							(pixel - mean_values_[c]) * scale;
					}
					else {
						transformed_data[top_index] = pixel * scale;
					}
				}
			}
		}
	}

}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::CopyBlob(Blob<Dtype>* source_blob,
	Blob<Dtype>* target_blob) {
	const int input_num = source_blob->num();
	const int input_height = source_blob->height();
	const int input_width = source_blob->width();

	const int channels = target_blob->channels();
	const int height = target_blob->height();
	const int width = target_blob->width();
	Dtype* input_data = source_blob->mutable_cpu_data();
	Dtype* target_data = target_blob->mutable_cpu_data();
	for (int n = 0; n < input_num; ++n) {
		int top_index_n = n * channels;
		int data_index_n = n * channels;
		for (int c = 0; c < channels; ++c) {
			int top_index_c = (top_index_n + c) * height;
			int data_index_c = (data_index_n + c) * input_height;
			for (int h = 0; h < height; ++h) {
				int top_index_h = (top_index_c + h) * width;
				int data_index_h = (data_index_c + h) * input_width;
				for (int w = 0; w < width; ++w) {
					target_data[top_index_h + w] = input_data[data_index_h + w];
				}
			}
		}
	}
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
	//const bool needs_rand = param_.mirror() ||
	//	(phase_ == TRAIN && param_.crop_size());
	//if (needs_rand) {
	//	const unsigned int rng_seed = caffe_rng_rand();
	//	rng_.reset(new Caffe::RNG(rng_seed));
	//}
	//else {
	//	rng_.reset();
	//}

	const unsigned int rng_seed = caffe_rng_rand();
	rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
