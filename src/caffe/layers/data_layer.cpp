#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
}

template <typename Dtype>
bool DataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void DataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

//
//  /////////////////dw edit panding  03.31/////////////////////////
//  Datum datum;
//  datum.ParseFromString(cursor_->value());
//  vector<int> one_batch_top_shape
//	  = this->data_transformer_->InferBlobShape(datum);
//  vector<int> top_shape = one_batch_top_shape;
//  this->transformed_data_.Reshape(one_batch_top_shape);
//  // Reshape batch according to the batch_size.
//  top_shape[0] = batch_size;
//  batch->data_.Reshape(top_shape);
//
//  Dtype* top_data = batch->data_.mutable_cpu_data();
//  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
//
//  if (this->output_labels_) {
//	  top_label = batch->label_.mutable_cpu_data();
//  }
//
//  vector<shared_ptr<Blob<Dtype> > > vTransformed_data;
//  for (int item_id = 0; item_id < batch_size; ++item_id) {
//	  vTransformed_data.push_back(shared_ptr<Blob<Dtype> >(
//		  new Blob<Dtype>(one_batch_top_shape)));
//  }
//
//  vector<Datum> vDatum;
//  for (int item_id = 0; item_id < batch_size; ++item_id) {
//	  while (Skip()) {
//		  Next();
//	  }
//	  CPUTimer timer;
//	  timer.Start();
//
//	  Datum datum;
//	  datum.ParseFromString(cursor_->value());
//	  vDatum.push_back(datum);
//
//	  if (this->output_labels_) {
//		  top_label[item_id] = datum.label();
//	  }
//
//	  read_time += timer.MicroSeconds();
//	  timer.Stop();
//	  Next();
//  }
//
////#if defined(_OPENMP)
////#pragma omp parallel for reduction(+:read_time, trans_time)
////#endif  // use_openmp
//  for (int item_id = 0; item_id < batch_size; ++item_id) {
//	  CPUTimer timer;
//	  timer.Start();
//
//	  this->data_transformer_->Transform(vDatum[item_id],
//		  vTransformed_data[item_id].get());
//
//	  trans_time += timer.MicroSeconds();
//	  timer.Stop();
//	  //Next();
//  }
//
//  for (int item_id = 0; item_id < batch_size; ++item_id) {
//	  CPUTimer cpytimer;
//	  int offset = batch->data_.offset(item_id);
//	  cpytimer.Start();
//	  this->transformed_data_.set_cpu_data(top_data + offset);
//	  this->data_transformer_->CopyBlob(vTransformed_data[item_id].get(),
//		  &(this->transformed_data_));
//	  trans_time += cpytimer.MicroSeconds();
//	  cpytimer.Stop();
//  }
//
//  batch_timer.Stop();
//  double prefetchBatchTime = batch_timer.MilliSeconds();
//  double readTransSum = read_time + trans_time;
//  read_time = (read_time / readTransSum)*prefetchBatchTime;
//  trans_time = (trans_time / readTransSum)*prefetchBatchTime;
//  DLOG(INFO) << "Prefetch batch: " << prefetchBatchTime << " ms.";
//  DLOG(INFO) << "     Read time: " << read_time << " ms.";
//  DLOG(INFO) << "Transform time: " << trans_time << " ms.";
//	  ////////////////////////////////////////////////
// 

  Datum datum;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      Dtype* top_label = batch->label_.mutable_cpu_data();
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();
    Next();
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
