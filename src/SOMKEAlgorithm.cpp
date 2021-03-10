//
// Created by tomev on 07/03/2021.
//

#include <iostream>

#include "../include/SOMKE/SOMKEAlgorithm.h"

SOMKEAlgorithm::SOMKEAlgorithm(vector<Point> *divergence_domain, KernelPtr kernel, unsigned int neurons_number, unsigned int epochs_number, unsigned int data_window_size)
  : neurons_number_(neurons_number), epochs_number_(epochs_number), kernel_ptr_(kernel),
  divergence_domain_(divergence_domain), data_window_size_(data_window_size)
{ }

void SOMKEAlgorithm::PerformStep(Point data_point) {
  data_window_.push_back(data_point);

  if(data_window_.size() == data_window_size_){
    AddNewSOMSequence(data_window_);
    data_window_.clear();
  }

  // TODO Merge
}

void SOMKEAlgorithm::AddNewSOMSequence(vector<Point> data_window) {

  ++data_window_iterator_;

  som_sequences_.push_back(SOMSequence());

  som_sequences_.back().net = GenerateNetwork(data_window);
  TrainNetwork(&(som_sequences_.back().net), data_window);

  som_sequences_.back().voronoi_regions = ComputeVoronoiRegionsForNeurons(som_sequences_.back().net, data_window);
  som_sequences_.back().range = std::pair<int, int>(data_window_iterator_, data_window_iterator_);

  som_sequences_.back().kl_divergence = 0;

  if(som_sequences_.size() > 1){
    int i = som_sequences_.size() - 1;
    som_sequences_.back().kl_divergence = ComputeDivergenceBetweenSOMSequences(som_sequences_[i], som_sequences_[i - 1]);
  }
}

KohonenNetwork SOMKEAlgorithm::GenerateNetwork(vector<Point> data_window) {

  KohonenNetwork net;
  DistanceFunction distance_function;
  ActivationFunction activation_function(2.0, 1);

  neural_net::generate_kohonen_network(1, neurons_number_, activation_function, distance_function, data_window,
                                       net, randomizer_);

  return net;
}

void SOMKEAlgorithm::TrainNetwork(KohonenNetwork *net,
                                  vector<Point> data_window) {
  Topology topology;
  SpaceFunctor space_functor(100, 1);
  NetFunctor net_functor(10, 1);
  DistanceFunction distance_function;
  TrainingWeight weight(net_functor, space_functor, topology, distance_function);
  TrainingFunctional training_functional(weight, 0.3);
  TrainingAlgorithm training_algorithm(training_functional);

  for(int epoch = 0; epoch < epochs_number_; ++epoch){

    //std::cout << "Epoch " << epoch << std::endl;

    training_algorithm(data_window.begin(), data_window.end(), net);
    training_algorithm.training_functional.generalized_training_weight.network_function.sigma *= 2.0/3.0;
    std::random_shuffle(data_window.begin(), data_window.end());
  }

  //I can modify weights of each neuron via: (*net).objects[0][0].weights = {2.22};
}

vector<int> SOMKEAlgorithm::ComputeVoronoiRegionsForNeurons(const KohonenNetwork &net, const vector<Point> &data_window) {
  vector<int> voronoi_regions = {};

  while(voronoi_regions.size() < neurons_number_){
    voronoi_regions.push_back(0);
  }

  DistanceFunction distance;

  for(auto data : data_window){

    // Initialize the distance for current data point. Remember we're in 1D case.
    int smallest_distance_index = 0;
    double smallest_distance = distance(data, net.objects[0][0].weights);

    // Find index neuron with smallest weight distance to given data point.
    for(size_t i = 1; i < net.objects[0].size(); ++i){

      double current_distance = distance(data, net.objects[0][i].weights);

      if(current_distance < smallest_distance){
        smallest_distance_index = i;
        smallest_distance = current_distance;
      }
    }

    // Update Voronoi regions
    voronoi_regions[smallest_distance_index] = voronoi_regions[smallest_distance_index] + 1;
  }

  return voronoi_regions;
}

double SOMKEAlgorithm::ComputeDivergenceBetweenSOMSequences(const SOMSequence &som_sequence1,
                                                            const SOMSequence &som_sequence2) {
  // Compute PDFs on divergence domain
  auto pdf1 = ComputePDFValuesForDivergenceComputation(som_sequence1, divergence_domain_);
  auto pdf2 = ComputePDFValuesForDivergenceComputation(som_sequence2, divergence_domain_);

  return ComputeKLDivergenceEstimatorBetweenPDFs(pdf1, pdf2);
}

vector<double> SOMKEAlgorithm::ComputePDFValuesForDivergenceComputation(const SOMSequence &som_sequence, vector<Point> *domain) {
  vector<double> pdf = {};

  // Do note we start from i = 1. This is related to formula (20) from SOMKE work.
  for(int i = 1; i < domain->size(); ++i){
    Point pt = {((*domain)[i][0] + (*domain)[i-1][0]) / 2.0}; // 1D assumption!
    pdf.push_back(ComputePDFValueAtPointFromSOMSequence(som_sequence, pt));
  }

  return pdf;
}

double SOMKEAlgorithm::ComputePDFValueAtPointFromSOMSequence(const SOMSequence &som_sequence, const Point &pt) {

  double pdf_value = 0;
  int cs_sum = std::accumulate(som_sequence.voronoi_regions.begin(), som_sequence.voronoi_regions.end(), 0);

  for(size_t i = 0; i < som_sequence.voronoi_regions.size(); ++i){
    Point point = {pt[0] - som_sequence.net.objects[0][i].weights[0]}; // Assuming 1D
    pdf_value += som_sequence.voronoi_regions[i] * kernel_ptr_->GetValue(point) / cs_sum;
  }

  return pdf_value;
}

double SOMKEAlgorithm::ComputeKLDivergenceEstimatorBetweenPDFs(const vector<double> &pdf1,
                                                               const vector<double> &pdf2) {
  // We will be using Riemann sum approximation, as in the paper.
  double divergence_estimator = 0;
  // We're assuming that the domain has equal ranges and at least 2 points (and 1D).
  double range_width = (*divergence_domain_)[1][0] - (*divergence_domain_)[0][0];

  auto f = [](double p, double q){
    return p * log(p / q);
  };

  for(int i = 0; i < pdf1.size(); ++i){
    divergence_estimator += range_width * f(pdf1[i], pdf2[i]);
  }

  return divergence_estimator;
}






