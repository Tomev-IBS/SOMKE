//
// Created by tomev on 07/03/2021.
//

#include <iostream>
#include <random>
#include <set>

#include "../include/SOMKE/SOMKEAlgorithm.h"

vector<double> StripPoints(const vector<Point> &points) {
  vector<double> stripped_points = {};

  for(auto point : points) {
    stripped_points.push_back(point[0]); // 1D assumption
  }

  return stripped_points;
}

double StandardDeviation(const vector<double> &values) {
  if(values.size() < 2) {
    return 0;
  }

  auto mean = accumulate(values.begin(), values.end(), 0.0) / values.size();
  auto squares_sum = inner_product(values.begin(), values.end(), values.begin(), 0.0);

  return sqrt(squares_sum / values.size() - pow(mean, 2));
}

SOMKEAlgorithm::SOMKEAlgorithm(KernelPtr kernel, MergingStrategyPtr merging_strategy, unsigned int neurons_number,
                               unsigned int epochs_number, unsigned int data_window_size, const double &sigma0,
                               const double &eta0, const double &tau2)
    : neurons_number_(neurons_number), epochs_number_(epochs_number), kernel_ptr_(kernel), eta0_(eta0), tau2_(tau2),
      data_window_size_(data_window_size), sigma0_(sigma0) {
  merging_strategy_ptr_ = merging_strategy;
  merging_strategy_ptr_->SetDataWindowIterator(&data_window_iterator_);
  tau1_ = 1000 / log(sigma0);
  }

  void SOMKEAlgorithm::PerformStep(Point data_point) {
    data_window_.push_back(data_point);

    if(data_window_.size() == data_window_size_) {
      AddNewSOMSequenceEntry(data_window_);
      UpdateDivergenceDomain();
      data_window_.clear();
    }

    // We're implementing only fixed memory strategy for now.
    while(merging_strategy_ptr_->ShouldMergeBePerformed(som_sequence_)) {
      MergeAdequateSOMSequenceEntries();
    }

  }

  void SOMKEAlgorithm::AddNewSOMSequenceEntry(vector<Point> data_window) {

    ++data_window_iterator_;

    som_sequence_.push_back(SOMSequenceEntry());

    som_sequence_.back().net = GenerateNetwork(data_window);
    TrainNetwork(&(som_sequence_.back().net), data_window);

    som_sequence_.back().voronoi_regions = ComputeVoronoiRegionWeightsForNeurons(som_sequence_.back().net, data_window);
    som_sequence_.back().range = std::pair<int, int>(data_window_iterator_, data_window_iterator_);

    som_sequence_.back().bandwidth = ComputeBandwidth(data_window);

    som_sequence_.back().kl_divergence = 0;

    if(som_sequence_.size() > 1) {
      int i = som_sequence_.size() - 1;
      som_sequence_.back().kl_divergence = ComputeDivergenceBetweenSOMSequences(som_sequence_[i],
                                                                                som_sequence_[i - 1]);
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

    for(int epoch = 0; epoch < epochs_number_; ++epoch) {
      training_algorithm(data_window.begin(), data_window.end(), net);
      training_algorithm.training_functional.generalized_training_weight.network_function.sigma =
          sigma0_ * exp(- epoch / tau1_);
      //double eta = eta0_ * exp(- epoch / tau2_);
      //training_algorithm.training_functional.parameter = eta;
    std::random_shuffle(data_window.begin(), data_window.end());
  }
}

vector<int> SOMKEAlgorithm::ComputeVoronoiRegionWeightsForNeurons(const KohonenNetwork &net,
                                                                  const vector<Point> &data_window) {
  vector<int> voronoi_regions = {};

  while(voronoi_regions.size() < neurons_number_) {
    voronoi_regions.push_back(0);
  }

  DistanceFunction distance;

  for(auto data : data_window) {

    // Initialize the distance for current data point. Remember we're in 1D case.
    int smallest_distance_index = 0;
    double smallest_distance = distance(data, net.objects[0][0].weights);

    // Find index neuron with smallest weight distance to given data point.
    for(size_t i = 1; i < net.objects[0].size(); ++i) {

      double current_distance = distance(data, net.objects[0][i].weights);

      if(current_distance < smallest_distance) {
        smallest_distance_index = i;
        smallest_distance = current_distance;
      }
    }

    // Update Voronoi regions
    voronoi_regions[smallest_distance_index] = voronoi_regions[smallest_distance_index] + 1;
  }

  return voronoi_regions;
}

double SOMKEAlgorithm::ComputeDivergenceBetweenSOMSequences(const SOMSequenceEntry &som_sequence1,
                                                            const SOMSequenceEntry &som_sequence2) {
  // Compute PDFs on divergence domain
  auto pdf1 = ComputePDFValuesForDivergenceComputation(som_sequence1, divergence_domain_);
  auto pdf2 = ComputePDFValuesForDivergenceComputation(som_sequence2, divergence_domain_);

  return ComputeKLDivergenceEstimatorBetweenPDFs(pdf1, pdf2);
}

vector<double> SOMKEAlgorithm::ComputePDFValuesForDivergenceComputation(const SOMSequenceEntry &som_sequence,
                                                                        const vector<Point> &domain) {
  vector<double> pdf = {};

  // Do note we start from i = 1. This is related to formula (20) from SOMKE work.
  for(int i = 1; i < domain.size(); ++i) {
    Point pt = {(domain[i][0] + domain[i - 1][0]) / 2.0}; // 1D assumption!
    pdf.push_back(ComputePDFValueAtPointFromSOMSequenceEntry(som_sequence, pt));
  }

  return pdf;
}

double SOMKEAlgorithm::ComputePDFValueAtPointFromSOMSequenceEntry(const SOMSequenceEntry &entry, const Point &pt) {

  double pdf_value = 0;
  int cs_sum = std::accumulate(entry.voronoi_regions.begin(), entry.voronoi_regions.end(), 0);

  for(size_t i = 0; i < entry.voronoi_regions.size(); ++i) {
    Point point = {(pt[0] - entry.net.objects[0][i].weights[0]) / entry.bandwidth}; // Assuming 1D
    pdf_value += entry.voronoi_regions[i] / entry.bandwidth * kernel_ptr_->GetValue(point) / cs_sum;
  }

  return pdf_value;
}

double SOMKEAlgorithm::ComputeKLDivergenceEstimatorBetweenPDFs(const vector<double> &pdf1,
                                                               const vector<double> &pdf2) {
  // We will be using Riemann sum approximation, as in the paper.
  double divergence_estimator = 0;
  // We're assuming that the domain has equal ranges and at least 2 points (and 1D).
  double range_width = divergence_domain_[1][0] - divergence_domain_[0][0];
  auto f = [](double p, double q) {
    return p * log(p / q);
  };

  for(int i = 0; i < pdf1.size(); ++i) {
    divergence_estimator += range_width * f(pdf1[i], pdf2[i]);
  }

  return divergence_estimator;
}

void SOMKEAlgorithm::MergeAdequateSOMSequenceEntries() {

  auto i = merging_strategy_ptr_->FindEntryToMerge(som_sequence_);
  SOMSequenceEntry merged_entry;
  vector<SOMSequenceEntry> entries_to_merge = {som_sequence_[i], som_sequence_[i - 1]};
  auto neuron_weights = GetNeuronWeightsFromEntries(entries_to_merge);
  auto voronoi_region_weights = GetVoronoiRegionWeightsFromEntries(entries_to_merge);
  auto data_window = DrawTrainingDataFromWeightsAndVoronoiRegions(neuron_weights, voronoi_region_weights);

  merged_entry.net = GenerateNetwork(data_window);
  TrainNetwork(&(merged_entry.net), data_window);

  merged_entry.range = std::pair<int, int>(std::min(entries_to_merge[0].range.first, entries_to_merge[1].range.first),
                                           std::max(entries_to_merge[0].range.second,
                                                    entries_to_merge[1].range.second));

  // Computing merged voronoi region weights
  merged_entry.voronoi_regions = {};
  while(merged_entry.voronoi_regions.size() < neurons_number_) {
    merged_entry.voronoi_regions.push_back(0);
  }

  DistanceFunction distance_function;

  for(size_t j = 0; j < neuron_weights.size(); ++j) {
    auto neuron_weight = neuron_weights[j];
    int closest_neuron_index = 0;
    double distance_to_closest_neuron = distance_function(neuron_weight, merged_entry.net.objects[0][0].weights);

    for(size_t i = 1; i < merged_entry.net.objects[0].size(); ++i) {
      double current_distance = distance_function(neuron_weight, merged_entry.net.objects[0][i].weights);
      if(current_distance < distance_to_closest_neuron) {
        distance_to_closest_neuron = current_distance;
        closest_neuron_index = i;
      }
    }

    merged_entry.voronoi_regions[closest_neuron_index] += voronoi_region_weights[j];
  }

  merged_entry.bandwidth = ComputeBandwidth(data_window);

  merged_entry.kl_divergence = 0;

  // Removing merged entries.
  som_sequence_.erase(som_sequence_.begin() + i - 1);
  som_sequence_.erase(som_sequence_.begin() + i - 1);

  // Adding merged one to the sequence at a proper place.
  SOMSequenceEntry *merged_entry_ptr = &merged_entry; // For convenience
  som_sequence_.insert(som_sequence_.begin() + i - 1, merged_entry);

  // Updating divergences.
  if(i > 1) {
    merged_entry_ptr->kl_divergence = ComputeDivergenceBetweenSOMSequences(merged_entry, som_sequence_[i - 2]);
  }

  if(i < som_sequence_.size()) {
    merged_entry_ptr->kl_divergence = ComputeDivergenceBetweenSOMSequences(merged_entry, som_sequence_[i]);
  }

}

vector<Point> SOMKEAlgorithm::DrawTrainingDataFromWeightsAndVoronoiRegions(const vector<Point> &neuron_weights,
                                                                           const vector<int> &voronoi_regions) {

  std::default_random_engine generator;
  std::discrete_distribution<int> distribution(voronoi_regions.begin(), voronoi_regions.end());
  vector<Point> training_data = {};

  while(training_data.size() < data_window_size_) {
    training_data.push_back(neuron_weights[distribution(generator)]);
  }

  return training_data;
}

vector<Point> SOMKEAlgorithm::GetNeuronWeightsFromEntries(const vector<SOMSequenceEntry> &entries) {
  vector<Point> neurons_weights = {};

  for(auto entry : entries) {
    for(size_t i = 0; i < entry.net.objects[0].size(); ++i) {
      neurons_weights.push_back(entry.net.objects[0][i].weights);
    }
  }

  return neurons_weights;
}

vector<int> SOMKEAlgorithm::GetVoronoiRegionWeightsFromEntries(const vector<SOMSequenceEntry> &entries) {
  vector<int> voronoi_regions = {};

  for(auto entry : entries) {
    for(size_t i = 0; i < entry.voronoi_regions.size(); ++i) {
      voronoi_regions.push_back(entry.voronoi_regions[i]);
    }
  }

  return voronoi_regions;
}

double SOMKEAlgorithm::GetValue(Point data_point, const vector<int> &ranges) {

  // We want KDE from the latest sequence.
  if(som_sequence_.size() < 1) {
    return 0;
  }

  vector<SOMSequenceEntry> sequences_of_interest = GetEntriesOfInterestFromRanges(ranges);

  auto neuron_weights = GetNeuronWeightsFromEntries(sequences_of_interest);
  auto voronoi_region_weights = GetVoronoiRegionWeightsFromEntries(sequences_of_interest);
  auto bandwidths = GetBandwidthsFromEntries(sequences_of_interest);
  auto voronoi_region_weights_sum = std::accumulate(voronoi_region_weights.begin(), voronoi_region_weights.end(), 0);
  double value = 0;
  int bandwidth_index = -1;
  double current_bandwidth = 1;

  for(size_t i; i < neuron_weights.size(); ++i) {

    if(i % neurons_number_ == 0) {
      ++bandwidth_index;
      current_bandwidth = bandwidths[bandwidth_index];
    }

    auto pt = {(data_point[0] - neuron_weights[i][0]) / current_bandwidth}; // 1D assumption
    value += voronoi_region_weights[i] / current_bandwidth * kernel_ptr_->GetValue(pt) / voronoi_region_weights_sum;
  }

  return value;
}

void SOMKEAlgorithm::UpdateDivergenceDomain() {

  divergence_domain_ = {};

  if(som_sequence_.empty()) {
    return;
  }

  auto neuron_weights = GetNeuronWeightsFromEntries(som_sequence_);
  auto stripped_neuron_weights = StripPoints(neuron_weights);
  auto st_dev = StandardDeviation(stripped_neuron_weights);
  auto min_weight = *(std::min_element(stripped_neuron_weights.begin(), stripped_neuron_weights.end()));
  auto max_weight = *(std::max_element(stripped_neuron_weights.begin(), stripped_neuron_weights.end()));
  auto domain_min = min_weight - 5 * st_dev;
  auto domain_max = max_weight + 5 * st_dev;
  auto step_size = (domain_max - domain_min) / divergence_domain_points_number;

  divergence_domain_.push_back({domain_min});

  while(divergence_domain_.size() < divergence_domain_points_number) {
    divergence_domain_.push_back({divergence_domain_.back()[0] + step_size}); // 1D assumption
  }

}

double SOMKEAlgorithm::ComputeBandwidth(const vector<Point> &data_window) {
  // Assuming normal kernel!
  double bandwidth = 1.06;
  bandwidth *= StandardDeviation(StripPoints(data_window));
  bandwidth *= pow(data_window.size(), -0.2);
  return bandwidth;
}

vector<double> SOMKEAlgorithm::GetBandwidthsFromEntries(const vector<SOMSequenceEntry> &entries) {
  vector<double> bandwidths = {};

  for(auto entry : entries) {
    bandwidths.push_back(entry.bandwidth);
  }

  return bandwidths;
}

vector<SOMSequenceEntry> SOMKEAlgorithm::GetEntriesOfInterestFromRanges(const vector<int> &ranges) const {
  vector<SOMSequenceEntry> sequences_of_interest;
  std::set<size_t> unique_indices_of_entries_of_interest = {};

  // Finding unique indices
  for(auto val : ranges){
    for(size_t i = 0; i < som_sequence_.size(); ++i){
      auto entry = som_sequence_[i];
      if(val >= entry.range.first  && val <= entry.range.second){
        unique_indices_of_entries_of_interest.insert(i);
        break;
      }
    }
  }

  // Get sequences of interest
  for(auto it = unique_indices_of_entries_of_interest.begin(); it != unique_indices_of_entries_of_interest.end(); ++it){
    sequences_of_interest.push_back(som_sequence_[*it]);
  }

  // If there are no sequences of interest, return the most actual one.
  if(sequences_of_interest.empty()){
    sequences_of_interest.push_back(som_sequence_.back());
  }

  return sequences_of_interest;
}








