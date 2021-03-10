//
// Created by tomev on 07/03/2021.
//

#ifndef SOMKE_SOMKEALGORITHM_H
#define SOMKE_SOMKEALGORITHM_H


#include "SOMSequenceEntry.h"
#include "Kernel.h"

class SOMKEAlgorithm {
  /**
   * @brief SOMKE algorithm implementation for 1D real input case.
   */
  public:

    SOMKEAlgorithm(vector<Point> *divergence_domain, KernelPtr kernel, unsigned int neurons_number = 100,
                   unsigned int epochs_number = 3000, unsigned int data_window_size = 500, unsigned int max_number_of_som_sequences = 1);
    void PerformStep(Point data_point);
    double GetValue(Point data_point);

  private:

    std::vector<SOMSequenceEntry> som_sequence_;
    int data_window_iterator_ = 0;
    vector<Point> data_window_ = {};
    int data_window_size_ = 0;

    neural_net::External_randomize randomizer_;
    unsigned int neurons_number_ = 0;
    unsigned int epochs_number_ = 0;
    unsigned int max_number_of_som_sequences_ = 0;

    KernelPtr kernel_ptr_;
    vector<Point> *divergence_domain_;

    double beta_ = 0; // Rate of increased probability of older entries merge

    void AddNewSOMSequenceEntry(vector<Point> data_window);
    KohonenNetwork GenerateNetwork(vector<Point> data_window);
    void TrainNetwork(KohonenNetwork *net, vector<Point> data_window);
    vector<int> ComputeVoronoiRegionWeightsForNeurons(const KohonenNetwork &net, const vector<Point> &data_window);
    double ComputeDivergenceBetweenSOMSequences(const SOMSequenceEntry &som_sequence1, const SOMSequenceEntry &som_sequence2);
    vector<double> ComputePDFValuesForDivergenceComputation(const SOMSequenceEntry &som_sequence, vector<Point> *domain);
    double ComputePDFValueAtPointFromSOMSequence(const SOMSequenceEntry &som_sequence, const Point &pt);
    double ComputeKLDivergenceEstimatorBetweenPDFs(const vector<double> &pdf1, const vector<double> &pdf2);

    void MergeMostSimilarSOMSequenceEntries();
    unsigned int FindIndexOfSOMSequenceEntryWithLowestModifiedDivergence();
    double ComputeModifiedDivergenceOfSOMSequenceEntry(const SOMSequenceEntry &entry);
    vector<Point> GetNeuronWeightsFromEntries(const vector<SOMSequenceEntry> &entries);
    vector<int> GetVoronoiRegionWeightsFromEntries(const vector<SOMSequenceEntry> &entries);
    vector<Point> DrawTrainingDataFromWeightsAndVoronoiRegions(const vector<Point> &neuron_weights,
                                                               const vector<int> &voronoi_regions);


};

#endif //SOMKE_SOMKEALGORITHM_H
