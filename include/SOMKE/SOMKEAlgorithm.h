//
// Created by tomev on 07/03/2021.
//

#ifndef SOMKE_SOMKEALGORITHM_H
#define SOMKE_SOMKEALGORITHM_H

#include "SOMSequenceEntry.h"
#include "SOMKEMergingStrategy.h"
#include "Kernel.h"

class SOMKEAlgorithm {
    /**
     * @brief SOMKE algorithm implementation for 1D real input case.
     */
  public:

    SOMKEAlgorithm(KernelPtr kernel, MergingStrategyPtr merging_strategy, unsigned int neurons_number = 100,
                   unsigned int epochs_number = 100, unsigned int data_window_size = 500, const double &sigma0 = 25,
                   const double &eta0 = 3, const double &tau2 = 1000);
    void PerformStep(Point data_point);
    double GetValue(Point data_point, const vector<int> &ranges = {});
    vector<Point> divergence_domain_;

  protected:

    std::vector<SOMSequenceEntry> som_sequence_;
    int data_window_iterator_ = 0;
    vector<Point> data_window_ = {};
    int data_window_size_ = 0;
    neural_net::External_randomize randomizer_;
    unsigned int neurons_number_ = 0;
    unsigned int epochs_number_ = 0;
    KernelPtr kernel_ptr_;
    MergingStrategyPtr merging_strategy_ptr_;
    size_t divergence_domain_points_number = 1001;
    double sigma0_;
    double tau1_;
    double eta0_;
    double tau2_;
    double m_ = 0.5; // For 1D data

    void AddNewSOMSequenceEntry(vector<Point> data_window);
    KohonenNetwork GenerateNetwork(vector<Point> data_window);
    void TrainNetwork(KohonenNetwork *net, vector<Point> data_window);
    vector<int> ComputeVoronoiRegionWeightsForNeurons(const KohonenNetwork &net, const vector<Point> &data_window);
    double ComputeDivergenceBetweenSOMSequences(const SOMSequenceEntry &som_sequence1,
                                                const SOMSequenceEntry &som_sequence2);
    vector<double> ComputePDFValuesForDivergenceComputation(const SOMSequenceEntry &som_sequence,
                                                            const vector<Point> &domain);
    double ComputePDFValueAtPointFromSOMSequenceEntry(const SOMSequenceEntry &entry, const Point &pt);
    double ComputeKLDivergenceEstimatorBetweenPDFs(const vector<double> &pdf1, const vector<double> &pdf2);
    void UpdateDivergenceDomain();
    double ComputeBandwidth(const vector<Point> &data_window);
    void MergeAdequateSOMSequenceEntries();
    vector<Point> GetNeuronWeightsFromEntries(const vector<SOMSequenceEntry> &entries);
    vector<int> GetVoronoiRegionWeightsFromEntries(const vector<SOMSequenceEntry> &entries);
    vector<double> GetBandwidthsFromEntries(const vector<SOMSequenceEntry> &entries);
    vector<Point> DrawTrainingDataFromWeightsAndVoronoiRegions(const vector<Point> &neuron_weights,
                                                               const vector<int> &voronoi_regions);

    vector<SOMSequenceEntry> GetEntriesOfInterestFromRanges(const vector<int> &ranges) const;
};

#endif //SOMKE_SOMKEALGORITHM_H
