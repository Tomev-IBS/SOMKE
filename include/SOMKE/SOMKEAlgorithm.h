//
// Created by tomev on 07/03/2021.
//

#ifndef SOMKE_SOMKEALGORITHM_H
#define SOMKE_SOMKEALGORITHM_H


#include "SOMSequence.h"
#include "Kernel.h"

class SOMKEAlgorithm {
  /**
   * @brief SOMKE algorithm implementation for 1D real input case.
   */
  public:

    SOMKEAlgorithm(vector<Point> *divergence_domain, KernelPtr kernel, unsigned int neurons_number = 100,
                   unsigned int epochs_number = 3000, unsigned int data_window_size = 500);
    void PerformStep(Point data_point);


  private:

    std::vector<SOMSequence> som_sequences_;
    int data_window_iterator_ = 0;
    vector<Point> data_window_ = {};
    int data_window_size_ = 0;

    neural_net::External_randomize randomizer_;
    unsigned int neurons_number_ = 0;
    unsigned int epochs_number_ = 0;

    KernelPtr kernel_ptr_;
    vector<Point> *divergence_domain_;

    void AddNewSOMSequence(vector<Point> data_window);
    KohonenNetwork GenerateNetwork(vector<Point> data_window);
    void TrainNetwork(KohonenNetwork *net, vector<Point> data_window);
    vector<int> ComputeVoronoiRegionsForNeurons(const KohonenNetwork &net, const vector<Point> &data_window);
    double ComputeDivergenceBetweenSOMSequences(const SOMSequence &som_sequence1, const SOMSequence &som_sequence2);
    vector<double> ComputePDFValuesForDivergenceComputation(const SOMSequence &som_sequence, vector<Point> *domain);
    double ComputePDFValueAtPointFromSOMSequence(const SOMSequence &som_sequence, const Point &pt);
    double ComputeKLDivergenceEstimatorBetweenPDFs(const vector<double> &pdf1, const vector<double> &pdf2);

};

#endif //SOMKE_SOMKEALGORITHM_H
