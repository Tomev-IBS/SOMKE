//
// Created by tomev on 07/03/2021.
//

#ifndef SOMKE_SOMSEQUENCE_H
#define SOMKE_SOMSEQUENCE_H

#include <vector>
#include <neural_net_headers.hpp>

using std::vector;
typedef std::vector<double> Point;

typedef neural_net::Cauchy_function<Point::value_type ,Point::value_type , int> ActivationFunction;
typedef distance::Euclidean_distance_function<Point> DistanceFunction;
typedef neural_net::Basic_neuron<ActivationFunction, DistanceFunction> Neuron;
typedef neural_net::Rectangular_container<Neuron> KohonenNetwork;
typedef neural_net::Max_topology<::boost::int32_t> Topology;
typedef neural_net::Gauss_function<double, double, ::boost::int32_t> SpaceFunctor;
typedef neural_net::Gauss_function<::boost::int32_t, Point::value_type , ::boost::int32_t> NetFunctor;
typedef neural_net::Classic_training_weight<Point, ::boost::int32_t, NetFunctor, SpaceFunctor, Topology, DistanceFunction, ::boost::int32_t > TrainingWeight;
typedef neural_net::Wtm_classical_training_functional<Point, double, ::boost::int32_t, ::boost::int32_t, TrainingWeight> TrainingFunctional;
typedef neural_net::Wtm_training_algorithm<KohonenNetwork, Point, vector<Point>::iterator, TrainingFunctional, size_t> TrainingAlgorithm;

#include <utility>

using std::pair;

struct SOMSequence{
  public:
    KohonenNetwork net;// SOM
    vector<int> voronoi_regions; // vector of numbers of input data in the Voronoi regions of neurons
    double kl_divergence;// kl divergence
    pair<int, int> range; // range of windows summarized
};

#endif //SOMKE_SOMSEQUENCE_H
