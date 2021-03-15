//
// Created by tomev on 14/03/2021.
//

#ifndef SOMKE_WTM_LOCALIZED_TRAINING_ALGORITHM_H
#define SOMKE_WTM_LOCALIZED_TRAINING_ALGORITHM_H

#include <cassert>
#include <algorithm>
#include <limits>

#include <boost/bind.hpp>

#include "training_functional.hpp"
#include "numeric_iterator.hpp"

/**
 * \file wtm_training_algorithm.hpp
 * \brief File contains template class Wtm_training_algoritm.
 * \ingroup neural_net
 */

namespace neural_net {
  /**
  * \addtogroup neural_net
  */
  /*\@{*/

  /**
   * \class Wtm_training_algorithm
   * \brief Class contains functionality for training
   * kohonen network using WTM method.
   * \param Network_type is a network type.
   * \param Value_type is a type os single data
   * in mathematical meaning, so it could be ::std::vector<double>, too.
   * \param Data_iterator_type is is iterator for container
   * with training data.
   * \param Training_functional_type is a type of functional.
   * \param Index_type is a type of index of neurons used in network.
   */
  template
      <
          typename Network_type,
          typename Value_type,
          typename Data_iterator_type,
          typename Training_functional_type,
          typename Index_type,
          typename Numeric_iterator_type
          = Linear_numeric_iterator<
              typename Training_functional_type::iteration_type
                                   >
      >
  class Wtm_localized_training_algorithm {

    public:

      typedef typename Training_functional_type::iteration_type iteration_type;
      typedef Network_type network_type;
      typedef Value_type value_type;
      typedef Data_iterator_type data_iterator_type;
      typedef Training_functional_type training_functional_type;
      typedef Index_type index_type;
      typedef Numeric_iterator_type numeric_iterator_type;
      /** Training functional. */
      Training_functional_type training_functional;

      /**
       * Constructor.
       * \param training_functional_ is a training functor.
       * \param numeric_iterator_ is a numeric iterator.
       */
      Wtm_localized_training_algorithm
          (
              Training_functional_type const &training_functional_,
              Numeric_iterator_type numeric_iterator_ = linear_numeric_iterator()
          )
          : training_functional(training_functional_),
            numeric_iterator(numeric_iterator_) {
        network = static_cast < Network_type * > ( 0 );
      }

      /**
       * Copy constructor.
       * It makes flat copy of neural network, so it copies only pointer not structure.
       */
      template
          <
              typename Network_type_2,
              typename Value_type_2,
              typename Data_iterator_type_2,
              typename Training_functional_type_2,
              typename Index_type_2,
              typename Numeric_iterator_type_2
          >
      inline Wtm_localized_training_algorithm
          (
              Wtm_training_algorithm
              <
              Network_type_2,
              Value_type_2,
              Data_iterator_type_2,
              Training_functional_type_2,
              Index_type_2,
              Numeric_iterator_type_2
              >

              const &wtm_training_alg_
          )
          :

          training_functional(wtm_training_alg_

                                  .training_functional),

          numeric_iterator(wtm_training_alg_

                               .numeric_iterator),

          iteration(wtm_training_alg_

                        .iteration) {
        network = wtm_training_alg_.network;
      }

/**
 * Function that starts training process.
 * \param network_ is a pointer to the existing kohonen neural network.
 * \param data_begin is a begin iterator, it could be revers.
 * \param data_end is end iterator.
 * \return error code.
 */
      ::boost::int32_t operator()
          (
              Data_iterator_type data_begin,
              Data_iterator_type data_end,
              Network_type *network_
          ) {
        network = network_;

        assert (network != static_cast < Network_type * > ( 0 ));

        // for each data train network
        ::std::for_each
            (
                data_begin, data_end,
                ::boost::bind
                    (
                        &Wtm_localized_training_algorithm
                            <
                                Network_type,
                                Value_type,
                                Data_iterator_type,
                                Training_functional_type,
                                Index_type,
                                Numeric_iterator_type
                            >::train,
                        this,
                        _1
                    )
            );

        return 0;
      }

    protected:

/** Pointer to the network. */
      Network_type *network;
      iteration_type iteration;
      Numeric_iterator_type numeric_iterator;

/**
 * Function trains neural network using single value.
 * \param value is a value.
 * As is set in WTM algorithm method is looking for the best neuron,
 * and train it to have better results with actual data in the future.
 */
      void train(Value_type const &value) {

        auto eta = training_functional.parameter;

        Index_type index_1 = Index_type();
        Index_type index_2 = Index_type();
        typename Network_type::value_type::result_type tmp_result;

        // reset max_result
        typename Network_type::value_type::result_type max_result
            = ::std::numeric_limits<
                typename Network_type::value_type::result_type
                                   >::min();
        typename Network_type::row_iterator r_iter;
        typename Network_type::column_iterator c_iter;

        // set ranges for iteration procedure
        ::boost::int32_t r_counter = 0;//network->objects.size();
        ::boost::int32_t c_counter = 0;//network->objects[0].size();

        auto max_weights = network->objects.begin()->begin()->weights;

        for(r_iter = network->objects.begin();
            r_iter != network->objects.end();
            ++r_iter
            ) {
          for(c_iter = r_iter->begin();
              c_iter != r_iter->end();
              ++c_iter
              ) {
            tmp_result = (*c_iter)(value);
            if(tmp_result > max_result) {
              index_1 = r_counter;
              index_2 = c_counter;
              max_result = tmp_result;
              max_weights = c_iter->weights;
            }
            ++c_counter;
          }
          ++r_counter;
          c_counter = 0;
        }

        r_counter = 0;
        c_counter = 0;

        // Start 1d assumption
        double m = 0.5;
        double d = 1.0;
        double delta_t = 1.0;
        double weight_value_difference = fabs(value[0] - max_weights[0]);
        // End 1d assumption

        double parameter_modifier = pow(1 / (delta_t * pow(weight_value_difference, d)), m);
        training_functional.parameter = eta * parameter_modifier;

        // train all neurons in network with respect to the
        // training functional
        for(r_iter = network->objects.begin();
            r_iter != network->objects.end();
            ++r_iter
            ) {
          for(c_iter = r_iter->begin();
              c_iter != r_iter->end();
              ++c_iter
              ) {
            (training_functional)
                (
                    c_iter->weights,
                    value,
                    iteration,
                    index_1, index_2, r_counter, c_counter
                );
            ++c_counter;
          }
          ++r_counter;
          c_counter = 0;
        }

        // increase iteration
        ++numeric_iterator;
        iteration = numeric_iterator();

        training_functional.parameter = eta;
      }

  };
/*\@}*/
} // namespace neural_net

#endif //SOMKE_WTM_LOCALIZED_TRAINING_ALGORITHM_H
