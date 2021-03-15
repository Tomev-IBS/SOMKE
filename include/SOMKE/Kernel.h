//
// Created by tomev on 10/03/2021.
//

#ifndef SOMKE_KERNEL_H
#define SOMKE_KERNEL_H

#include <memory>
#include <vector>

typedef std::vector<double> Point;

class Kernel{
  public:
    virtual double GetValue(const Point &pt) = 0;
};

typedef std::shared_ptr<Kernel> KernelPtr;

#endif //SOMKE_KERNEL_H
