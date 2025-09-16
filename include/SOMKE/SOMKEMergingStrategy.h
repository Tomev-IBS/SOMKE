//
// Created by tomev on 12/03/2021.
//

#ifndef DEDSTA_SOMKEMERGINGSTRATEGY_H
#define DEDSTA_SOMKEMERGINGSTRATEGY_H

#include "SOMSequenceEntry.h"

class SOMKEMergingStrategy{
  public:
    virtual bool ShouldMergeBePerformed(const vector<SOMSequenceEntry> &entries) const = 0;
    virtual size_t FindEntryToMerge(const vector<SOMSequenceEntry> &entries) const = 0;
    virtual void SetDataWindowIterator(int *iterator) = 0;
};

typedef std::shared_ptr<SOMKEMergingStrategy> MergingStrategyPtr;

#endif //DEDSTA_SOMKEMERGINGSTRATEGY_H
