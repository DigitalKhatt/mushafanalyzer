#pragma once

#include "SegmentationTypes.h"

// Computes word-independent probabilities from component geometry, contour
// shape, and position relative to the detected baseline.
void classifyComponentsPreliminary(PageModel& page);
