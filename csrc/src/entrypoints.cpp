#include <torch/extension.h>

#include "block_swapping.h"

PYBIND11_MODULE(swiftllm_c, m) {
  m.def("swap_blocks", &swap_blocks);
}
