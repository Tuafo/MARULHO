#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace {

__global__ void marulho_conditional_loop_tick(
    cudaGraphConditionalHandle handle,
    int *remaining
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int after_this_iteration = atomicSub(remaining, 1) - 1;
    cudaGraphSetConditional(
        handle,
        after_this_iteration > 0 ? 1u : 0u
    );
}

}  // namespace

extern "C" cudaError_t marulho_add_conditional_loop_kernel_node(
    cudaGraphNode_t *node,
    cudaGraph_t graph,
    const cudaGraphNode_t *dependencies,
    size_t dependency_count,
    cudaGraphConditionalHandle handle,
    int *remaining
) {
    cudaKernelNodeParams params{};
    cudaGraphConditionalHandle handle_arg = handle;
    int *remaining_arg = remaining;
    void *kernel_args[] = {&handle_arg, &remaining_arg};
    params.func = reinterpret_cast<void *>(marulho_conditional_loop_tick);
    params.gridDim = dim3(1, 1, 1);
    params.blockDim = dim3(1, 1, 1);
    params.sharedMemBytes = 0;
    params.kernelParams = kernel_args;
    params.extra = nullptr;
    return cudaGraphAddKernelNode(
        node,
        graph,
        dependencies,
        dependency_count,
        &params
    );
}
