#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" cudaError_t marulho_add_conditional_loop_kernel_node(
    cudaGraphNode_t *node,
    cudaGraph_t graph,
    const cudaGraphNode_t *dependencies,
    size_t dependency_count,
    cudaGraphConditionalHandle handle,
    int *remaining
);

namespace {

void check_cuda(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(error)
    );
}

class ConditionalLoopGraphExec {
public:
    ConditionalLoopGraphExec(
        cudaGraph_t parent_graph,
        cudaGraphExec_t graph_exec,
        at::Tensor counter,
        std::int64_t count
    )
        : parent_graph_(parent_graph),
          graph_exec_(graph_exec),
          counter_(std::move(counter)),
          count_(count) {}

    ConditionalLoopGraphExec(const ConditionalLoopGraphExec &) = delete;
    ConditionalLoopGraphExec &operator=(const ConditionalLoopGraphExec &) = delete;

    ~ConditionalLoopGraphExec() {
        if (graph_exec_ != nullptr) {
            cudaGraphExecDestroy(graph_exec_);
        }
        if (parent_graph_ != nullptr) {
            cudaGraphDestroy(parent_graph_);
        }
    }

    cudaGraphExec_t graph_exec() const {
        return graph_exec_;
    }

    std::int64_t count() const {
        return count_;
    }

    std::uintptr_t counter_address() const {
        return reinterpret_cast<std::uintptr_t>(counter_.data_ptr<int>());
    }

private:
    cudaGraph_t parent_graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    at::Tensor counter_;
    std::int64_t count_ = 0;
};

std::shared_ptr<ConditionalLoopGraphExec> make_conditional_loop_graph_exec_impl(
    std::uintptr_t child_graph_address,
    std::int64_t count
) {
    if (child_graph_address == 0) {
        throw std::invalid_argument("child_graph_address must be nonzero");
    }
    if (count <= 0) {
        throw std::invalid_argument("count must be positive");
    }
    if (count > 32) {
        throw std::invalid_argument("count must be <= 32 for this prototype");
    }

    cudaGraph_t child_graph = reinterpret_cast<cudaGraph_t>(child_graph_address);
    at::Tensor counter = at::empty(
        {1},
        at::TensorOptions()
            .device(c10::DeviceType::CUDA)
            .dtype(c10::ScalarType::Int)
    );

    cudaGraph_t parent_graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    check_cuda(cudaGraphCreate(&parent_graph, 0), "cudaGraphCreate");
    try {
        cudaGraphConditionalHandle handle = 0;
        check_cuda(
            cudaGraphConditionalHandleCreate(
                &handle,
                parent_graph,
                1,
                cudaGraphCondAssignDefault
            ),
            "cudaGraphConditionalHandleCreate"
        );

        cudaGraphNode_t counter_init_node = nullptr;
        cudaMemsetParams memset_params{};
        memset_params.dst = counter.data_ptr<int>();
        memset_params.pitch = 0;
        memset_params.value = static_cast<unsigned int>(count);
        memset_params.elementSize = sizeof(int);
        memset_params.width = 1;
        memset_params.height = 1;
        check_cuda(
            cudaGraphAddMemsetNode(
                &counter_init_node,
                parent_graph,
                nullptr,
                0,
                &memset_params
            ),
            "cudaGraphAddMemsetNode"
        );

        cudaGraphNode_t conditional_node = nullptr;
        cudaGraphNodeParams conditional_params{};
        conditional_params.type = cudaGraphNodeTypeConditional;
        conditional_params.conditional.handle = handle;
        conditional_params.conditional.type = cudaGraphCondTypeWhile;
        conditional_params.conditional.size = 1;
        check_cuda(
            cudaGraphAddNode(
                &conditional_node,
                parent_graph,
                &counter_init_node,
                nullptr,
                1,
                &conditional_params
            ),
            "cudaGraphAddNode"
        );

        cudaGraph_t body_graph = conditional_params.conditional.phGraph_out[0];
        cudaGraphNode_t child_node = nullptr;
        check_cuda(
            cudaGraphAddChildGraphNode(
                &child_node,
                body_graph,
                nullptr,
                0,
                child_graph
            ),
            "cudaGraphAddChildGraphNode"
        );

        cudaGraphNode_t loop_node = nullptr;
        check_cuda(
            marulho_add_conditional_loop_kernel_node(
                &loop_node,
                body_graph,
                &child_node,
                1,
                handle,
                counter.data_ptr<int>()
            ),
            "marulho_add_conditional_loop_kernel_node"
        );

        check_cuda(
            cudaGraphInstantiate(&graph_exec, parent_graph, 0),
            "cudaGraphInstantiate"
        );
    } catch (...) {
        if (graph_exec != nullptr) {
            cudaGraphExecDestroy(graph_exec);
        }
        if (parent_graph != nullptr) {
            cudaGraphDestroy(parent_graph);
        }
        throw;
    }
    return std::make_shared<ConditionalLoopGraphExec>(
        parent_graph,
        graph_exec,
        counter,
        count
    );
}

void replay_conditional_loop_graph_exec_impl(
    const ConditionalLoopGraphExec &graph
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    check_cuda(
        cudaGraphLaunch(graph.graph_exec(), stream),
        "cudaGraphLaunch"
    );
}

pybind11::dict cuda_graph_sequence_capabilities_impl() {
    pybind11::dict capabilities;
    capabilities["executor"] = "cuda_graph_conditional_while";
    capabilities["compiled"] = true;
    capabilities["cudart_version"] = CUDART_VERSION;
    capabilities["max_tokens"] = 32;
    capabilities["has_conditional_nodes"] = true;
    capabilities["has_device_conditional_setter"] = true;
    return capabilities;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<
        ConditionalLoopGraphExec,
        std::shared_ptr<ConditionalLoopGraphExec>
    >(module, "ConditionalLoopGraphExec")
        .def_property_readonly("count", &ConditionalLoopGraphExec::count)
        .def_property_readonly(
            "counter_address",
            &ConditionalLoopGraphExec::counter_address
        );
    module.def(
        "make_conditional_loop_graph_exec",
        [](std::uintptr_t child_graph_address, std::int64_t count) {
            pybind11::gil_scoped_release release;
            return make_conditional_loop_graph_exec_impl(
                child_graph_address,
                count
            );
        },
        "Instantiate a conditional-WHILE parent graph around a child graph."
    );
    module.def(
        "replay_conditional_loop_graph_exec",
        [](const std::shared_ptr<ConditionalLoopGraphExec> &graph) {
            pybind11::gil_scoped_release release;
            replay_conditional_loop_graph_exec_impl(*graph);
        },
        "Replay a conditional-WHILE parent CUDA graph once."
    );
    module.def(
        "cuda_graph_sequence_capabilities",
        &cuda_graph_sequence_capabilities_impl,
        "Report compiled CUDA graph sequence executor capabilities."
    );
}
