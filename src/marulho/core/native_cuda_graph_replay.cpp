#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

#ifdef _WIN32

using cuda_graph_launch_t = cudaError_t(__cdecl *)(cudaGraphExec_t, cudaStream_t);
using cuda_graph_create_t = cudaError_t(__cdecl *)(cudaGraph_t *, unsigned int);
using cuda_graph_add_child_graph_node_t = cudaError_t(
    __cdecl *
)(
    cudaGraphNode_t *,
    cudaGraph_t,
    const cudaGraphNode_t *,
    size_t,
    cudaGraph_t
);
using cuda_graph_instantiate_t = cudaError_t(
    __cdecl *
)(cudaGraphExec_t *, cudaGraph_t, unsigned long long);
using cuda_graph_destroy_t = cudaError_t(__cdecl *)(cudaGraph_t);
using cuda_graph_exec_destroy_t = cudaError_t(__cdecl *)(cudaGraphExec_t);
using cuda_get_error_string_t = const char *(__cdecl *)(cudaError_t);

HMODULE load_cudart_module() {
    const char *names[] = {
        "cudart64_12.dll",
        "cudart64_13.dll",
        "cudart64_11.dll",
    };
    for (const char *name : names) {
        HMODULE module = GetModuleHandleA(name);
        if (module == NULL) {
            module = LoadLibraryA(name);
        }
        if (module != NULL) {
            return module;
        }
    }
    throw std::runtime_error("cudart runtime dll not available");
}

template <typename Fn>
Fn require_symbol(HMODULE module, const char *name) {
    auto pointer = reinterpret_cast<Fn>(GetProcAddress(module, name));
    if (pointer == nullptr) {
        throw std::runtime_error(std::string("missing cudart symbol: ") + name);
    }
    return pointer;
}

#else

using cuda_graph_launch_t = decltype(&cudaGraphLaunch);
using cuda_graph_create_t = decltype(&cudaGraphCreate);
using cuda_graph_add_child_graph_node_t = decltype(&cudaGraphAddChildGraphNode);
using cuda_graph_instantiate_t = decltype(&cudaGraphInstantiate);
using cuda_graph_destroy_t = decltype(&cudaGraphDestroy);
using cuda_graph_exec_destroy_t = decltype(&cudaGraphExecDestroy);
using cuda_get_error_string_t = decltype(&cudaGetErrorString);

#endif

struct RuntimeSymbols {
    cuda_graph_launch_t graph_launch = nullptr;
    cuda_graph_create_t graph_create = nullptr;
    cuda_graph_add_child_graph_node_t graph_add_child = nullptr;
    cuda_graph_instantiate_t graph_instantiate = nullptr;
    cuda_graph_destroy_t graph_destroy = nullptr;
    cuda_graph_exec_destroy_t graph_exec_destroy = nullptr;
    cuda_get_error_string_t get_error_string = nullptr;
};

RuntimeSymbols runtime_symbols() {
#ifdef _WIN32
    HMODULE module = load_cudart_module();
    return RuntimeSymbols{
        require_symbol<cuda_graph_launch_t>(module, "cudaGraphLaunch"),
        require_symbol<cuda_graph_create_t>(module, "cudaGraphCreate"),
        require_symbol<cuda_graph_add_child_graph_node_t>(
            module,
            "cudaGraphAddChildGraphNode"
        ),
        require_symbol<cuda_graph_instantiate_t>(module, "cudaGraphInstantiate"),
        require_symbol<cuda_graph_destroy_t>(module, "cudaGraphDestroy"),
        require_symbol<cuda_graph_exec_destroy_t>(module, "cudaGraphExecDestroy"),
        require_symbol<cuda_get_error_string_t>(module, "cudaGetErrorString"),
    };
#else
    return RuntimeSymbols{
        &cudaGraphLaunch,
        &cudaGraphCreate,
        &cudaGraphAddChildGraphNode,
        &cudaGraphInstantiate,
        &cudaGraphDestroy,
        &cudaGraphExecDestroy,
        &cudaGetErrorString,
    };
#endif
}

void check_cuda(cudaError_t error, const char *operation, const RuntimeSymbols &symbols) {
    if (error == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string(operation) + " failed: " + symbols.get_error_string(error)
    );
}

class RepeatedGraphExec {
public:
    RepeatedGraphExec(cudaGraph_t parent_graph, cudaGraphExec_t graph_exec, std::int64_t count)
        : parent_graph_(parent_graph), graph_exec_(graph_exec), count_(count) {}

    RepeatedGraphExec(const RepeatedGraphExec &) = delete;
    RepeatedGraphExec &operator=(const RepeatedGraphExec &) = delete;

    ~RepeatedGraphExec() {
        RuntimeSymbols symbols;
        try {
            symbols = runtime_symbols();
        } catch (...) {
            return;
        }
        if (graph_exec_ != nullptr) {
            symbols.graph_exec_destroy(graph_exec_);
        }
        if (parent_graph_ != nullptr) {
            symbols.graph_destroy(parent_graph_);
        }
    }

    cudaGraphExec_t graph_exec() const {
        return graph_exec_;
    }

    std::int64_t count() const {
        return count_;
    }

private:
    cudaGraph_t parent_graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    std::int64_t count_ = 0;
};

std::shared_ptr<RepeatedGraphExec> make_repeated_graph_exec_impl(
    std::uintptr_t child_graph_address,
    std::int64_t count
) {
    if (child_graph_address == 0) {
        throw std::invalid_argument("child_graph_address must be nonzero");
    }
    if (count <= 0) {
        throw std::invalid_argument("count must be positive");
    }
    RuntimeSymbols symbols = runtime_symbols();
    cudaGraph_t child_graph = reinterpret_cast<cudaGraph_t>(child_graph_address);
    cudaGraph_t parent_graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    check_cuda(symbols.graph_create(&parent_graph, 0), "cudaGraphCreate", symbols);
    cudaGraphNode_t previous = nullptr;
    try {
        for (std::int64_t index = 0; index < count; ++index) {
            cudaGraphNode_t node = nullptr;
            const cudaGraphNode_t *deps = previous == nullptr ? nullptr : &previous;
            size_t dep_count = previous == nullptr ? 0 : 1;
            check_cuda(
                symbols.graph_add_child(
                    &node,
                    parent_graph,
                    deps,
                    dep_count,
                    child_graph
                ),
                "cudaGraphAddChildGraphNode",
                symbols
            );
            previous = node;
        }
        check_cuda(
            symbols.graph_instantiate(&graph_exec, parent_graph, 0),
            "cudaGraphInstantiate",
            symbols
        );
    } catch (...) {
        if (graph_exec != nullptr) {
            symbols.graph_exec_destroy(graph_exec);
        }
        if (parent_graph != nullptr) {
            symbols.graph_destroy(parent_graph);
        }
        throw;
    }
    return std::make_shared<RepeatedGraphExec>(parent_graph, graph_exec, count);
}

void replay_repeated_graph_exec_impl(const RepeatedGraphExec &graph) {
    RuntimeSymbols symbols = runtime_symbols();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    check_cuda(
        symbols.graph_launch(graph.graph_exec(), stream),
        "cudaGraphLaunch",
        symbols
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<RepeatedGraphExec, std::shared_ptr<RepeatedGraphExec>>(
        module,
        "RepeatedGraphExec"
    )
        .def_property_readonly("count", &RepeatedGraphExec::count);
    module.def(
        "make_repeated_graph_exec",
        [](std::uintptr_t child_graph_address, std::int64_t count) {
            pybind11::gil_scoped_release release;
            return make_repeated_graph_exec_impl(child_graph_address, count);
        },
        "Instantiate a parent CUDA graph containing repeated child graph nodes."
    );
    module.def(
        "replay_repeated_graph_exec",
        [](const std::shared_ptr<RepeatedGraphExec> &graph) {
            pybind11::gil_scoped_release release;
            replay_repeated_graph_exec_impl(*graph);
        },
        "Replay a pre-instantiated repeated child-graph exec once."
    );
}
