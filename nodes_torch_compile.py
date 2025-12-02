from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.torch_helpers import set_torch_compile_wrapper

class TorchCompileModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TorchCompileModel",
            category="_for_testing",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "backend",
                    options=["inductor", "cudagraphs"],
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, backend) -> io.NodeOutput:
        m = model.clone()
        compile_kwargs = {
            "backend": "inductor",
            "mode": "max-autotune",
            "fullgraph": True,
            "dynamic": False,  # MUST be False for consistent latents
            "options": {
                # Core optimizations
                "max_autotune": True,
                "max_autotune_gemm": True,
                "max_autotune_gemm_backends": "TRITON",  # Removed CUTLASS to eliminate warnings
                
                # Memory management for 18GB models
                "triton.cudagraphs": False,
                "shape_padding": True,
                "epilogue_fusion": True,
                
                # Caching for faster subsequent compiles
                "fx_graph_cache": True,
                "autotune_local_cache": True,
                
                # Additional performance
                "coordinate_descent_tuning": True,
                "coordinate_descent_check_all_directions": True,
            }
        }
        set_torch_compile_wrapper(model=m, backend=backend, **compile_kwargs)
        return io.NodeOutput(m)


class TorchCompileExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TorchCompileModel,
        ]


async def comfy_entrypoint() -> TorchCompileExtension:
    return TorchCompileExtension()
