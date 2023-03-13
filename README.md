# xla_fp8_testing

## To run fp8 + mixed precision
TF_CPP_VMODULE=gemm_rewriter=1 TF_XLA_FLAGS="--tf_xla_auto_jit=2" XLA_FLAGS="--xla_gpu_enable_cublaslt=true" python benchmark.py --fp8 --mixed

or dump HLOs

TF_DUMP_GRAPH_PREFIX=/tmp/generated TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=/tmp/generated --xla_gpu_enable_cublaslt=true --xla_gpu_simplify_all_fp_conversions=false --xla_dump_hlo_pass_re=.*" python benchmark.py --fp8 --mixed
