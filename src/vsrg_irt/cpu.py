import os


# CPU / threading --------------------------------------------------------------
def configure_cpu(threads_per_worker: int) -> None:
    t = str(max(1, threads_per_worker))
    os.environ.setdefault("OMP_NUM_THREADS", t)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", t)
    os.environ.setdefault("MKL_NUM_THREADS", t)
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")