//! Milestone-0001 (agentic-yarn-vllm) acceptance test for the cuda-oxide sm_75
//! codegen port. A Turing-safe persistent/cooperative kernel: cooperative launch
//! + `grid::sync()` + cross-block visibility after the grid-wide barrier. It uses
//! NO sm_80-only ops (no redux.sync / bf16x2 / stmatrix / mbarrier / cluster), so
//! the patched compiler classifies it `DetectedFeatures::Basic`, emits `.target
//! sm_75`, and (with an sm_75 device hint) honours it rather than downgrading.
//!
//! Expected, when the patch is correct:
//!   - emitted PTX header is `.target sm_75` (CUDA_OXIDE_VERBOSE=1 logs the source)
//!   - `ptxas -arch=sm_75` accepts it; `cuModuleLoadData` JIT-loads with no
//!     INVALID_PTX; the cooperative launch runs and every block observes the
//!     full barrier-flushed marker sum.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cooperative_launch, grid, kernel, thread};
use cuda_host::cuda_module;

// `#[cooperative_launch]` makes the generated launch method submit a cooperative
// launch (`cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`), which
// `grid::sync()` requires. cuda-oxide supports this on sm_75 (cc 7.5 has
// CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH).
#[cuda_module]
mod grid_sync_kernels {
    use super::*;

    /// Each block's thread 0 writes a marker (`blockIdx.x + 1`); the grid
    /// synchronises; thread 0 then sums every block's marker via the base
    /// pointer and writes it to `out[blockIdx.x]`. Every block must observe the
    /// same total `gridDim.x * (gridDim.x + 1) / 2` — proving the grid-wide
    /// barrier and cross-block memory visibility.
    #[kernel]
    #[cooperative_launch]
    pub fn test_grid_sync(mut markers: DisjointSlice<u32>, mut out: DisjointSlice<u32>) {
        let block_id = thread::blockIdx_x();
        let n = thread::gridDim_x();

        if thread::threadIdx_x() == 0 {
            unsafe {
                *markers.get_unchecked_mut(block_id as usize) = block_id + 1;
            }
        }

        grid::sync();

        if thread::threadIdx_x() == 0 {
            let base = markers.as_mut_ptr() as *const u32;
            let mut sum: u32 = 0;
            let mut i: u32 = 0;
            while i < n {
                unsafe {
                    sum = sum.wrapping_add(*base.add(i as usize));
                }
                i += 1;
            }
            unsafe {
                *out.get_unchecked_mut(block_id as usize) = sum;
            }
        }
    }
}

fn main() {
    println!("=== sm_75 cooperative grid.sync acceptance ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = ctx
        .load_module_from_file("sm75_coop_smoke.ptx")
        .expect("Failed to load PTX module");

    let grid_sync_module =
        grid_sync_kernels::from_module(module.clone()).expect("Failed to init typed module");

    const BLOCKS: u32 = 32;
    let block_threads = 128u32;
    let coop_cfg = LaunchConfig {
        block_dim: (block_threads, 1, 1),
        grid_dim: (BLOCKS, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut markers = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();
    let mut sums = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();

    grid_sync_module
        .test_grid_sync(stream.as_ref(), coop_cfg, &mut markers, &mut sums)
        .expect("test_grid_sync cooperative launch failed");

    let host = sums.to_host_vec(&stream).unwrap();
    let expected: u32 = (1..=BLOCKS).sum();
    let ok = host.iter().all(|&s| s == expected);
    println!("  expected sum = {expected}; all blocks agree = {ok}");
    if !ok {
        println!("  observed sums: {host:?}");
        std::process::exit(1);
    }
    println!("PASS: sm_75 cooperative grid.sync verified on device");
}
