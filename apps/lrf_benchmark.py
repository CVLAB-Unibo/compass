import argparse
import yaml
from benchmarks import lrf_benchmark as lrf_bench


def parse_commandline():

    parser = argparse.ArgumentParser(description="Test local reference frame on 3DMatchBenchmark, StanfordViews or ETH.")

    parser.add_argument("--config_file", type=str, required=True, help="Configuration file for the benchmark.")
    parser.add_argument("--path_results", type=str, required=True, help="Path to save the results.")
    
    # Dataset
    parser.add_argument("--path_ds", type=str, required=True, help="Path to dataset.")
    
    # Load network weights
    parser.add_argument("--path_ckp_layer_s2", type=str, required=True, help="Path to trained S2 layer.")
    parser.add_argument("--path_ckp_layer_lrf", type=str, required=True, help="Path to trained lrf layer.")
    

    # Misc
    parser.add_argument("--is_batch", type=int, default=1, help="Set to 0 to enable visualization.")
    
    parser.add_argument("--use_gpu", type=int, default=1, help="Default 1, set it to 0 to use the CPU only. WARNING: this slows down the computation, use only for testing purposes or if you don't have a CUDA capable GPU.")
    parser.add_argument("--id_gpu", type=int, default=0, help="Gpu ID")
    parser.add_argument("--size_batch", type=int, default=8, help="Size of batch for forward pass. Affects GPU VRAM occupation.")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers for the PyTorch dataloader (#CPU Cores - 1).")
    
    return parser.parse_args()


def main(args):

    benchmark = lrf_bench.LocalReferenceFrameBenchmark(args)

    benchmark.prepare()

    benchmark.start()

    benchmark.print_results()
    benchmark.store_results()

    benchmark.shutdown()


if __name__ == "__main__":

    arguments = parse_commandline()

    with open(arguments.config_file) as config_file:
        params = yaml.load(config_file, Loader=yaml.FullLoader)
        print(params)
        for key, value in params.items():
            setattr(arguments, key, value)

    main(arguments)

