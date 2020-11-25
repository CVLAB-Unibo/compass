import argparse
from benchmarks import lrf_benchmark as lrf_bench


def parse_commandline():

    parser = argparse.ArgumentParser(description="Test local reference frame on 3DMatchBenchmark, StanfordViews or ETH.")

    parser.add_argument("--is_batch", type=int, required=True, help="Enable visualization.")
    parser.add_argument("--id_gpu", type=int, required=True, help="Gpu ID")
    parser.add_argument("--ext_cloud", type=str, default="ply", help="Point cloud files extension.")
    parser.add_argument('--encoder_bandwidths', type=int, nargs='+', required=False, help='List of input and ouput bandwidths for the encoder')
    parser.add_argument('--encoder_features', type=int, nargs='+', required=False, help='List of features size for encoder')
    parser.add_argument("--leaf", type=float, required=True, help="Leaf for subsampling.")
    parser.add_argument('--lrf_bandwidths', type=int, nargs='+', required='True', help='List of input and ouput bandwidths for the lrf layer')
    parser.add_argument('--lrf_features', type=int, nargs='+', required='True', help='List of features size for the lrf layer.')
    parser.add_argument("--path_ckp_layer_s2", type=str, required=False, help="Path to trained s2 layer.")
    parser.add_argument("--path_ckp_layer_lrf", type=str, required=True, help="Path to trained local reference frame network.")
    parser.add_argument("--path_ds", type=str, required=True, help="Path to dataset.")
    parser.add_argument("--path_results", type=str, required=True, help="Path to .csv file with results.")
    parser.add_argument("--min_distance", type=float, required=True, help="Min distance for perfect detector.")
    parser.add_argument("--radius_detector", type=float, required=True, help="Radius detector.")
    parser.add_argument("--radius_descriptor", type=float, required=True, help="Radius descriptor.")
    parser.add_argument("--size_batch", type=int, required=True, help="Size of batch for forward pass.")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for the PyTorch dataloader.")
    parser.add_argument("--size_point_cloud", type=int, required=True, help="Size of input point cloud for descriptor.")
    parser.add_argument("--th_cosine", type=float, required=True, help="Threshold for cosine.")
    parser.add_argument("--softmax_temp", type=float, default=1.0, help="Softmax temperature.")
    parser.add_argument("--use_simple_softargmax", type=int, default=0, help="Select the activation function of the LRF Estimator. Default SoftArgMax.")
    parser.add_argument("--use_gpu", type=int, default=1, help="Default 1, set it to 0 to use the CPU only. WARNING: this slows down the computation, use only for testing purposes or if you don't have a CUDA capable GPU.")
    
    parser.add_argument("--name_data_set", type=str, default="3DMatch", help="Name of dataset: 3DMatch or StanfordViews.")
    parser.add_argument("--name_file_gt", type=str,  default="gt.log", help="Name of the file containing the ground truth poses.")
    parser.add_argument("--name_file_overlap", type=str, required=False, default=None, help="Name of the file containing the overlapping areas. Not used for 3DMatch")
    parser.add_argument("--overlap_threshold", type=float, required=False, default=0.10, help="Minimum overlap required to consider a pair of clouds. Works for StanfordViews only.")

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
    main(arguments)

