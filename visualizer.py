import argparse, cv2

def parser():

    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--pred_file_path_gt', type=str, required=True, help='path to the orig file containing predictions')
    arg_parser.add_argument('--pred_file_path_fc', type=str, required=True, help='path to the orig fc file containing predictions')
    arg_parser.add_argument('--pred_file_path_cnn', type=str, required=True, help='path to the orig cnn file containing predictions')
    arg_parser.add_argument('--pred_file_path_pre', type=str, required=True, help='path to the orig pre file containing predictions')
    arg_parser.add_argument('--pred_file_path_seq', type=str, required=True, help='path to the orig seq file containing predictions')
    arg_parser.add_argument('--path_to_video', type=str, required=True, help='path to the input video file')
    arg_parser.add_argument('--output_path', type=str, required=True, help='Output dir path.')

    return arg_parser.parse_args()

if __name__=='__main__':
    args = parser()

    