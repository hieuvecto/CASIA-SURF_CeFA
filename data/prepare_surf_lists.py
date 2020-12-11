import argparse
import os
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_path',
                        type=str,
                        default='./',
                        help='Path to your casia_surf dataset')
    args = parser.parse_args()

    df = pd.read_csv('./shuffled_train_list.csv')
    df['rgb_path'] = df.rgb_path.apply(lambda x: os.path.join(args.data_path, x))
    df['ir_path'] = df.ir_path.apply(lambda x: os.path.join(args.data_path, x))
    df['depth_path'] = df.depth_path.apply(lambda x: os.path.join(args.data_path, x))
    df.to_csv('./processed_shuffled_train_list.txt', index=False)

    df = pd.read_csv('./shuffled_val_list.csv')
    df['rgb_path'] = df.rgb_path.apply(lambda x: os.path.join(args.data_path, x))
    df['ir_path'] = df.ir_path.apply(lambda x: os.path.join(args.data_path, x))
    df['depth_path'] = df.depth_path.apply(lambda x: os.path.join(args.data_path, x))
    df.to_csv('./processed_shuffled_val_list.txt', index=False)

    df = pd.read_csv('./shuffled_test_list.csv')
    df['rgb_path'] = df.rgb_path.apply(lambda x: os.path.join(args.data_path, x))
    df['ir_path'] = df.ir_path.apply(lambda x: os.path.join(args.data_path, x))
    df['depth_path'] = df.depth_path.apply(lambda x: os.path.join(args.data_path, x))
    df.to_csv('./processed_shuffled_test_list.txt', index=False)

