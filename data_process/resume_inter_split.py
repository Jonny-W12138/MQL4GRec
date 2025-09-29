import argparse
import collections
import os
from utils import load_json, check_path

def load_inters_from_json(json_file):
    all_inters = load_json(json_file)
    train_inters, valid_inters, test_inters = {}, {}, {}
    for u_index, items in all_inters.items():
        u_index = int(u_index)
        train_inters[u_index] = [str(i) for i in items[:-2]]
        valid_inters[u_index] = [str(items[-2])]
        test_inters[u_index] = [str(items[-1])]
    return train_inters, valid_inters, test_inters

def convert_to_atomic_files(args, train_data, valid_data, test_data):
    print('转换数据集：')
    print(' 数据集：', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='数据集名称，例如 Sports')
    parser.add_argument('--output_path', type=str, default='./datasets/LC-Rec_image')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    inter_json_path = os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter.json')
    if not os.path.exists(inter_json_path):
        print(f'错误：{inter_json_path} 不存在！')
    else:
        print(f'加载 {inter_json_path}')
        train_inters, valid_inters, test_inters = load_inters_from_json(inter_json_path)
        check_path(os.path.join(args.output_path, args.dataset))
        convert_to_atomic_files(args, train_inters, valid_inters, test_inters)