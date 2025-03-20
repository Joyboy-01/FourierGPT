import numpy as np
import pandas as pd
import argparse
import os
from typing import Union

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Extract features by linear interpolation
def get_features(wavelet_data: Union[str, pd.DataFrame], interp_len: int = 500):
    if isinstance(wavelet_data, str):
        df = pd.read_csv(wavelet_data)
    else:
        df = wavelet_data

    if 'sid' not in df.columns:
        df['sdiff'] = df['freq'] < df['freq'].shift(1, fill_value=0)
        df['sdiff'] = df['sdiff'].astype(int)
        df['sid'] = df['sdiff'].cumsum()

    features_interp = []
    for sid, group in df.groupby('sid'):
        freqs = group['freq'].values
        features = group['power'].values
        
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]
        features = features[sort_idx]
        
        freq_min, freq_max = min(freqs), max(freqs)
        new_freq = np.linspace(freq_min, freq_max, interp_len)
        new_feat = np.interp(new_freq, freqs, features)
        features_interp.append(new_feat)
    
    if len(features_interp) == 0:
        print("警告: 未能提取任何特征，请检查输入数据格式")
        return np.array([])
        
    return np.array(features_interp)


# 运行分类
def run_classification(human_wavelet_file: str, model_wavelet_file: str, k_best: int = 120):
    
    x_human = get_features(human_wavelet_file)
    y_human = np.zeros(x_human.shape[0])
    x_model = get_features(model_wavelet_file)
    y_model = np.ones(x_model.shape[0])
    
    x = np.concatenate([x_human, x_model], axis=0)
    y = np.concatenate([y_human, y_model], axis=0)
    
    cls = make_pipeline(
        StandardScaler(),
        SelectKBest(k=k_best),
        SVC(gamma='auto', kernel='rbf', C=1)
    )
    
    scores = cross_val_score(cls, x, y, cv=5)
    
    return scores


def main(args):
    scores = run_classification(
        args.human, 
        args.model, 
        k_best=args.k_best
    )
    print(f'Cross-validated acc: {scores}')
    print(f'Mean acc: {np.mean(scores)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--human', type=str, required=True, help='人类文本的小波变换数据文件')
    parser.add_argument('--model', type=str, required=True, help='模型生成文本的小波变换数据文件')
    parser.add_argument('--k_best', type=int, default=120, help='SelectKBest选择的特征数量')
    parser.add_argument('--save_intermid', action='store_true', default=False, help='Save intermidiate results')
    
    args = parser.parse_args()
    assert os.path.exists(args.human), f'File {args.human} does not exist'
    assert os.path.exists(args.model), f'File {args.model} does not exist'
    main(args)