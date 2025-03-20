import pywt
from typing import Union
import numpy as np
import pandas as pd
import tqdm
import argparse
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP'] 
plt.rcParams['axes.unicode_minus'] = False 

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='', help='输入文件')
parser.add_argument('--output', '-o', type=str, default='', help='输出文件或目录')
parser.add_argument('--wavelet', '-w', type=str, default='db4', help='小波类型 (db4, morl等)')
parser.add_argument('--level', '-l', type=int, default=25, help='小波分解级别')
parser.add_argument('--mode', '-m', type=str, default='dwt', choices=['dwt', 'cwt'], help='小波变换模式: 离散(dwt)或连续(cwt)')
parser.add_argument('--preprocess', '-p', type=str, default='none', choices=['none', 'zscore', 'minmax', 'log', 'logzs'],help='预处理方法')
parser.add_argument('--scales', '-s', type=str, default='auto', help='CWT小波尺度，以逗号分隔或"auto"')
parser.add_argument('--require_sid', action='store_true', default=True, help='在输出文件中添加序列ID')
parser.add_argument('--plot', action='store_true', help='绘制结果')
parser.add_argument('--verbose', action='store_true', help='详细模式')


class WaveletProcessor(object):
    def __init__(self, wavelet_type='db4', mode='dwt', level=5, preprocess='none', scales='auto', require_sid=True, verbose=False):
        self.wavelet_type = wavelet_type
        self.mode = mode
        self.level = level
        self.preprocess = preprocess
        self.scales = scales
        self.require_sid = require_sid
        self.verbose = verbose
    
    def _read_data(self, data_file: str, N: int = np.inf):
        data = []
        with open(data_file, 'r') as f:
            count = 0
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                num = list(map(float, line.split()))
                data.append(num)
                count += 1
                if count >= N:
                    break
        return data
    
    def _preprocess(self, input_data: list):
        data = input_data.copy()
        if self.preprocess == 'zscore':
            data_zs = []
            epsilon = 1e-6
            for d in data:
                d = np.asarray(d)
                d_mean = np.mean(d)
                d_std = np.std(d)
                d_norm = (d - d_mean) / (d_std + epsilon)
                data_zs.append(d_norm)
            data = data_zs.copy()
        elif self.preprocess == 'minmax':
            data_mm = []
            for d in data:
                d = np.asarray(d)
                d_min = np.min(d)
                d_max = np.max(d)
                d_norm = (d - d_min) / (d_max - d_min)
                data_mm.append(d_norm)
            data = data_mm.copy()
        elif self.preprocess == 'log':
            data_log = []
            for d in data:
                d = np.asarray(d)
                d_log = np.log(d + 1)
                data_log.append(d_log)
            data = data_log.copy()
        elif self.preprocess == 'logzs':
            data_logzs = []
            epsilon = 1e-6
            for d in data:
                d = np.asarray(d)
                d_log = np.log(d + 1)
                d_mean = np.mean(d_log)
                d_std = np.std(d_log)
                d_norm = (d_log - d_mean) / (d_std + epsilon)
                data_logzs.append(d_norm)
            data = data_logzs.copy()
        elif self.preprocess != 'none':
            raise ValueError(f'Unknown preprocess method: {self.preprocess}. Please choose from [none, zscore, minmax, log, logzs].')
        return data
    
    def _dwt_batch(self, data: list, require_sid=False):
        coeffs, details, approxs, sids = [], [], [], []
        for i in tqdm.tqdm(range(len(data)), disable=not self.verbose):
            try:
                c = self._dwt(data[i])
                cA = c[0]
                cDs = c[1:]
                
                coeffs.append(c)
                for level, cD in enumerate(cDs):
                    details.append({
                        'level': self.level - level,
                        'energy': np.sum(cD**2),
                        'sid': i if require_sid else -1
                    })
                approxs.append({
                    'energy': np.sum(cA**2),
                    'sid': i if require_sid else -1
                })
                
                if require_sid:
                    sids.append(i)
                    
            except Exception as e:
                print(f'sample {i} error: {e}')
                raise
        
        df_details = pd.DataFrame(details)
        df_approxs = pd.DataFrame(approxs)
        return coeffs, df_details, df_approxs, sids if require_sid else None
    
    def _dwt(self, data: Union[np.ndarray, list]):
        """
        return: list of coefficients
        """
        if isinstance(data, list):
            data = np.asarray(data)
        coeffs = pywt.wavedec(data, self.wavelet_type, level=self.level)
        return coeffs
    
    def _cwt_batch(self, data: list, require_sid=False):
        """
        retuen: list of scalograms, list of scales
        """
        scalograms, scales_list, sids = [], [], []
        
        for i in tqdm.tqdm(range(len(data)), disable=not self.verbose):
            try:
                coef, scales = self._cwt(data[i])
                
                power = np.abs(coef)**2
                
                scalograms.append(power)
                scales_list.append(scales)
                
                if require_sid:
                    sids.append(i)
                    
            except Exception as e:
                print(f'样本 {i} 出错: {e}')
                raise
        
        return scalograms, scales_list, sids if require_sid else None
    
    def _cwt(self, data: Union[np.ndarray, list]):
        """
        return: coefficients, scales
        """
        if isinstance(data, list):
            data = np.asarray(data)
        if self.scales == 'auto':
            scales = np.logspace(0.5, 4, 50)
        else:
            scales = np.array([float(s) for s in self.scales.split(',')])
        coef, freqs = pywt.cwt(data, scales, self.wavelet_type)
        return coef, scales
    
    def _create_wavelet_df(self, mode, **kwargs):
        if mode == 'dwt':
            df_details = kwargs.get('df_details')
            df_approxs = kwargs.get('df_approxs')
            formatted_details = []
            for _, row in df_details.iterrows():
                formatted_details.append({
                    'sid': row['sid'],
                    'freq': row['level'],  
                    'power': row['energy']
                })
            
            formatted_approxs = []
            for _, row in df_approxs.iterrows():
                formatted_approxs.append({
                    'sid': row['sid'],
                    'freq': 0,
                    'power': row['energy']
                })
            
            combined_df = pd.DataFrame(formatted_details + formatted_approxs)
            combined_df = combined_df.sort_values(['sid', 'freq'], ascending=[True, False])
            return combined_df, df_approxs
        else: 
            scalograms = kwargs.get('scalograms')
            scales = kwargs.get('scales')
            sids = kwargs.get('sids')
            
            rows = []
            for i, (scalo, scale) in enumerate(zip(scalograms, scales)):
                sid = sids[i] if sids is not None else -1
                
                for k in range(scalo.shape[1]):
                    for j, s in enumerate(scale):
                        rows.append({
                            'sid': sid,
                            'freq': s, 
                            'power': scalo[j][k]
                        })
            
            return pd.DataFrame(rows)
    
    def _plot_results(self, data, results, output_prefix=None):
        if self.mode == 'dwt':
            coeffs, df_details, df_approxs, sids = results
            
            for i, c in enumerate(coeffs):
                plt.figure(figsize=(12, 8))
                
                plt.subplot(self.level + 2, 1, 1)
                plt.title(f'原始信号 (序列 {i})')
                plt.plot(data[i])
                
                plt.subplot(self.level + 2, 1, 2)
                plt.title(f'近似系数 (级别 {self.level})')
                plt.plot(c[0])
                
                for j, d in enumerate(c[1:]):
                    plt.subplot(self.level + 2, 1, j + 3)
                    plt.title(f'详细系数 (级别 {self.level - j})')
                    plt.plot(d)
                
                plt.tight_layout()
                
                if output_prefix:
                    plt.savefig(f"{output_prefix}_dwt_seq{i}.png")
                    plt.close()
                else:
                    plt.show()
            
            if sids is not None:
                plt.figure(figsize=(10, 6))
                
                level_groups = df_details.groupby('level')
                
                for level, group in level_groups:
                    energy_by_sid = group.groupby('sid')['energy'].mean()
                    plt.bar(energy_by_sid.index + (level - 1) * 0.1, 
                           energy_by_sid.values, 
                           width=0.1, 
                           label=f'Level {level}')
                
                plt.legend()
                plt.title('各级详细系数的能量分布')
                plt.xlabel('序列ID')
                plt.ylabel('能量')
                
                if output_prefix:
                    plt.savefig(f"{output_prefix}_dwt_energy.png")
                    plt.close()
                else:
                    plt.show()
                
        else: 
            scalograms, scales, sids = results
            for i, (scalo, scale) in enumerate(zip(scalograms[:min(3, len(scalograms))], scales)):
                plt.figure(figsize=(10, 8))
                
                plt.subplot(2, 1, 1)
                plt.title(f'原始信号 (序列 {i if sids is None else sids[i]})')
                plt.plot(data[i])
                
                plt.subplot(2, 1, 2)
                plt.title('连续小波变换时频图')
                plt.imshow(np.log1p(scalo), aspect='auto', cmap='viridis', 
                          extent=[0, len(data[i]), scale[-1], scale[0]])
                plt.colorbar(label='功率 (Log(1+x))')
                plt.ylabel('尺度')
                plt.xlabel('时间')
                
                plt.tight_layout()
                
                if output_prefix:
                    plt.savefig(f"{output_prefix}_cwt_seq{i}.png")
                    plt.close()
                else:
                    plt.show()
    
    def process(self, input_data: Union[str, list], plot=False, output_prefix=None):
        if isinstance(input_data, str):
            data_list = self._read_data(input_data)
            data = [np.asarray(d) for d in data_list]
        else:
            data = input_data.copy()

        data = self._preprocess(data)

        if self.mode == 'dwt':
            coeffs, df_details, df_approxs, sids = self._dwt_batch(data, require_sid=self.require_sid)
            results = (coeffs, df_details, df_approxs, sids)
            if plot:
                self._plot_results(data, results, output_prefix)
            combined_df, _ = self._create_wavelet_df('dwt', df_details=df_details, df_approxs=df_approxs)
            
            return combined_df
            
        elif self.mode == 'cwt':
            scalograms, scales, sids = self._cwt_batch(data, require_sid=self.require_sid)
            results = (scalograms, scales, sids)
            if plot:
                self._plot_results(data, results, output_prefix)
            
            df = self._create_wavelet_df('cwt', scalograms=scalograms, scales=scales, sids=sids)
            
            return df


def main(args):
    wavelet_processor = WaveletProcessor(
        wavelet_type=args.wavelet,
        mode=args.mode,
        level=args.level,
        preprocess=args.preprocess,
        scales=args.scales,
        require_sid=args.require_sid,
        verbose=args.verbose
    )
    
    # 处理输入数据
    if args.mode == 'dwt':
        df = wavelet_processor.process(args.input, plot=args.plot, 
                                      output_prefix=os.path.splitext(args.output)[0])
        df.to_csv(args.output, index=False)
    else: 
        df = wavelet_processor.process(args.input, plot=args.plot, 
                                     output_prefix=os.path.splitext(args.output)[0])
        df.to_csv(args.output, index=False)


if __name__ == '__main__':
    args = parser.parse_args() 
    main(args)
    
    
    
# 以下只尝试了morl基，且只尝试了zscore预处理
# 0.99
# python run_wavelet.py -i data/pubmed/pubmed_gpt-3.5.original.mistral.nll.txt     -o data/pubmed/pubmed_gpt-3.5.original.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_wavelet.py -i data/pubmed/pubmed_gpt-3.5.sampled.mistral.nll.txt     -o data/pubmed/pubmed_gpt-3.5.sampled.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_pwh_cls.py --human data/pubmed/pubmed_gpt-3.5.original.mistral.nllzs.waveletnull.txt --model data/pubmed/pubmed_gpt-3.5.sampled.mistral.nllzs.waveletnull.txt
# 0.9533
# python run_wavelet.py -i data/pubmed/pubmed_gpt-4.original.mistral.nll.txt     -o data/pubmed/pubmed_gpt-4.original.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_wavelet.py -i data/pubmed/pubmed_gpt-4.sampled.mistral.nll.txt     -o data/pubmed/pubmed_gpt-4.sampled.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_pwh_cls.py --human data/pubmed/pubmed_gpt-4.original.mistral.nllzs.waveletnull.txt --model data/pubmed/pubmed_gpt-4.sampled.mistral.nllzs.waveletnull.txt

# 0.9933
# python run_wavelet.py -i data/writing/writing_gpt-3.5.original.mistral.nll.txt     -o data/writing/writing_gpt-3.5.original.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_wavelet.py -i data/writing/writing_gpt-3.5.sampled.mistral.nll.txt     -o data/writing/writing_gpt-3.5.sampled.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_pwh_cls.py --human data/writing/writing_gpt-3.5.original.mistral.nllzs.waveletnull.txt --model data/writing/writing_gpt-3.5.sampled.mistral.nllzs.waveletnull.txt
# 0.9467
# python run_wavelet.py -i data/writing/writing_gpt-4.original.mistral.nll.txt     -o data/writing/writing_gpt-4.original.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_wavelet.py -i data/writing/writing_gpt-4.sampled.mistral.nll.txt     -o data/writing/writing_gpt-4.sampled.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_pwh_cls.py --human data/writing/writing_gpt-4.original.mistral.nllzs.waveletnull.txt --model data/writing/writing_gpt-4.sampled.mistral.nllzs.waveletnull.txt

# 0.9933
# python run_wavelet.py -i data/xsum/xsum_gpt-3.5.original.mistral.nll.txt     -o data/xsum/xsum_gpt-3.5.original.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_wavelet.py -i data/xsum/xsum_gpt-3.5.sampled.mistral.nll.txt     -o data/xsum/xsum_gpt-3.5.sampled.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_pwh_cls.py --human data/xsum/xsum_gpt-3.5.original.mistral.nllzs.waveletnull.txt --model data/xsum/xsum_gpt-3.5.sampled.mistral.nllzs.waveletnull.txt

# 0.8533
# python run_wavelet.py -i data/xsum/xsum_gpt-4.original.mistral.nll.txt     -o data/xsum/xsum_gpt-4.original.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_wavelet.py -i data/xsum/xsum_gpt-4.sampled.mistral.nll.txt     -o data/xsum/xsum_gpt-4.sampled.mistral.nllzs.waveletnull.txt -p zscore -m cwt -w morl
# python run_pwh_cls.py --human data/xsum/xsum_gpt-4.original.mistral.nllzs.waveletnull.txt --model data/xsum/xsum_gpt-4.sampled.mistral.nllzs.waveletnull.txt
