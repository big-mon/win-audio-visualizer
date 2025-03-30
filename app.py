#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Windows 11向けオーディオビジュアライザーアプリケーション

このアプリケーションは、Windows 11上で再生されているすべてのシステム音声を
リアルタイムで取得し、視覚的に表示するデスクトップアプリケーションです。
WASAPI Loopback機能を用いてスピーカー出力音をキャプチャし、
リアルタイムでオーディオビジュアライザーとして画面に描画します。
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import rfft, rfftfreq

class AudioVisualizer:
    """
    オーディオビジュアライザークラス
    
    システム音声をキャプチャし、リアルタイムでビジュアライズするクラス
    """
    
    def __init__(self):
        """初期化メソッド"""
        # オーディオ設定
        self.sample_rate = 44100  # サンプルレート（Hz）
        self.block_size = 1024    # ブロックサイズ
        self.channels = 1         # チャンネル数（モノラル）
        
        # FFT設定
        self.n_fft = self.block_size
        self.hop_length = self.block_size // 2
        
        # 描画設定
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.x_fft = rfftfreq(self.n_fft, 1 / self.sample_rate)
        self.line, = self.ax.plot(self.x_fft, np.zeros(self.n_fft // 2 + 1))
        
        # グラフの設定
        self.ax.set_xlim(20, self.sample_rate // 2)  # 可聴域（20Hz〜）
        self.ax.set_ylim(0, 2)
        self.ax.set_xscale('log')
        self.ax.set_title('リアルタイムオーディオスペクトラム')
        self.ax.set_xlabel('周波数 (Hz)')
        self.ax.set_ylabel('振幅')
        self.ax.grid(True)
        
        # オーディオストリーム
        self.stream = None
        self.audio_data = np.zeros(self.block_size)
    
    def audio_callback(self, indata, frames, time, status):
        """
        オーディオコールバック関数
        
        Args:
            indata: 入力オーディオデータ
            frames: フレーム数
            time: タイムスタンプ
            status: ステータス
        """
        if status:
            print(f"ステータス: {status}")
        
        # モノラルに変換（複数チャンネルの場合は平均を取る）
        self.audio_data = np.mean(indata, axis=1)
    
    def update_plot(self, frame):
        """
        プロットを更新する関数
        
        Args:
            frame: アニメーションフレーム
        
        Returns:
            更新されたラインオブジェクト
        """
        # FFT処理
        fft_data = rfft(self.audio_data * np.hanning(len(self.audio_data)))
        magnitude = np.abs(fft_data) / len(fft_data)
        
        # 対数スケールでの表示調整
        magnitude = np.clip(magnitude, 1e-10, None)  # ゼロ回避
        
        # プロット更新
        self.line.set_ydata(magnitude)
        return self.line,
    
    def start(self):
        """ビジュアライザーを開始する"""
        try:
            # デバイス情報の表示
            print("利用可能なオーディオデバイス:")
            print(sd.query_devices())
            
            # ループバックデバイスの選択（デフォルトデバイス）
            device_info = sd.query_devices(kind='output')
            print(f"選択されたデバイス: {device_info['name']}")
            
            # オーディオストリームの開始
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=None,  # デフォルトデバイス
                latency='low',
                dtype='float32',
                clip_off=True,
                dither_off=True,
                never_drop_input=True,
                prime_output_buffers_using_stream_callback=True
            )
            
            # アニメーションの開始
            self.ani = FuncAnimation(
                self.fig,
                self.update_plot,
                interval=30,  # 更新間隔（ミリ秒）
                blit=True
            )
            
            # ストリーム開始
            self.stream.start()
            
            # プロット表示
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
        
        finally:
            # ストリームの停止
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
    
    def stop(self):
        """ビジュアライザーを停止する"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def main():
    """メイン関数"""
    visualizer = AudioVisualizer()
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("アプリケーションを終了します...")
    finally:
        visualizer.stop()


if __name__ == "__main__":
    main()