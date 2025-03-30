#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Windows Audio Session API (WASAPI)を使用したオーディオビジュアライザー
PC上で再生されている音声をキャプチャして可視化します
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import rfft, rfftfreq
import queue
import threading
import time

class AudioVisualizer:
    """
    WASAPIを使用したオーディオビジュアライザークラス
    Windowsで再生されている音声をキャプチャして可視化します
    """
    
    def __init__(self, device=None, window_size=2048, sample_rate=44100, channels=2):
        """
        初期化メソッド
        
        Parameters
        ----------
        device : int or str, optional
            使用するオーディオデバイスのID または 名前
        window_size : int, optional
            FFTウィンドウサイズ
        sample_rate : int, optional
            サンプリングレート
        channels : int, optional
            チャンネル数
        """
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.q = queue.Queue()
        self.plot_data = np.zeros(window_size)
        self.spectrum_data = np.zeros(window_size // 2 + 1)
        self.stream = None
        self.running = False
        self.fig = None
        self.animation = None
        
        # ビジュアライザーの設定
        self.spectrum_x = rfftfreq(window_size, 1 / sample_rate)
        self.spectrum_max = 0.1  # スペクトラムの最大値（自動調整用）
        self.wave_max = 0.1      # 波形の最大値（自動調整用）
        self.smoothing_factor = 0.2  # データの平滑化係数
    
    def audio_callback(self, indata, frames, time, status):
        """
        オーディオコールバック関数
        
        Parameters
        ----------
        indata : ndarray
            入力オーディオデータ
        frames : int
            フレーム数
        time : CData
            タイムスタンプ
        status : CallbackFlags
            ステータスフラグ
        """
        if status:
            print(f"ステータス: {status}")
        # モノラルに変換してキューに追加
        if self.channels > 1:
            self.q.put(np.mean(indata, axis=1))
        else:
            self.q.put(indata.copy())
    
    def find_loopback_device(self):
        """
        ループバックデバイスを検索する
        
        Returns
        -------
        int or None
            ループバックデバイスのID、見つからない場合はNone
        """
        devices = sd.query_devices()
        
        # 1. 'Loopback'という名前が含まれるデバイスを優先的に探す
        for i, device in enumerate(devices):
            if 'loopback' in device['name'].lower() and device['max_input_channels'] > 0:
                print(f"ループバックデバイスを検出しました: {device['name']}")
                return i
                
        # 2. 'ループバック'という名前が含まれるデバイスを探す（日本語環境向け）
        for i, device in enumerate(devices):
            if 'ループバック' in device['name'] and device['max_input_channels'] > 0:
                print(f"ループバックデバイスを検出しました: {device['name']}")
                return i
        
        # 3. 'WASAPI'という名前が含まれる入力デバイスを探す
        for i, device in enumerate(devices):
            if 'wasapi' in device['name'].lower() and device['max_input_channels'] > 0:
                print(f"WASAPIデバイスを検出しました: {device['name']}")
                return i
        
        # 4. 'プライマリ サウンド キャプチャ'という名前のデバイスを探す（Windows向け）
        for i, device in enumerate(devices):
            if 'プライマリ サウンド キャプチャ' in device['name'] and device['max_input_channels'] > 0:
                print(f"プライマリサウンドキャプチャデバイスを検出しました: {device['name']}")
                return i
                
        # 5. 入力チャンネルを持つデバイスを探す（最終手段）
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"入力デバイスを検出しました: {device['name']}")
                return i
        
        return None
    
    def list_audio_devices(self):
        """
        利用可能なオーディオデバイスを一覧表示
        
        Returns
        -------
        list
            デバイス情報のリスト
        """
        devices = sd.query_devices()
        print("利用可能なオーディオデバイス:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} (入力チャンネル: {device['max_input_channels']}, 出力チャンネル: {device['max_output_channels']})")
        return devices
    
    def start_capture(self, device_id=None):
        """
        オーディオキャプチャを開始
        
        Parameters
        ----------
        device_id : int, optional
            使用するデバイスID（指定がない場合は自動検出）
        """
        if self.running:
            print("すでにキャプチャ中です")
            return
        
        # デバイスが指定されている場合はそれを使用、なければ自動検出
        if device_id is not None:
            self.device = device_id
        elif self.device is None:
            self.device = self.find_loopback_device()
            
        if self.device is None:
            print("適切なオーディオキャプチャデバイスが見つかりませんでした。デバイス一覧:")
            self.list_audio_devices()
            print("\nデバイスIDを指定して再試行してください。例: visualizer.start_capture(2)")
            return
        
        try:
            device_info = sd.query_devices(self.device)
            print(f"使用するデバイス: {device_info['name']}")
            
            # デバイスのサンプルレートを使用
            if 'default_samplerate' in device_info:
                self.sample_rate = int(device_info['default_samplerate'])
                # サンプルレートが変更された場合、スペクトラムのX軸も更新
                self.spectrum_x = rfftfreq(self.window_size, 1 / self.sample_rate)
            
            # ストリームを開始
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=self.window_size,
                latency='low',
                dtype='float32'
            )
            self.stream.start()
            self.running = True
            print(f"オーディオキャプチャを開始しました（サンプルレート: {self.sample_rate}Hz）")
        except Exception as e:
            print(f"オーディオキャプチャの開始に失敗しました: {e}")
    
    def stop_capture(self):
        """
        オーディオキャプチャを停止
        """
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.running = False
            print("オーディオキャプチャを停止しました")
    
    def get_audio_data(self):
        """
        キューからオーディオデータを取得
        
        Returns
        -------
        ndarray
            オーディオデータ
        """
        try:
            data = self.q.get_nowait()
            return data
        except queue.Empty:
            return np.zeros(self.window_size)
    
    def update_visualizer(self, frame):
        """
        ビジュアライザーの更新関数
        
        Parameters
        ----------
        frame : int
            アニメーションフレーム番号
            
        Returns
        -------
        list
            更新されたmatplotlibのアーティストのリスト
        """
        if not self.running:
            return []
        
        # オーディオデータを取得
        audio_data = self.get_audio_data()
        
        # 波形データを更新（平滑化）
        self.plot_data = self.smoothing_factor * audio_data + (1 - self.smoothing_factor) * self.plot_data
        
        # スペクトラムデータを計算
        if audio_data.size > 0:
            # 窓関数を適用（ハミング窓）
            windowed_data = audio_data * np.hamming(len(audio_data))
            # FFTを計算
            spectrum = np.abs(rfft(windowed_data)) / len(windowed_data)
            # 対数スケールに変換（小さな値を避けるために1e-10を加算）
            log_spectrum = 20 * np.log10(spectrum + 1e-10)
            # スペクトラムデータを更新（平滑化）
            self.spectrum_data = self.smoothing_factor * log_spectrum + (1 - self.smoothing_factor) * self.spectrum_data
        
        # 最大値を更新（自動スケーリング用）
        current_max = np.max(np.abs(self.plot_data))
        if current_max > self.wave_max:
            self.wave_max = current_max
        else:
            self.wave_max = 0.99 * self.wave_max + 0.01 * current_max
        
        current_spectrum_max = np.max(self.spectrum_data)
        if current_spectrum_max > self.spectrum_max:
            self.spectrum_max = current_spectrum_max
        else:
            self.spectrum_max = 0.99 * self.spectrum_max + 0.01 * current_spectrum_max
        
        # 波形グラフを更新
        self.wave_line.set_ydata(self.plot_data)
        
        # スペクトラムグラフを更新（表示範囲分のデータのみ）
        display_len = min(len(self.spectrum_data), len(self.spectrum_x_display))
        self.spectrum_line.set_ydata(self.spectrum_data[:display_len])
        
        # Y軸の範囲を更新
        wave_max = max(0.01, self.wave_max * 1.2)  # 最小値を設定して0に近づきすぎないようにする
        self.wave_ax.set_ylim(-wave_max, wave_max)
        
        spectrum_min = self.spectrum_max - 80  # 80dBのダイナミックレンジ
        self.spectrum_ax.set_ylim(spectrum_min, self.spectrum_max + 5)
        
        return [self.wave_line, self.spectrum_line]
    
    def start_visualizer(self):
        """
        ビジュアライザーを開始
        """
        if not self.running:
            print("先にオーディオキャプチャを開始してください")
            return
        
        # 既存のビジュアライザーを閉じる
        if self.fig is not None:
            plt.close(self.fig)
        
        # 新しいフィギュアを作成
        self.fig, (self.wave_ax, self.spectrum_ax) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('オーディオビジュアライザー')
        
        # 波形グラフの初期化
        x = np.arange(self.window_size)
        self.wave_line, = self.wave_ax.plot(x, np.zeros(self.window_size), '-', lw=1, color='cyan')
        self.wave_ax.set_title('波形')
        self.wave_ax.set_ylim(-0.1, 0.1)
        self.wave_ax.set_xlim(0, self.window_size)
        self.wave_ax.set_xlabel('サンプル')
        self.wave_ax.set_ylabel('振幅')
        self.wave_ax.grid(True)
        
        # スペクトラムグラフの初期化（表示範囲を制限）
        max_freq_idx = min(self.window_size // 4, len(self.spectrum_x))
        self.spectrum_x_display = self.spectrum_x[:max_freq_idx]
        self.spectrum_line, = self.spectrum_ax.plot(
            self.spectrum_x_display, 
            np.zeros_like(self.spectrum_x_display), 
            '-', lw=1, color='magenta'
        )
        self.spectrum_ax.set_title('スペクトラム')
        self.spectrum_ax.set_ylim(-80, 0)
        self.spectrum_ax.set_xlim(0, self.sample_rate // 8)  # 表示範囲を制限（高周波は見づらいため）
        self.spectrum_ax.set_xlabel('周波数 (Hz)')
        self.spectrum_ax.set_ylabel('振幅 (dB)')
        self.spectrum_ax.grid(True)
        
        # グラフの見た目を調整
        self.fig.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # ダークテーマを適用
        self.fig.patch.set_facecolor('#121212')
        self.wave_ax.set_facecolor('#1e1e1e')
        self.spectrum_ax.set_facecolor('#1e1e1e')
        
        for ax in [self.wave_ax, self.spectrum_ax]:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#333333')
            ax.grid(True, linestyle='--', alpha=0.3)
        
        # アニメーションを開始
        self.animation = FuncAnimation(
            self.fig, self.update_visualizer, 
            interval=30, blit=True
        )
        
        # 表示
        plt.show(block=False)
        print("ビジュアライザーを開始しました（ウィンドウを閉じると終了します）")
        
        # ウィンドウが閉じられたときの処理
        self.fig.canvas.mpl_connect('close_event', lambda event: self.on_close())
    
    def on_close(self):
        """
        ビジュアライザーウィンドウが閉じられたときの処理
        """
        self.stop_capture()
        print("ビジュアライザーを終了しました")

def main():
    """
    メイン関数
    """
    # オーディオビジュアライザーのインスタンスを作成
    visualizer = AudioVisualizer()
    
    # 利用可能なデバイスを表示
    visualizer.list_audio_devices()
    
    # 自動検出でキャプチャを開始
    visualizer.start_capture()
    
    # 自動検出に失敗した場合、ユーザーにデバイス選択を促す
    if not visualizer.running:
        try:
            device_id = int(input("使用するデバイスIDを入力してください: "))
            visualizer.start_capture(device_id)
        except ValueError:
            print("有効なデバイスIDを入力してください")
            return
    
    # ビジュアライザーを開始
    if visualizer.running:
        visualizer.start_visualizer()
        
        # メインスレッドを維持（ビジュアライザーウィンドウが閉じられるまで）
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nプログラムを終了します...")
        finally:
            # キャプチャを停止
            visualizer.stop_capture()
    else:
        print("オーディオキャプチャを開始できなかったため、ビジュアライザーを起動できません")

if __name__ == "__main__":
    main()