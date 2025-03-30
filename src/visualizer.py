#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
オーディオデータの可視化を担当するモジュール
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation

# 日本語フォント設定
plt.rcParams['font.family'] = 'Yu Gothic'  # Windows用日本語フォント
rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止

class Visualizer:
    """
    オーディオデータの可視化を行うクラス
    """

    def __init__(self, window_size=2048, sample_rate=44100):
        """
        初期化メソッド

        Parameters
        ----------
        window_size : int, optional
            FFTウィンドウサイズ
        sample_rate : int, optional
            サンプリングレート
        """
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.plot_data = np.zeros(window_size)
        self.spectrum_data = np.zeros(window_size // 2 + 1)
        self.fig = None
        self.animation = None

        # ビジュアライザーの設定
        self.spectrum_max = 0.1  # スペクトラムの最大値（自動調整用）
        self.wave_max = 0.1      # 波形の最大値（自動調整用）
        self.smoothing_factor = 0.2  # データの平滑化係数

    def setup_plot(self):
        """
        プロットの初期設定
        """
        self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 波形プロット
        ax1.set_title('波形')
        ax1.set_ylim(-1, 1)
        ax1.set_xlim(0, self.window_size)
        self.wave_line, = ax1.plot(np.arange(self.window_size), np.zeros(self.window_size))
        self.wave_line.set_color('skyblue')

        # スペクトラムプロット
        ax2.set_title('スペクトラム')
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, self.sample_rate // 2)
        ax2.set_xscale('log')
        self.spectrum_line, = ax2.plot(np.linspace(0, self.sample_rate//2, self.window_size//2 + 1),
                                     np.zeros(self.window_size//2 + 1))
        self.spectrum_line.set_color('limegreen')

        # グラフの体裁を整える
        plt.tight_layout()
        return self.fig

    def update_plot(self, frame, audio_processor):
        """
        プロットを更新

        Parameters
        ----------
        frame : int
            フレーム番号
        audio_processor : AudioProcessor
            オーディオプロセッサーインスタンス

        Returns
        -------
        tuple
            更新されたプロットのライン
        """
        # オーディオデータを取得
        data = audio_processor.get_audio_data()
        if data is not None:
            # データを処理
            wave_data, spectrum_data = audio_processor.process_audio_data(data)

            if wave_data is not None and spectrum_data is not None:
                # 波形データを更新
                self.wave_max *= 0.995  # 徐々に減衰
                self.wave_max = max(self.wave_max, np.max(np.abs(wave_data)))
                normalized_wave = wave_data / self.wave_max
                self.plot_data = normalized_wave

                # スペクトラムデータを更新
                self.spectrum_max *= 0.995  # 徐々に減衰
                self.spectrum_max = max(self.spectrum_max, np.max(spectrum_data))
                normalized_spectrum = spectrum_data / self.spectrum_max
                self.spectrum_data = (1 - self.smoothing_factor) * self.spectrum_data + \
                                   self.smoothing_factor * normalized_spectrum

                # プロットを更新
                self.wave_line.set_ydata(self.plot_data)
                self.spectrum_line.set_ydata(self.spectrum_data)

        return self.wave_line, self.spectrum_line

    def start_animation(self, audio_processor, interval=10):
        """
        アニメーションを開始

        Parameters
        ----------
        audio_processor : AudioProcessor
            オーディオプロセッサーインスタンス
        interval : int, optional
            アニメーションの更新間隔（ミリ秒）
        """
        if self.fig is None:
            self.setup_plot()

        self.animation = FuncAnimation(
            self.fig,
            self.update_plot,
            fargs=(audio_processor,),
            interval=interval,
            blit=True,
            cache_frame_data=False
        )
        plt.show()

    def stop_animation(self):
        """
        アニメーションを停止
        """
        if self.animation is not None:
            self.animation.event_source.stop()
            plt.close(self.fig)
            self.fig = None
            self.animation = None