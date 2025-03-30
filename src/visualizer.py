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

        self.spectrum_alpha = 1.0 # 初期透明度
        self.alpha_decay = 0.05   # 透明度減少係数

    def setup_plot(self):
        """
        プロットの初期設定
        """
        self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 背景色を黒色に
        self.fig.patch.set_facecolor('black') # 図全体の背景
        ax1.set_facecolor('black') # 波形プロットの背景
        ax2.set_facecolor('black') # スペクトラムプロットの背景

        # 波形プロット
        ax1.set_title('波形', color='white')
        ax1.set_ylim(-1, 1)
        ax1.set_xlim(0, self.window_size)
        self.wave_line, = ax1.plot(np.arange(self.window_size), np.zeros(self.window_size))
        self.wave_line.set_color('skyblue')

        # スペクトラムプロット
        ax2.set_title('スペクトラム', color='white')
        ax2.set_ylim(-100, 50)
        ax2.set_xlim(20, 10000)
        ax2.set_xscale('log')
        self.spectrum_line, = ax2.plot(np.linspace(0, self.sample_rate//2, self.window_size//2 + 1),
                                     np.zeros(self.window_size//2 + 1))
        self.spectrum_line.set_color('limegreen')

        # 軸・ラベルなどの色
        for ax in (ax1, ax2):
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.grid(color='gray')

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
                self.wave_max *= 0.995
                self.wave_max = max(self.wave_max, np.max(np.abs(wave_data)))
                normalized_wave = wave_data / self.wave_max
                self.plot_data = normalized_wave
                try:
                    self.wave_line.set_ydata(self.plot_data)
                except RuntimeError:
                    # ウィンドウが閉じられて描画対象が無くなった場合のエラーを無視
                    pass

                # フェード処理(スペクトラム)
                amplitude = np.max(np.abs(wave_data))
                if amplitude < 0.003:
                    self.spectrum_alpha = max(0.0, self.spectrum_alpha - self.alpha_decay)
                else:
                    self.spectrum_alpha = min(1.0, self.spectrum_alpha + self.alpha_decay * 2)
                    self.spectrum_data = (1 - self.smoothing_factor) * self.spectrum_data + \
                         self.smoothing_factor * spectrum_data

                try:
                    # プロットを更新
                    self.spectrum_line.set_ydata(self.spectrum_data)
                    self.spectrum_line.set_alpha(self.spectrum_alpha)
                except RuntimeError:
                    # ウィンドウが閉じられて描画対象が無くなった場合のエラーを無視
                    pass

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