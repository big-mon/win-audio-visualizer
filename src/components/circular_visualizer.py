#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
オーディオデータの可視化を担当するモジュール
PyQtGraphを使用した高速なリアルタイム描画を実現
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QPen, QKeySequence, QLinearGradient, QRadialGradient, QBrush
from PyQt5.QtWidgets import QShortcut
import math
import random
import colorsys

"""
オーディオデータの可視化を行うクラス（PyQtGraph版）
"""

class CircularVisualizer:
    """
    オーディオデータの円形可視化を行うクラス（PyQtGraph版）
    極座標を使用して波形とスペクトラムを円形に表示
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
        self.wave_glow_data = np.zeros(window_size)
        self.spectrum_glow_data = np.zeros(window_size // 2 + 1)

        # グロー効果の設定
        self.glow_layers = 4
        self.glow_curves = []

        # PyQtGraphアプリケーションとウィンドウ
        self.app = None
        self.win = None
        self.timer = None

        # ビジュアライゼーションパラメータ
        self.spectrum_max = 0.1
        self.wave_max = 0.1
        self.smoothing_factor = 0.2
        self.spectrum_alpha = 1.0
        self.alpha_decay = 0.05

        # プロットウィジェットとデータアイテム
        self.wave_plot = None
        self.wave_curve = None
        self.wave_glow_curve = None

        # 円周上の点の数と基本半径
        self.num_points = 360
        self.base_radius = 0.4
        self.wave_scale = 0.2
        self.spectrum_scale = 0.2
        self.theta = np.linspace(0, 2*np.pi, self.num_points)
        self.spectrum_indices = np.linspace(0, self.window_size//2, self.num_points, dtype=int)
        self.hue = 0.0
        self.hue_shift_speed = 0.001

        # 中心光のコア
        self.core_glow_layers = 8
        self.core_glow_curves = []
        self.core_base_radius = 0.04
        self.core_pulse_strength = 0.01
        self.core_pulse_speed = 0.05
        self.core_base_alpha = 35

    def setup_plot(self):
        """
        プロットウィンドウを設定
        """
        if QtWidgets.QApplication.instance() is None:
            self.app = QtWidgets.QApplication([])
        else:
            self.app = QtWidgets.QApplication.instance()

        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("神秘的ビジュアライザー")
        self.win.resize(1000, 1000)

        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setBackground((5, 5, 10))
        self.wave_plot.setAspectLocked(True)
        self.wave_plot.hideAxis('left')
        self.wave_plot.hideAxis('bottom')
        self.wave_plot.setXRange(-1, 1)
        self.wave_plot.setYRange(-1, 1)

        # カーブの初期化
        self.wave_curve = pg.PlotCurveItem()
        self.wave_glow_curve = pg.PlotCurveItem()

        # グロー層の追加（最初に描画されるように）
        for i in range(self.glow_layers):
            glow_curve = pg.PlotCurveItem()
            self.wave_plot.addItem(glow_curve)
            self.glow_curves.append(glow_curve)

        # 波形カーブ（主）とそのグロー
        self.wave_plot.addItem(self.wave_glow_curve)
        self.wave_plot.addItem(self.wave_curve)

        # 光のコア（中心の揺れる円）
        for _ in range(self.core_glow_layers):
            curve = pg.PlotCurveItem()
            self.core_glow_curves.append(curve)
            self.wave_plot.addItem(curve)

        layout.addWidget(self.wave_plot)
        self.win.show()

        return self.win

    def _polar_to_cartesian(self, radius, theta):
        """
        ポーラ座標を直交座標に変換

        Parameters
        ----------
        radius : float
            半径
        theta : float
            角度

        Returns
        -------
        x : float
            x座標
        y : float
            y座標
        """
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y

    def update_plot(self, audio_processor):
        """
        プロットを更新

        Parameters
        ----------
        audio_processor : AudioProcessor
            音声処理クラス
        """
        # 色相を更新
        self.hue = (self.hue + self.hue_shift_speed) % 1.0

        # 音声データを取得
        data = audio_processor.get_audio_data()
        if data is not None:
            wave_data, spectrum_data = audio_processor.process_audio_data(data)

            # 波形とスペクトラムを正規化
            if wave_data is not None and spectrum_data is not None:
                # 波形の正規化
                self.wave_max *= 0.995
                self.wave_max = max(self.wave_max, np.max(np.abs(wave_data)))
                normalized_wave = wave_data / self.wave_max

                # スペクトラムの正規化
                self.spectrum_max *= 0.995
                self.spectrum_max = max(self.spectrum_max, np.max(np.abs(spectrum_data)))
                normalized_spectrum = spectrum_data / 100.0

                # 波形をサンプリング
                resampled_wave = np.interp(
                    np.linspace(0, len(wave_data) - 1, self.num_points),
                    np.arange(len(wave_data)),
                    normalized_wave
                )
                spectrum_values = normalized_spectrum[self.spectrum_indices]
                spectrum_values = (spectrum_values + 100) / 100

                # 波形とスペクトラムを組み合わせる
                combined = 0.6 * resampled_wave + 0.4 * spectrum_values
                organic_factor = 0.05 * np.sin(self.theta * 2 + self.hue * 10)
                radius = self.base_radius + combined * self.wave_scale + organic_factor
                x, y = self._polar_to_cartesian(radius, self.theta)

                # 波形を描画
                r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(self.hue, 0.8, 1.0)]
                self.wave_curve.setData(x, y)
                self.wave_curve.setPen(pg.mkPen(color=QColor(r, g, b), width=2))

                # 単層のグロー（ベースとして残しても良い）
                self.wave_glow_curve.setData(x, y)
                self.wave_glow_curve.setPen(pg.mkPen(color=QColor(r, g, b, 30), width=10))

                # 🌟 多重グローエフェクトの追加（最終的な神秘感の主役）
                for i, glow_curve in enumerate(self.glow_curves):
                    alpha = int(20 * (1.0 - i / self.glow_layers)) + 10
                    width = 10 + i * 3
                    glow_curve.setData(x, y)
                    glow_curve.setPen(pg.mkPen(color=QColor(r, g, b, alpha), width=width))

                # 光のコア（儚く脈打つ多重円）
                pulse = self.core_pulse_strength * np.sin(self.hue * 2 * np.pi)  # ゆっくり鼓動

                for i, curve in enumerate(self.core_glow_curves):
                    ratio = (i + 1) / self.core_glow_layers
                    radius = self.core_base_radius * (1 + ratio * 1.5) + pulse * (1 - ratio)
                    alpha = int(self.core_base_alpha * (1 - ratio)**1.5)  # より外を儚く
                    width = 1 + int(3 * (1 - ratio))

                    theta = np.linspace(0, 2 * np.pi, 100)
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)

                    # 白〜淡い青系で幻想的に
                    r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb((self.hue + 0.55) % 1.0, 0.3, 1.0)]
                    curve.setData(x, y)
                    curve.setPen(pg.mkPen(QColor(r, g, b, alpha), width=width))

    def start_animation(self, audio_processor, interval=16):
        """
        アニメーションを開始

        Parameters
        ----------
        audio_processor : AudioProcessor
            音声処理クラス
        interval : int
            タイマー間隔
        """
        # ウィンドウを初期化
        if self.win is None:
            self.setup_plot()

        # タイマーを設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self.update_plot(audio_processor))
        self.timer.start(interval)

        # アプリケーションを開始
        if self.app is not None:
            self.app.exec_()

    def stop_animation(self):
        """
        アニメーションを停止
        """
        # タイマーを停止
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

        # ウィンドウを閉じる
        if self.win is not None:
            self.win.close()
            self.win = None