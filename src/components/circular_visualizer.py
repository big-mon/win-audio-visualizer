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
        # オーディオ処理用のバッファ
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.plot_data = np.zeros(window_size)
        self.spectrum_data = np.zeros(window_size // 2 + 1)
        self.wave_glow_data = np.zeros(window_size)
        self.spectrum_glow_data = np.zeros(window_size // 2 + 1)

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

        self.num_points = 360
        self.base_radius = 0.4
        self.wave_scale = 0.2
        self.spectrum_scale = 0.2
        self.theta = np.linspace(0, 2*np.pi, self.num_points)
        self.spectrum_indices = np.linspace(0, self.window_size//2, self.num_points, dtype=int)
        self.hue = 0.0
        self.hue_shift_speed = 0.001

        # 中心光のコア
        self.core_glow_items = []
        self.core_circle = None
        self.core_radius = 0.15
        self.core_alpha = 100
        self.core_pulse_phase = 0.0
        self.core_glow_layers = 3

    def setup_plot(self):
        """
        プロットウィンドウを設定
        """
        if QtWidgets.QApplication.instance() is None:
            self.app = QtWidgets.QApplication([])
        else:
            self.app = QtWidgets.QApplication.instance()

        # PyQtGraphウィンドウ
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("神秘的ビジュアライザー")
        self.win.resize(1000, 1000)

        # 中央ウィンドウ
        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # 波形プロット
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setBackground((5, 5, 10))
        self.wave_plot.setAspectLocked(True)
        self.wave_plot.hideAxis('left')
        self.wave_plot.hideAxis('bottom')
        self.wave_plot.setXRange(-1, 1)
        self.wave_plot.setYRange(-1, 1)

        # 波形カーブ
        self.wave_curve = pg.PlotCurveItem()
        self.wave_glow_curve = pg.PlotCurveItem()
        self.core_circle = pg.PlotCurveItem()

        # グロー用アイテムを生成しておく
        for i in range(self.core_glow_layers):
            glow_item = pg.PlotCurveItem()
            self.core_glow_items.append(glow_item)
            self.wave_plot.addItem(glow_item)

        # 順序: 背面から前面へ
        self.wave_plot.addItem(self.core_circle)
        self.wave_plot.addItem(self.wave_glow_curve)
        self.wave_plot.addItem(self.wave_curve)

        # プロットウィンドウをレイアウトに追加
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
        complement_hue = (self.hue + 0.5) % 1.0

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

                # 色を計算
                r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(self.hue, 0.8, 1.0)]
                cr, cg, cb = [int(c * 255) for c in colorsys.hsv_to_rgb(complement_hue, 0.7, 0.8)]

                # 波形を更新
                self.wave_curve.setData(x, y)
                self.wave_curve.setPen(pg.mkPen(color=QColor(r, g, b), width=2))
                self.wave_glow_curve.setData(x, y)
                self.wave_glow_curve.setPen(pg.mkPen(color=QColor(cr, cg, cb, 30), width=12))

                # 光のコアを更新
                self.core_pulse_phase += 0.05
                noise = 0.002 * np.random.randn()
                pulse_variation = 0.01 * np.sin(self.core_pulse_phase) + noise
                core_r = self.core_radius + pulse_variation

                theta = np.linspace(0, 2*np.pi, 100)
                cx = core_r * np.cos(theta)
                cy = core_r * np.sin(theta)

                # 彩度を時間で変化
                sat = 0.6 + 0.3 * np.sin(self.core_pulse_phase * 0.8)
                core_r_color, core_g, core_b = [int(c * 255) for c in colorsys.hsv_to_rgb(self.hue, sat, 1.0)]
                core_alpha = int(self.core_alpha + 50 * np.sin(self.core_pulse_phase * 0.5))

                self.core_circle.setData(cx, cy)
                self.core_circle.setPen(pg.mkPen(color=QColor(core_r_color, core_g, core_b, core_alpha), width=20))

                # グロー更新（前フレームから使いまわす）
                for i, glow_item in enumerate(self.core_glow_items):
                    glow_radius = core_r + 0.015 * i
                    glow_alpha = max(10, int((self.core_alpha - i * 30) * (1 + 0.2 * np.sin(self.core_pulse_phase + i))))
                    glow_width = 20 + i * 10

                    gx = glow_radius * np.cos(theta)
                    gy = glow_radius * np.sin(theta)

                    glow_color = QColor(core_r_color, core_g, core_b, glow_alpha)
                    glow_item.setData(gx, gy)
                    glow_item.setPen(pg.mkPen(color=glow_color, width=glow_width))

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