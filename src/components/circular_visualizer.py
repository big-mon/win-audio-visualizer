#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
オーディオデータの可視化を担当するモジュール
PyQtGraphを使用した高速なリアルタイム描画を実現
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QPen, QKeySequence
from PyQt5.QtWidgets import QShortcut
import math

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

        # アプリケーションとウィンドウの設定
        self.app = None
        self.win = None
        self.timer = None

        # ビジュアライザーの設定
        self.spectrum_max = 0.1  # スペクトラムの最大値（自動調整用）
        self.wave_max = 0.1      # 波形の最大値（自動調整用）
        self.smoothing_factor = 0.2  # データの平滑化係数

        self.spectrum_alpha = 1.0  # 初期透明度
        self.alpha_decay = 0.05    # 透明度減少係数

        # 円形プロット用のパラメータ
        self.num_points = 360  # 円周上の点の数
        self.base_radius = 0.3  # 基本半径
        self.wave_scale = 0.2   # 波形の振幅スケール
        self.spectrum_scale = 0.3  # スペクトラムの振幅スケール

        # 表示モード（1: 波形のみ、2: スペクトラムのみ、3: 両方）
        self.display_mode = 3

        # プロットウィジェットとデータアイテム
        self.plot_widget = None
        self.wave_curve = None
        self.wave_glow_curve = None
        self.spectrum_curve = None
        self.spectrum_glow_curve = None

        # 極座標変換用の角度配列
        self.theta = np.linspace(0, 2*np.pi, self.num_points)

        # スペクトラムのインデックスマッピング
        self.spectrum_indices = np.linspace(0, min(self.window_size//2, 1000), self.num_points, dtype=int)

    def setup_plot(self):
        """
        プロットの初期設定
        """
        # アプリケーションが存在しない場合は作成
        if QtWidgets.QApplication.instance() is None:
            self.app = QtWidgets.QApplication([])
        else:
            self.app = QtWidgets.QApplication.instance()

        # ウィンドウの設定
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('円形オーディオビジュアライザー')
        self.win.resize(1000, 1000)

        # 中央ウィジェットとレイアウトの設定
        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # 円形プロットの設定
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(QColor(5, 5, 10))  # 黒系の背景色
        self.plot_widget.showGrid(False, False)
        self.plot_widget.setAspectLocked(True)  # アスペクト比を固定
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.setXRange(-1, 1)
        self.plot_widget.setYRange(-1, 1)

        # 波形カーブの初期化
        wave_pen = pg.mkPen(color=QColor(0, 255, 255), width=2)
        self.wave_curve = pg.PlotCurveItem(pen=wave_pen)
        self.plot_widget.addItem(self.wave_curve)

        # 波形グローエフェクトの初期化
        wave_glow_pen = pg.mkPen(color=QColor(0, 255, 255, 30), width=8)
        self.wave_glow_curve = pg.PlotCurveItem(pen=wave_glow_pen)
        self.plot_widget.addItem(self.wave_glow_curve)

        # スペクトラムカーブの初期化
        spectrum_pen = pg.mkPen(color=QColor(138, 43, 226), width=2)
        self.spectrum_curve = pg.PlotCurveItem(pen=spectrum_pen)
        self.plot_widget.addItem(self.spectrum_curve)

        # スペクトラムグローエフェクトの初期化
        spectrum_glow_pen = pg.mkPen(color=QColor(138, 43, 226, 30), width=8)
        self.spectrum_glow_curve = pg.PlotCurveItem(pen=spectrum_glow_pen)
        self.plot_widget.addItem(self.spectrum_glow_curve)

        # レイアウトにプロットを追加
        layout.addWidget(self.plot_widget)

        # ショートカットキーの設定
        self._setup_shortcuts()

        # ウィンドウを表示
        self.win.show()

        return self.win

    def _setup_shortcuts(self):
        """
        ショートカットキーの設定
        """
        # 1キー: 波形のみ表示
        shortcut1 = QShortcut(QKeySequence("1"), self.win)
        shortcut1.activated.connect(lambda: self._set_display_mode(1))

        # 2キー: スペクトラムのみ表示
        shortcut2 = QShortcut(QKeySequence("2"), self.win)
        shortcut2.activated.connect(lambda: self._set_display_mode(2))

        # 3キー: 両方表示
        shortcut3 = QShortcut(QKeySequence("3"), self.win)
        shortcut3.activated.connect(lambda: self._set_display_mode(3))

    def _set_display_mode(self, mode):
        """
        表示モードを設定

        Parameters
        ----------
        mode : int
            表示モード (1: 波形のみ、2: スペクトラムのみ、3: 両方)
        """
        self.display_mode = mode
        # 波形の表示/非表示
        self.wave_curve.setVisible(mode == 1 or mode == 3)
        self.wave_glow_curve.setVisible(mode == 1 or mode == 3)
        # スペクトラムの表示/非表示
        self.spectrum_curve.setVisible(mode == 2 or mode == 3)
        self.spectrum_glow_curve.setVisible(mode == 2 or mode == 3)

    def _polar_to_cartesian(self, radius, theta):
        """
        極座標からデカルト座標への変換

        Parameters
        ----------
        radius : ndarray
            半径の配列
        theta : ndarray
            角度の配列

        Returns
        -------
        tuple
            (x座標の配列, y座標の配列)
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
            オーディオプロセッサーインスタンス
        """
        # オーディオデータを取得
        data = audio_processor.get_audio_data()
        if data is not None:
            # データを処理
            wave_data, spectrum_data = audio_processor.process_audio_data(data)

            if wave_data is not None and spectrum_data is not None:
                # 波形データの更新
                self.wave_max *= 0.995
                self.wave_max = max(self.wave_max, np.max(np.abs(wave_data)))
                normalized_wave = wave_data / self.wave_max
                self.plot_data = normalized_wave

                # スペクトラムデータの更新
                self.spectrum_max *= 0.995
                self.spectrum_max = max(self.spectrum_max, np.max(np.abs(spectrum_data)))
                normalized_spectrum = spectrum_data / 100.0  # dBスケール調整

                # 波形の極座標変換
                if self.display_mode == 1 or self.display_mode == 3:
                    # 波形データを円周上に配置するためのリサンプリング
                    resampled_wave = np.interp(
                        np.linspace(0, len(wave_data) - 1, self.num_points),
                        np.arange(len(wave_data)),
                        wave_data
                    )

                    # 波形の半径を計算（基本半径 + 波形値 * スケール）
                    wave_radius = self.base_radius + resampled_wave * self.wave_scale

                    # 極座標からデカルト座標に変換
                    wave_x, wave_y = self._polar_to_cartesian(wave_radius, self.theta)

                    # 波形プロットの更新
                    self.wave_curve.setData(wave_x, wave_y)
                    self.wave_glow_curve.setData(wave_x, wave_y)

                # スペクトラムの極座標変換
                if self.display_mode == 2 or self.display_mode == 3:
                    # スペクトラムデータを円周上に配置
                    spectrum_values = normalized_spectrum[self.spectrum_indices]

                    # スペクトラムの半径を計算（基本半径 + スペクトラム値 * スケール）
                    # 負の値を持つスペクトラムデータを正規化
                    spectrum_values = (spectrum_values + 100) / 100  # -100dB〜0dBを0〜1に
                    spectrum_radius = self.base_radius + spectrum_values * self.spectrum_scale

                    # 極座標からデカルト座標に変換
                    spectrum_x, spectrum_y = self._polar_to_cartesian(spectrum_radius, self.theta)

                    # スペクトラムプロットの更新
                    self.spectrum_curve.setData(spectrum_x, spectrum_y)
                    self.spectrum_glow_curve.setData(spectrum_x, spectrum_y)

    def start_animation(self, audio_processor, interval=16):
        """
        アニメーションを開始

        Parameters
        ----------
        audio_processor : AudioProcessor
            オーディオプロセッサーインスタンス
        interval : int, optional
            アニメーションの更新間隔（ミリ秒）、約60FPSを目標
        """
        if self.win is None:
            self.setup_plot()

        # タイマーを設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self.update_plot(audio_processor))
        self.timer.start(interval)

        # イベントループを開始
        if self.app is not None:
            self.app.exec_()

    def stop_animation(self):
        """
        アニメーションを停止
        """
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

        if self.win is not None:
            self.win.close()
            self.win = None