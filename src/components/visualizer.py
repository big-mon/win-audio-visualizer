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

class Visualizer:
    """
    オーディオデータの可視化を行うクラス（PyQtGraph版）
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

        # プロットウィジェットとデータアイテム
        self.wave_plot = None
        self.spectrum_plot = None
        self.wave_curve = None
        self.wave_glow_curve = None
        self.spectrum_curve = None
        self.spectrum_glow_curve = None

        # スペクトラムのX軸データ
        self.spectrum_x = np.linspace(20, self.sample_rate//2, self.window_size//2 + 1)

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
        self.win.setWindowTitle('オーディオビジュアライザー')
        self.win.resize(1200, 800)

        # 中央ウィジェットとレイアウトの設定
        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # 波形プロットの設定
        self.wave_plot = pg.PlotWidget(title='波形')
        self.wave_plot.setBackground('black')
        self.wave_plot.showGrid(x=True, y=True, alpha=0.3)
        self.wave_plot.setYRange(-1, 1)
        self.wave_plot.setXRange(0, self.window_size)

        # 波形カーブの設定
        self.wave_curve = self.wave_plot.plot(
            np.arange(self.window_size),
            np.zeros(self.window_size),
            pen=pg.mkPen(color='skyblue', width=1)
        )

        # 波形グローエフェクトの設定
        glow_pen = pg.mkPen(color=QColor(0, 191, 255, 13), width=5)
        self.wave_glow_curve = self.wave_plot.plot(
            np.arange(self.window_size),
            self.wave_glow_data,
            pen=glow_pen
        )

        # スペクトラムプロットの設定
        self.spectrum_plot = pg.PlotWidget(title='スペクトラム')
        self.spectrum_plot.setBackground('black')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_plot.setYRange(-100, 50)
        self.spectrum_plot.setXRange(20, 10000)
        self.spectrum_plot.setLogMode(x=True, y=False)  # X軸をlogスケールに

        # スペクトラムカーブの設定
        self.spectrum_curve = self.spectrum_plot.plot(
            self.spectrum_x,
            np.zeros(self.window_size//2 + 1),
            pen=pg.mkPen(color='limegreen', width=1)
        )

        # スペクトラムグローエフェクトの設定
        spectrum_glow_pen = pg.mkPen(color=QColor(60, 179, 113, 13), width=5)
        self.spectrum_glow_curve = self.spectrum_plot.plot(
            self.spectrum_x,
            self.spectrum_glow_data,
            pen=spectrum_glow_pen
        )

        # テキストの色を白に設定
        self._set_text_color(self.wave_plot)
        self._set_text_color(self.spectrum_plot)

        # レイアウトにプロットを追加
        layout.addWidget(self.wave_plot)
        layout.addWidget(self.spectrum_plot)

        # ウィンドウを表示
        self.win.show()

        return self.win

    def _set_text_color(self, plot_widget):
        """
        プロットウィジェットのテキスト色を設定

        Parameters
        ----------
        plot_widget : pg.PlotWidget
            テキスト色を設定するプロットウィジェット
        """
        # タイトルの色を設定（直接文字列を指定せず、現在のタイトルを維持）
        title = plot_widget.plotItem.titleLabel.text
        plot_widget.setTitle(title, color='white')

        # 軸ラベルの色を設定
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='white'))
        plot_widget.getAxis('left').setPen(pg.mkPen(color='white'))

        # 軸の数値の色を設定
        plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='white'))
        plot_widget.getAxis('left').setTextPen(pg.mkPen(color='white'))

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
                self.wave_glow_data = normalized_wave  # ← 毎回更新

                # 波形プロットの更新
                self.wave_curve.setData(np.arange(self.window_size), self.plot_data)
                self.wave_glow_curve.setData(np.arange(self.window_size), self.wave_glow_data)

                # フェード処理（スペクトラム）
                amplitude = np.max(np.abs(wave_data))
                if amplitude < 0.003:
                    self.spectrum_alpha = max(0.0, self.spectrum_alpha - self.alpha_decay)
                else:
                    self.spectrum_alpha = min(1.0, self.spectrum_alpha + self.alpha_decay * 2)
                    self.spectrum_data = (1 - self.smoothing_factor) * self.spectrum_data + \
                         self.smoothing_factor * spectrum_data
                self.spectrum_glow_data = self.spectrum_data  # ← 条件外でも更新

                # スペクトラムプロットの更新
                self.spectrum_curve.setData(self.spectrum_x, self.spectrum_data)

                # 透明度の設定
                spectrum_pen = pg.mkPen(color=QColor(50, 205, 50, int(255 * self.spectrum_alpha)), width=1)
                self.spectrum_curve.setPen(spectrum_pen)

                spectrum_glow_pen = pg.mkPen(
                    color=QColor(60, 179, 113, int(255 * self.spectrum_alpha * 0.3)),
                    width=5
                )
                self.spectrum_glow_curve.setData(self.spectrum_x, self.spectrum_glow_data)
                self.spectrum_glow_curve.setPen(spectrum_glow_pen)

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
