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
        
        # 同心円レイヤー
        self.inner_circles = []
        self.num_inner_circles = 3
        
        # パーティクルエフェクト
        self.particles = []
        self.num_particles = 50
        self.particle_life = 100  # フレーム数
        
        # パルスエフェクト
        self.pulse_radius = 0.0
        self.pulse_alpha = 0.0
        self.pulse_circle = None
        self.pulse_active = False
        self.pulse_threshold = 0.5  # パルスを発生させる音量閾値
        
        # 色の変化
        self.hue = 0.0  # 色相（0.0-1.0）
        self.hue_shift_speed = 0.002  # 色相変化速度
        self.wave_base_color = (0, 255, 255)  # シアン
        self.spectrum_base_color = (138, 43, 226)  # パープル
        
        # 時間経過とアニメーション
        self.frame_count = 0
        self.time_factor = 0.0
        
        # 有機的な動きのためのノイズ
        self.phase_shift = np.zeros(self.num_points)
        self.phase_shift_speed = np.linspace(0.01, 0.05, self.num_points)

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
        self.win.setWindowTitle('神秘的オーディオビジュアライザー')
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
        wave_pen = pg.mkPen(color=QColor(*self.wave_base_color), width=2)
        self.wave_curve = pg.PlotCurveItem(pen=wave_pen)
        self.plot_widget.addItem(self.wave_curve)

        # 波形グローエフェクトの初期化
        wave_glow_pen = pg.mkPen(color=QColor(*self.wave_base_color, 30), width=8)
        self.wave_glow_curve = pg.PlotCurveItem(pen=wave_glow_pen)
        self.plot_widget.addItem(self.wave_glow_curve)

        # スペクトラムカーブの初期化
        spectrum_pen = pg.mkPen(color=QColor(*self.spectrum_base_color), width=2)
        self.spectrum_curve = pg.PlotCurveItem(pen=spectrum_pen)
        self.plot_widget.addItem(self.spectrum_curve)

        # スペクトラムグローエフェクトの初期化
        spectrum_glow_pen = pg.mkPen(color=QColor(*self.spectrum_base_color, 30), width=8)
        self.spectrum_glow_curve = pg.PlotCurveItem(pen=spectrum_glow_pen)
        self.plot_widget.addItem(self.spectrum_glow_curve)
        
        # 同心円レイヤーの初期化
        for i in range(self.num_inner_circles):
            radius = 0.1 + i * 0.1
            alpha = 40 - i * 10
            circle = pg.PlotCurveItem(pen=pg.mkPen(color=QColor(100, 200, 255, alpha), width=1))
            self.inner_circles.append(circle)
            self.plot_widget.addItem(circle)
            
            # 円の座標を計算
            theta = np.linspace(0, 2*np.pi, 100)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            circle.setData(x, y)
        
        # パルスエフェクトの初期化
        self.pulse_circle = pg.PlotCurveItem(pen=pg.mkPen(color=QColor(255, 255, 255, 0), width=2))
        self.plot_widget.addItem(self.pulse_circle)
        
        # パーティクルの初期化
        for _ in range(self.num_particles):
            particle = self._create_particle(active=False)
            self.particles.append(particle)
            self.plot_widget.addItem(particle)

        # レイアウトにプロットを追加
        layout.addWidget(self.plot_widget)

        # ショートカットキーの設定
        self._setup_shortcuts()

        # ウィンドウを表示
        self.win.show()

        return self.win
    
    def _create_particle(self, active=True):
        """
        パーティクルを作成
        
        Parameters
        ----------
        active : bool, optional
            パーティクルをアクティブにするかどうか
            
        Returns
        -------
        pg.PlotCurveItem
            パーティクルのプロットアイテム
        """
        if active:
            # ランダムな角度と半径
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0.2, 0.4)
            
            # 速度と寿命
            speed = random.uniform(0.005, 0.015)
            life = random.randint(30, self.particle_life)
            
            # 色
            hue = random.uniform(0, 1)
            r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.8, 1.0)]
            
            # パーティクルの属性を設定
            particle = {
                'angle': angle,
                'radius': radius,
                'speed': speed,
                'life': life,
                'max_life': life,
                'color': (r, g, b)
            }
        else:
            particle = {
                'angle': 0,
                'radius': 0,
                'speed': 0,
                'life': 0,
                'max_life': 0,
                'color': (0, 0, 0)
            }
        
        # プロットアイテムを作成
        plot_item = pg.PlotCurveItem(pen=None)
        plot_item.setData([0], [0])
        plot_item.particle_data = particle
        
        return plot_item

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
    
    def _update_particles(self, amplitude):
        """
        パーティクルの更新
        
        Parameters
        ----------
        amplitude : float
            音の振幅
        """
        # 音量が大きい場合、新しいパーティクルを生成
        if amplitude > 0.1:
            spawn_count = int(amplitude * 5)  # 音量に応じてパーティクル数を調整
            for _ in range(min(spawn_count, 5)):  # 一度に生成するパーティクル数を制限
                for particle_item in self.particles:
                    if particle_item.particle_data['life'] <= 0:
                        # 非アクティブなパーティクルを再利用
                        particle_item.particle_data = self._create_particle(active=True).particle_data
                        break
        
        # 各パーティクルを更新
        for particle_item in self.particles:
            particle = particle_item.particle_data
            
            if particle['life'] > 0:
                # パーティクルの寿命を減らす
                particle['life'] -= 1
                
                # 半径を増加（外側に移動）
                particle['radius'] += particle['speed']
                
                # 透明度を計算（寿命に応じて徐々に透明に）
                alpha = int(255 * (particle['life'] / particle['max_life']))
                
                # 座標を計算
                x = particle['radius'] * math.cos(particle['angle'])
                y = particle['radius'] * math.sin(particle['angle'])
                
                # サイズを計算（寿命に応じて小さく）
                size = 0.02 * (particle['life'] / particle['max_life'])
                
                # パーティクルの描画（小さな円として）
                theta = np.linspace(0, 2*np.pi, 20)
                px = x + size * np.cos(theta)
                py = y + size * np.sin(theta)
                
                # 色を設定
                r, g, b = particle['color']
                particle_item.setPen(pg.mkPen(color=QColor(r, g, b, alpha), width=2))
                
                # データを更新
                particle_item.setData(px, py)
            else:
                # 非アクティブなパーティクルは見えないように
                particle_item.setData([0], [0])
                particle_item.setPen(None)
    
    def _update_pulse(self, amplitude):
        """
        パルスエフェクトの更新
        
        Parameters
        ----------
        amplitude : float
            音の振幅
        """
        # 音量が閾値を超えた場合、新しいパルスを開始
        if amplitude > self.pulse_threshold and not self.pulse_active:
            self.pulse_active = True
            self.pulse_radius = 0.1
            self.pulse_alpha = 150
        
        # パルスが有効な場合、更新
        if self.pulse_active:
            # パルスを拡大
            self.pulse_radius += 0.02
            
            # 透明度を徐々に下げる
            self.pulse_alpha = max(0, self.pulse_alpha - 3)
            
            # パルスの座標を計算
            theta = np.linspace(0, 2*np.pi, 100)
            x = self.pulse_radius * np.cos(theta)
            y = self.pulse_radius * np.sin(theta)
            
            # パルスの描画
            self.pulse_circle.setData(x, y)
            self.pulse_circle.setPen(pg.mkPen(color=QColor(255, 255, 255, self.pulse_alpha), width=2))
            
            # パルスが十分に大きくなったら終了
            if self.pulse_radius > 1.0 or self.pulse_alpha <= 0:
                self.pulse_active = False
                self.pulse_circle.setData([0], [0])
    
    def _update_inner_circles(self):
        """
        同心円レイヤーの更新
        """
        for i, circle in enumerate(self.inner_circles):
            # 基本半径
            radius = 0.1 + i * 0.1
            
            # 時間経過による変動を追加
            variation = 0.02 * math.sin(self.frame_count * 0.05 + i * 0.5)
            current_radius = radius + variation
            
            # 円の座標を計算（わずかに歪ませる）
            theta = np.linspace(0, 2*np.pi, 100)
            distortion = 0.03 * np.sin(theta * (i+2) + self.frame_count * 0.02)
            x = (current_radius + distortion) * np.cos(theta)
            y = (current_radius + distortion) * np.sin(theta)
            
            # 色を時間とともに変化
            hue = (self.hue + i * 0.1) % 1.0
            r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.8)]
            
            # 透明度も時間とともに変化
            alpha = int(30 + 20 * math.sin(self.frame_count * 0.03 + i * 0.7))
            
            # 円を更新
            circle.setData(x, y)
            circle.setPen(pg.mkPen(color=QColor(r, g, b, alpha), width=1))

    def update_plot(self, audio_processor):
        """
        プロットを更新

        Parameters
        ----------
        audio_processor : AudioProcessor
            オーディオプロセッサーインスタンス
        """
        # フレームカウントを更新
        self.frame_count += 1
        self.time_factor = self.frame_count * 0.01
        
        # 色相を徐々に変化
        self.hue = (self.hue + self.hue_shift_speed) % 1.0
        
        # 同心円レイヤーを更新
        self._update_inner_circles()
        
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
                
                # 現在の音量を計算
                current_amplitude = np.max(np.abs(normalized_wave))
                
                # パーティクルを更新
                self._update_particles(current_amplitude)
                
                # パルスエフェクトを更新
                self._update_pulse(current_amplitude)

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
                    
                    # 位相シフトを更新（有機的な動き）
                    self.phase_shift += self.phase_shift_speed
                    
                    # 有機的な動きを追加
                    organic_factor = 0.05 * np.sin(self.theta + self.phase_shift + self.time_factor)
                    
                    # 波形の半径を計算（基本半径 + 波形値 * スケール + 有機的な動き）
                    wave_radius = self.base_radius + resampled_wave * self.wave_scale + organic_factor
                    
                    # 極座標からデカルト座標に変換
                    wave_x, wave_y = self._polar_to_cartesian(wave_radius, self.theta)
                    
                    # 波形の色を計算（音量に応じて色を変化）
                    wave_hue = (self.hue + current_amplitude * 0.2) % 1.0
                    wave_r, wave_g, wave_b = [int(c * 255) for c in colorsys.hsv_to_rgb(wave_hue, 0.9, 1.0)]
                    
                    # 波形プロットの更新
                    self.wave_curve.setData(wave_x, wave_y)
                    self.wave_curve.setPen(pg.mkPen(color=QColor(wave_r, wave_g, wave_b), width=2))
                    
                    # 波形グローエフェクトの更新
                    glow_width = int(8 + current_amplitude * 10)  # 音量に応じてグロー幅を変化
                    self.wave_glow_curve.setData(wave_x, wave_y)
                    self.wave_glow_curve.setPen(pg.mkPen(color=QColor(wave_r, wave_g, wave_b, 30), width=glow_width))

                # スペクトラムの極座標変換
                if self.display_mode == 2 or self.display_mode == 3:
                    # スペクトラムデータを円周上に配置
                    spectrum_values = normalized_spectrum[self.spectrum_indices]
                    
                    # スペクトラムの半径を計算（基本半径 + スペクトラム値 * スケール）
                    # 負の値を持つスペクトラムデータを正規化
                    spectrum_values = (spectrum_values + 100) / 100  # -100dB〜0dBを0〜1に
                    
                    # 有機的な動きを追加
                    spectrum_organic = 0.03 * np.sin(self.theta * 3 + self.time_factor * 0.5)
                    
                    # 最終的な半径を計算
                    spectrum_radius = self.base_radius + spectrum_values * self.spectrum_scale + spectrum_organic
                    
                    # 極座標からデカルト座標に変換
                    spectrum_x, spectrum_y = self._polar_to_cartesian(spectrum_radius, self.theta)
                    
                    # スペクトラムの色を計算（周波数に応じてグラデーション）
                    spectrum_hue = (self.hue + 0.5) % 1.0  # 波形と補色関係に
                    spectrum_r, spectrum_g, spectrum_b = [int(c * 255) for c in colorsys.hsv_to_rgb(spectrum_hue, 0.8, 1.0)]
                    
                    # スペクトラムプロットの更新
                    self.spectrum_curve.setData(spectrum_x, spectrum_y)
                    self.spectrum_curve.setPen(pg.mkPen(color=QColor(spectrum_r, spectrum_g, spectrum_b), width=2))
                    
                    # スペクトラムグローエフェクトの更新
                    spectrum_glow_width = int(8 + np.mean(spectrum_values) * 10)
                    self.spectrum_glow_curve.setData(spectrum_x, spectrum_y)
                    self.spectrum_glow_curve.setPen(pg.mkPen(color=QColor(spectrum_r, spectrum_g, spectrum_b, 30), width=spectrum_glow_width))

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