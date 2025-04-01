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
from scipy.interpolate import splprep, splev

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
        self.inner_wave_curve = None
        self.outer_wave_curve = None

        self.num_points = 720  # 360から720に増やして滑らかさを向上
        self.base_radius = 0.4
        self.wave_scale = 0.2
        self.spectrum_scale = 0.2
        self.theta = np.linspace(0, 2*np.pi, self.num_points, endpoint=True)  # endpointをTrueに設定して円を閉じる
        self.spectrum_indices = np.linspace(0, self.window_size//2, self.num_points, dtype=int)
        self.hue = 0.0
        self.hue_shift_speed = 0.001
        
        # 波の動きに関するパラメータ
        self.wave_time = 0.0
        self.wave_speed = 0.008  # 呼吸のような自然な速度に調整
        self.harmonic_factors = [0.1, 0.3, 0.5, 0.7, 0.9]  # より低い周波数に調整
        self.harmonic_weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # 低周波成分をさらに強調
        self.phase_offsets = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        # 呼吸のような自然な揺らぎのための追加パラメータ
        self.breath_phase = 0.0
        self.breath_speed = 0.003  # 呼吸の速度（ゆっくり）
        self.breath_depth = 0.06   # 呼吸の深さ（揺らぎの大きさ）
        
        # 波形の履歴（残像効果用）
        self.wave_history = []
        self.history_length = 5
        self.history_decay = 0.7

        # 中心光のコア
        self.core_circle = None
        self.core_radius = 0.15
        self.core_alpha = 100
        self.core_pulse_phase = 0.0
        
        # 内側の光の輪
        self.inner_core = None
        self.inner_core_radius = 0.08
        
        # 光の粒子
        self.particle_item = None
        self.particles = [{'x': random.uniform(-1, 1),
                           'y': random.uniform(-1, 1),
                           'vx': random.uniform(-0.0005, 0.0005),
                           'vy': random.uniform(-0.0005, 0.0005),
                           'size': random.uniform(2, 4),
                           'opacity': random.uniform(20, 100),
                           'life': random.uniform(0.8, 1.0)} for _ in range(150)]
        
        # 光の線（知性を表現）
        self.light_lines = []
        self.line_items = []
        self.max_lines = 8
        self.line_spawn_chance = 0.02
        self.line_life_max = 1.0

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
        self.wave_plot.setBackground((5, 5, 15))  # より暗く、青みがかった背景
        self.wave_plot.setAspectLocked(True)
        self.wave_plot.hideAxis('left')
        self.wave_plot.hideAxis('bottom')
        self.wave_plot.setXRange(-1, 1)
        self.wave_plot.setYRange(-1, 1)

        # 波形レイヤーの初期化（奥から前へ）
        self.inner_core = pg.PlotCurveItem(pen=pg.mkPen(width=15), connect="all")  # connectパラメータを追加
        self.core_circle = pg.PlotCurveItem(pen=pg.mkPen(width=20), connect="all")
        self.inner_wave_curve = pg.PlotCurveItem(pen=pg.mkPen(width=1.5), connect="all")
        self.wave_glow_curve = pg.PlotCurveItem(pen=pg.mkPen(width=15), connect="all")
        self.wave_curve = pg.PlotCurveItem(pen=pg.mkPen(width=2), connect="all")
        self.outer_wave_curve = pg.PlotCurveItem(pen=pg.mkPen(width=1.5), connect="all")
        
        # 粒子の初期化
        self.particle_item = pg.ScatterPlotItem()
        
        # 光の線の初期化
        for _ in range(self.max_lines):
            line_item = pg.PlotCurveItem()
            self.line_items.append(line_item)
            self.wave_plot.addItem(line_item)

        # 順序: 背面から前面へ
        self.wave_plot.addItem(self.inner_core)
        self.wave_plot.addItem(self.core_circle)
        self.wave_plot.addItem(self.inner_wave_curve)
        self.wave_plot.addItem(self.wave_glow_curve)
        self.wave_plot.addItem(self.wave_curve)
        self.wave_plot.addItem(self.outer_wave_curve)
        self.wave_plot.addItem(self.particle_item)

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
        
    def _create_harmonic_wave(self, base_amplitude, phase_offset=0.0):
        """
        複数の高調波を組み合わせた有機的な波を生成
        
        Parameters
        ----------
        base_amplitude : float
            基本振幅
        phase_offset : float
            位相オフセット
            
        Returns
        -------
        wave : ndarray
            生成された波形
        """
        wave = np.zeros(self.num_points)
        
        # 低周波のゆったりとした動きを生成
        for i, (factor, weight) in enumerate(zip(self.harmonic_factors, self.harmonic_weights)):
            phase = self.wave_time * factor + self.phase_offsets[i] + phase_offset
            # 円形に閉じるよう、最後の点が最初の点と一致するようにする
            harmonic = weight * np.sin(self.theta * factor + phase)
            wave += harmonic
        
        # 呼吸のような自然な揺らぎを追加
        breath_effect = self.breath_depth * np.sin(self.breath_phase + self.theta * 0.5)
        wave += breath_effect
        
        # 波形をスムージング（移動平均）して急激な変化を抑制
        window_size = 5
        smoothed_wave = np.zeros_like(wave)
        padded_wave = np.pad(wave, (window_size//2, window_size//2), mode='wrap')
        
        for i in range(len(wave)):
            smoothed_wave[i] = np.mean(padded_wave[i:i+window_size])
            
        return smoothed_wave * base_amplitude
        
    def _spawn_light_line(self, intensity):
        """
        音の強度に応じて光の線を生成
        
        Parameters
        ----------
        intensity : float
            音の強度（0.0〜1.0）
        """
        if len(self.light_lines) < self.max_lines and random.random() < self.line_spawn_chance * intensity:
            # 線の始点（中心付近）
            start_radius = self.core_radius * (0.8 + random.random() * 0.4)
            start_angle = random.random() * 2 * np.pi
            
            # 線の終点（外側）- 始点と同じ方向に伸ばす（角度の差を小さくする）
            end_radius = self.base_radius + random.random() * 0.3
            # 角度の差をさらに小さくして、より円に沿った動きにする
            angle_diff = random.uniform(-0.1, 0.1)  # 角度の差を±0.1ラジアン（約±5.7度）に制限
            end_angle = start_angle + angle_diff
            
            # 線の中間点（曲線を作るため）- 始点と終点の間に配置
            mid_radius = (start_radius + end_radius) * 0.5
            # 中間点も同様に角度の差を小さくする
            mid_angle = (start_angle + end_angle) * 0.5 + random.uniform(-0.05, 0.05)
            
            # 始点と終点の距離をチェック - 距離が大きすぎる場合は生成しない
            sx, sy = self._polar_to_cartesian(start_radius, start_angle)
            ex, ey = self._polar_to_cartesian(end_radius, end_angle)
            distance = np.sqrt((ex-sx)**2 + (ey-sy)**2)
            
            # 距離が適切な場合のみ線を生成
            if distance <= 0.8:  # 最大距離を制限
                # 色相はメインの色相に近いものを選択
                hue_offset = random.uniform(-0.1, 0.1)
                line_hue = (self.hue + hue_offset) % 1.0
                
                self.light_lines.append({
                    'start': (start_radius, start_angle),
                    'mid': (mid_radius, mid_angle),
                    'end': (end_radius, end_angle),
                    'width': random.uniform(1.5, 3.0),
                    'hue': line_hue,
                    'life': self.line_life_max,
                    'max_opacity': random.randint(150, 220)
                })

    def update_plot(self, audio_processor):
        """
        プロットを更新

        Parameters
        ----------
        audio_processor : AudioProcessor
            音声処理クラス
        """
        # 時間と色相を更新
        self.wave_time += self.wave_speed
        self.breath_phase += self.breath_speed  # 呼吸の位相を更新
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
                
                # 音の強度を計算（光の線の生成に使用）
                sound_intensity = np.mean(np.abs(resampled_wave)) * 2.0
                sound_intensity = min(1.0, sound_intensity)
                
                # 光の線を生成
                self._spawn_light_line(sound_intensity)
                
                # 波形とスペクトラムを組み合わせる
                combined = 0.5 * resampled_wave + 0.5 * spectrum_values
                
                # 有機的な波動を生成（呼吸のような自然な揺らぎ）
                organic_wave = self._create_harmonic_wave(0.06)  # 揺らぎを適度に増加
                
                # 音声に反応する波動を生成
                reactive_wave = combined * self.wave_scale
                
                # 最終的な波形を計算
                radius_raw = self.base_radius + reactive_wave + organic_wave
                
                # 内側と外側の波形を計算（有機的な波動の振幅を調整）
                inner_radius_raw = self.base_radius * 0.85 + reactive_wave * 0.7 + self._create_harmonic_wave(0.04, 0.5)
                outer_radius_raw = self.base_radius * 1.15 + reactive_wave * 1.2 + self._create_harmonic_wave(0.07, 1.0)
                
                # スプライン補間で滑らかな閉曲線を作成
                def create_smooth_closed_curve(radius_values, theta_values, smoothness=0.0):
                    # 極座標からデカルト座標に変換
                    x = radius_values * np.cos(theta_values)
                    y = radius_values * np.sin(theta_values)
                    
                    # スプライン補間のためのパラメータ（周期的なスプライン）
                    tck, u = splprep([x, y], s=smoothness, per=True)
                    
                    # 補間された曲線を生成（元の点数と同じ点数で評価）
                    u_new = np.linspace(0, 1, len(radius_values), endpoint=True)
                    x_new, y_new = splev(u_new, tck)
                    
                    return x_new, y_new
                
                # 滑らかな閉曲線を生成
                theta_closed = self.theta  # すでにendpoint=Trueで生成されているthetaを使用
                x, y = create_smooth_closed_curve(radius_raw, theta_closed, smoothness=0.1)
                inner_x, inner_y = create_smooth_closed_curve(inner_radius_raw, theta_closed, smoothness=0.1)
                outer_x, outer_y = create_smooth_closed_curve(outer_radius_raw, theta_closed, smoothness=0.1)
                
                # 波形の履歴に追加（スプライン補間後のデータ）
                if len(x) > 0:
                    # 極座標に戻す（履歴用）
                    new_radius = np.sqrt(x**2 + y**2)
                    self.wave_history.append(new_radius)
                    if len(self.wave_history) > self.history_length:
                        self.wave_history.pop(0)
                
                # 色を計算
                main_r, main_g, main_b = [int(c * 255) for c in colorsys.hsv_to_rgb(self.hue, 0.8, 1.0)]
                comp_r, comp_g, comp_b = [int(c * 255) for c in colorsys.hsv_to_rgb(complement_hue, 0.7, 0.8)]
                inner_r, inner_g, inner_b = [int(c * 255) for c in colorsys.hsv_to_rgb((self.hue + 0.1) % 1.0, 0.7, 0.9)]
                outer_r, outer_g, outer_b = [int(c * 255) for c in colorsys.hsv_to_rgb((self.hue - 0.1) % 1.0, 0.9, 0.8)]
                
                # 波形を更新
                self.wave_curve.setData(x, y)
                self.wave_curve.setPen(pg.mkPen(color=QColor(main_r, main_g, main_b, 200), width=2))
                
                # 内側と外側の波形を更新
                self.inner_wave_curve.setData(inner_x, inner_y)
                self.inner_wave_curve.setPen(pg.mkPen(color=QColor(inner_r, inner_g, inner_b, 150), width=1.5))
                
                self.outer_wave_curve.setData(outer_x, outer_y)
                self.outer_wave_curve.setPen(pg.mkPen(color=QColor(outer_r, outer_g, outer_b, 130), width=1.5))
                
                # グロー効果
                self.wave_glow_curve.setData(x, y)
                self.wave_glow_curve.setPen(pg.mkPen(color=QColor(comp_r, comp_g, comp_b, 40), width=15))
                
                # 残像効果（履歴の波形を描画）
                for i, hist_radius in enumerate(self.wave_history[:-1]):
                    decay_factor = (i + 1) / len(self.wave_history)
                    opacity = int(40 * decay_factor * self.history_decay)
                    hx, hy = self._polar_to_cartesian(hist_radius, self.theta)
                    
                # 光の線を更新
                for i, line in enumerate(self.light_lines):
                    line['life'] -= 0.01
                    if line['life'] <= 0:
                        # 寿命が尽きた線を削除
                        self.light_lines.remove(line)
                        self.line_items[i].setData([], [])
                        continue
                        
                    # 線の不透明度を計算（寿命に応じて）
                    life_factor = line['life'] / self.line_life_max
                    opacity = int(line['max_opacity'] * life_factor)
                    
                    # 線の色を計算
                    lr, lg, lb = [int(c * 255) for c in colorsys.hsv_to_rgb(line['hue'], 0.8, 1.0)]
                    
                    # 線の座標を計算（ベジェ曲線で滑らかに）
                    t = np.linspace(0, 1, 20)
                    sx, sy = self._polar_to_cartesian(line['start'][0], line['start'][1])
                    mx, my = self._polar_to_cartesian(line['mid'][0], line['mid'][1])
                    ex, ey = self._polar_to_cartesian(line['end'][0], line['end'][1])
                    
                    # 二次ベジェ曲線
                    bx = (1-t)**2 * sx + 2*(1-t)*t * mx + t**2 * ex
                    by = (1-t)**2 * sy + 2*(1-t)*t * my + t**2 * ey
                    
                    # 線の長さをチェック - 異常に長い線は描画しない
                    line_length = np.sqrt((ex-sx)**2 + (ey-sy)**2)
                    max_allowed_length = 1.0  # 画面の半分程度を最大長とする
                    
                    # 線を描画（長さが適切な場合のみ）
                    if i < len(self.line_items) and line_length <= max_allowed_length:
                        self.line_items[i].setData(bx, by)
                        self.line_items[i].setPen(pg.mkPen(color=QColor(lr, lg, lb, opacity), width=line['width']))
                    else:
                        # 異常な線は空のデータで描画をスキップ
                        self.line_items[i].setData([], [])
                
                # 粒子を更新
                for p in self.particles:
                    # 粒子の位置を更新
                    p['x'] += p['vx']
                    p['y'] += p['vy']
                    
                    # 粒子の寿命を減少
                    p['life'] -= 0.002
                    
                    # 画面外に出たか寿命が尽きた粒子をリセット
                    if abs(p['x']) > 1 or abs(p['y']) > 1 or p['life'] <= 0:
                        # 中心付近から発生するように
                        angle = random.random() * 2 * np.pi
                        distance = random.random() * 0.2 + 0.1
                        p['x'] = distance * np.cos(angle)
                        p['y'] = distance * np.sin(angle)
                        
                        # 外側に向かって移動
                        speed = random.uniform(0.0005, 0.002)
                        p['vx'] = speed * np.cos(angle)
                        p['vy'] = speed * np.sin(angle)
                        
                        p['size'] = random.uniform(1.5, 3.5)
                        p['opacity'] = random.uniform(20, 80)
                        p['life'] = random.uniform(0.8, 1.0)
                
                # 粒子の描画
                spots = [{'pos': (p['x'], p['y']), 
                          'brush': QColor(255, 255, 255, int(p['opacity'] * p['life'])), 
                          'size': p['size'] * p['life']} 
                         for p in self.particles]
                self.particle_item.setData(spots)

                # 光のコアを更新（滑らかに鼓動する）
                self.core_pulse_phase += 0.02
                pulse_base = np.sin(self.core_pulse_phase) * 0.008
                envelope = 0.5 * (1 + np.sin(self.core_pulse_phase * 0.5))
                pulse_variation = pulse_base * envelope
                core_r = self.core_radius + pulse_variation
                
                # 音の強度に応じてコアが反応
                core_r += sound_intensity * 0.02
                
                # 円形のコアを描画するためのスプライン補間
                theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
                
                # コアの円は完全な円なので、単純に最初の点を最後に追加
                cx = core_r * np.cos(np.append(theta, theta[0]))
                cy = core_r * np.sin(np.append(theta, theta[0]))
                
                # 内側のコア
                inner_core_r = self.inner_core_radius + pulse_variation * 0.5 + sound_intensity * 0.01
                icx = inner_core_r * np.cos(np.append(theta, theta[0]))
                icy = inner_core_r * np.sin(np.append(theta, theta[0]))
                
                # コアの色を計算
                core_r_color, core_g, core_b = [int(c * 255) for c in colorsys.hsv_to_rgb(self.hue, 0.5, 1.0)]
                core_alpha = int(self.core_alpha + 30 * envelope + sound_intensity * 50)
                
                inner_hue = (self.hue + 0.05) % 1.0
                inner_r_color, inner_g, inner_b = [int(c * 255) for c in colorsys.hsv_to_rgb(inner_hue, 0.6, 1.0)]
                inner_alpha = int(150 + 50 * envelope + sound_intensity * 55)

                # コアを描画
                self.core_circle.setData(cx, cy)
                self.core_circle.setPen(pg.mkPen(color=QColor(core_r_color, core_g, core_b, core_alpha), width=20))
                
                self.inner_core.setData(icx, icy)
                self.inner_core.setPen(pg.mkPen(color=QColor(inner_r_color, inner_g, inner_b, inner_alpha), width=15))

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