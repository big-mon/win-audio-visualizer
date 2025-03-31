#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
オーディオデータのキャプチャと処理を担当するモジュール
"""

import numpy as np
import sounddevice as sd
import queue
from scipy.fft import rfft, rfftfreq

class AudioProcessor:
    """
    オーディオデータのキャプチャと処理を行うクラス
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
        self.stream = None
        self.running = False

        # FFT用の周波数軸を初期化
        self.spectrum_x = rfftfreq(window_size, 1 / sample_rate)

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
            print("\nデバイスIDを指定して再試行してください。例: processor.start_capture(2)")
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
        ndarray or None
            オーディオデータ、データがない場合はNone
        """
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def process_audio_data(self, data):
        """
        オーディオデータを処理してFFTを計算

        Parameters
        ----------
        data : ndarray
            処理するオーディオデータ

        Returns
        -------
        tuple
            (波形データ, スペクトラムデータ)
        """
        if data is None:
            return None, None

        # 振幅スペクトラム
        spectrum = rfft(data * np.hanning(len(data)))
        spectrum_mag = np.abs(spectrum)
        spectrum_mag = np.maximum(spectrum_mag, 1e-10)

        # 対数スケール(db)に変換
        spectrum_db = 20 * np.log10(spectrum_mag)

        return data, spectrum_db