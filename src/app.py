#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Windows Audio Session API (WASAPI)を使用したオーディオビジュアライザー
PC上で再生されている音声をキャプチャして可視化します
"""

from audio_processor import AudioProcessor
from visualizer import Visualizer

def main():
    """
    メイン関数
    アプリケーションのエントリーポイント
    """
    # オーディオプロセッサーとビジュアライザーを初期化
    processor = AudioProcessor()
    visualizer = Visualizer()

    try:
        # オーディオキャプチャを開始
        processor.start_capture()

        # ビジュアライゼーションを開始
        visualizer.start_animation(processor)

    except KeyboardInterrupt:
        print("\nアプリケーションを停止しました")
    finally:
        # キャプチャを停止
        processor.stop_capture()

if __name__ == "__main__":
    main()