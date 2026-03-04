from __future__ import annotations

import unittest

from src.indicators.atr import calculate_atr
from src.indicators.bollinger import calculate_bollinger_bands
from src.indicators.ema import calculate_ema
from src.indicators.macd import calculate_macd
from src.indicators.rsi import calculate_rsi


def _sequence(size: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + (idx * step) for idx in range(size)]


class IndicatorTests(unittest.TestCase):
    def test_ema_length_and_seed(self) -> None:
        closes = _sequence(30)
        ema_10 = calculate_ema(closes, 10)

        self.assertEqual(len(ema_10), len(closes))
        self.assertIsNone(ema_10[8])
        self.assertIsNotNone(ema_10[9])
        self.assertGreater(float(ema_10[-1]), float(ema_10[9]))

    def test_rsi_rising_series_near_overbought(self) -> None:
        closes = _sequence(40)
        rsi = calculate_rsi(closes, 14)

        self.assertEqual(len(rsi), len(closes))
        self.assertIsNone(rsi[13])
        self.assertIsNotNone(rsi[-1])
        self.assertGreater(float(rsi[-1]), 90.0)

    def test_macd_shapes(self) -> None:
        closes = _sequence(100)
        macd = calculate_macd(closes)

        self.assertEqual(set(macd.keys()), {"macd", "signal", "histogram"})
        self.assertEqual(len(macd["macd"]), len(closes))
        self.assertEqual(len(macd["signal"]), len(closes))
        self.assertEqual(len(macd["histogram"]), len(closes))
        self.assertIsNone(macd["macd"][24])
        self.assertIsNotNone(macd["macd"][25])

    def test_atr_positive_when_prices_move(self) -> None:
        highs = [101.0, 103.0, 104.0, 106.0, 105.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]
        lows = [99.0, 100.0, 101.0, 103.0, 102.0, 103.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0]
        closes = [100.0, 102.0, 103.0, 104.0, 103.5, 106.0, 107.0, 108.0, 109.0, 109.5, 111.0, 112.0, 113.0, 114.0, 115.0]

        atr = calculate_atr(highs, lows, closes, 14)

        self.assertEqual(len(atr), len(closes))
        self.assertIsNone(atr[12])
        self.assertIsNotNone(atr[13])
        self.assertGreater(float(atr[13]), 0.0)

    def test_bollinger_bounds_ordering(self) -> None:
        closes = _sequence(50)
        bb = calculate_bollinger_bands(closes, period=20, std_dev_mult=2)

        self.assertEqual(len(bb["middle"]), len(closes))
        self.assertIsNone(bb["middle"][18])
        self.assertIsNotNone(bb["middle"][19])

        upper = float(bb["upper"][-1])
        middle = float(bb["middle"][-1])
        lower = float(bb["lower"][-1])
        self.assertGreaterEqual(upper, middle)
        self.assertGreaterEqual(middle, lower)

    def test_empty_inputs_return_empty(self) -> None:
        self.assertEqual(calculate_ema([], 21), [])
        self.assertEqual(calculate_rsi([], 14), [])
        self.assertEqual(calculate_atr([], [], [], 14), [])
        self.assertEqual(calculate_bollinger_bands([], 20, 2.0), {"middle": [], "upper": [], "lower": []})


if __name__ == "__main__":
    unittest.main()
