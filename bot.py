import ccxt
import pandas as pd
import numpy as np
import requests
import time
import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

# ═════════════════════════════════════════════════════════════════
# تنظیمات
# ═════════════════════════════════════════════════════════════════

TOKEN = os.environ.get('TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')
TIMEFRAME = os.environ.get('TIMEFRAME', '15m')

SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 
    'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT',
    'LINK/USDT', 'POL/USDT'
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
# تلگرام
# ═════════════════════════════════════════════════════════════════

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send(self, text: str, parse_mode: str = 'HTML') -> bool:
        if not self.token or not self.chat_id:
            return False
        try:
            r = requests.post(
                f"{self.base_url}/sendMessage",
                json={'chat_id': self.chat_id, 'text': text, 'parse_mode': parse_mode, 'disable_web_page_preview': True},
                timeout=15
            )
            if r.status_code == 200:
                logger.info("📨 پیام ارسال شد")
                return True
            logger.error(f"❌ خطای تلگرام: {r.status_code}")
            return False
        except Exception as e:
            logger.error(f"❌ خطا در ارسال: {e}")
            return False

# ═════════════════════════════════════════════════════════════════
# دیتا
# ═════════════════════════════════════════════════════════════════

class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.kucoin({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.last_request = 0
        self.min_delay = 1.5
    
    def fetch(self, symbol: str, tf: str, limit: int = 200) -> Optional[pd.DataFrame]:
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        for attempt in range(3):
            try:
                self.last_request = time.time()
                ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                if ohlcv and len(ohlcv) > 50:
                    df = pd.DataFrame(ohlcv, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df.set_index('timestamp')
            except ccxt.RateLimitExceeded:
                time.sleep(3 ** attempt)
            except Exception as e:
                logger.warning(f"⚠️ {symbol}: {str(e)[:80]}")
                time.sleep(1)
        return None

# ═════════════════════════════════════════════════════════════════
# اندیکاتورها
# ═════════════════════════════════════════════════════════════════

class Indicators:
    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        for p in [8, 13, 21, 34, 55, 89, 200]:
            df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['close'].shift())
        lc = abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        df['rsi_ma'] = df['rsi'].rolling(14).mean()
        
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
        
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
        
        df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        df['senkou_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        
        recent_high = df['high'].rolling(50).max()
        recent_low = df['low'].rolling(50).min()
        diff = recent_high - recent_low
        df['fib_382'] = recent_high - 0.382 * diff
        df['fib_500'] = recent_high - 0.500 * diff
        df['fib_618'] = recent_high - 0.618 * diff
        
        df['swing_high'] = df['high'].rolling(5, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(5, center=True).min() == df['low']
        
        df['dist_ema8'] = (df['close'] - df['ema8']) / df['ema8'] * 100
        
        return df

# ═════════════════════════════════════════════════════════════════
# سیگنال
# ═════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    signal: str  # 'LONG' یا 'SHORT'
    entry: float
    sl: float
    tp1: float
    tp2: float
    strategy: str
    confidence: int
    reasons: List[str]
    df: pd.DataFrame
    
    @property
    def time(self):
        return self.df.index[-1]
    
    def rr(self, target: float) -> float:
        if self.signal == 'LONG':
            risk = self.entry - self.sl
            reward = target - self.entry
        else:
            risk = self.sl - self.entry
            reward = self.entry - target
        return round(reward / risk, 2) if risk > 0 else 0

# ═════════════════════════════════════════════════════════════════
# استراتژی‌ها - LONG و SHORT
# ═════════════════════════════════════════════════════════════════

class StrategyEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.last = df.iloc[-1]
        self.prev = df.iloc[-2]
        self.prev2 = df.iloc[-3]
        self.prev3 = df.iloc[-4]
        self.prev4 = df.iloc[-5]
    
    # ─── فیلترهای مشترک ───────────────────────────────────────
    def common_filters(self, direction: str) -> Tuple[bool, List[str]]:
        reasons = []
        
        if direction == 'LONG':
            if self.last['dist_ema8'] > 3:
                return False, [f"❌ پامپ شدید ({self.last['dist_ema8']:.1f}%)"]
        else:  # SHORT
            if self.last['dist_ema8'] < -3:
                return False, [f"❌ دامپ شدید ({self.last['dist_ema8']:.1f}%)"]
        
        if self.last['atr_pct'] < 0.1:
            return False, ["❌ ATR خیلی کم"]
        
        return True, []
    
    # ═══════════════════════════════════════════════════════════
    # استراتژی‌های LONG
    # ═══════════════════════════════════════════════════════════
    
    def ema_golden_cross(self) -> Optional[SignalResult]:
        cross = (self.prev['ema8'] <= self.prev['ema21'] and 
                 self.last['ema8'] > self.last['ema21'])
        if not cross:
            return None
        
        ema55_ok = self.last['ema21'] > self.last['ema55']
        rsi_ok = 40 < self.last['rsi'] < 65
        vol_ok = self.last['vol_ratio'] > 0.8
        macd_ok = self.last['macd'] > self.last['macd_signal']
        
        confidence = 50
        reasons = ['EMA8 کراس EMA21']
        
        if ema55_ok: confidence += 15; reasons.append('EMA21 > EMA55')
        if rsi_ok: confidence += 10; reasons.append(f'RSI={self.last["rsi"]:.1f}')
        if vol_ok: confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        if macd_ok: confidence += 10; reasons.append('MACD صعودی')
        
        if confidence < 60:
            return None
        
        entry = self.last['close']
        sl = min(self.last['ema21'], self.last['low']) - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'EMA Golden Cross', confidence, reasons, self.df)
    
    def rsi_divergence_long(self) -> Optional[SignalResult]:
        lows = self.df['low'].iloc[-30:-1]
        rsi_vals = self.df['rsi'].iloc[-30:-1]
        recent_lows = lows[lows == lows.rolling(5, center=True).min()]
        recent_rsi = rsi_vals[lows == lows.rolling(5, center=True).min()]
        
        if len(recent_lows) < 2:
            return None
        
        low1, low2 = recent_lows.iloc[-2], recent_lows.iloc[-1]
        rsi1, rsi2 = recent_rsi.iloc[-2], recent_rsi.iloc[-1]
        
        if not (low2 < low1 * 0.995 and rsi2 > rsi1 * 1.01):
            return None
        
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 70
        reasons = [f'RSI Divergence', f'قیمت: {low1:.4f} → {low2:.4f}', f'RSI: {rsi1:.1f} → {rsi2:.1f}']
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = low2 - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2.5
        tp2 = entry + risk * 4
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'RSI Divergence', confidence, reasons, self.df)
    
    def bb_squeeze_breakout_long(self) -> Optional[SignalResult]:
        bb_width_before = self.df['bb_width'].iloc[-6:-1].mean()
        if bb_width_before > 3:
            return None
        
        if not (self.last['close'] > self.last['bb_mid'] and 
                self.last['close'] > self.prev['close'] and
                self.last['close'] > self.last['open']):
            return None
        
        if self.last['vol_ratio'] < 1.5:
            return None
        
        confidence = 65
        reasons = [f'BB Squeeze ({bb_width_before:.1f}%)', f'Breakout {self.last["close"]:.4f}', f'Vol={self.last["vol_ratio"]:.1f}x']
        
        entry = self.last['close']
        sl = self.last['bb_lower'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = self.last['bb_upper']
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'BB Squeeze Breakout', confidence, reasons, self.df)
    
    def macd_histogram_reversal_long(self) -> Optional[SignalResult]:
        hist_trend = (self.prev3['macd_hist'] > self.prev2['macd_hist'] > 
                      self.prev['macd_hist'])
        hist_now = self.last['macd_hist'] > self.prev['macd_hist']
        
        if not (hist_trend and hist_now):
            return None
        
        if self.last['macd'] > 0:
            return None
        
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 60
        reasons = ['MACD Histogram Reversal', f'MACD={self.last["macd"]:.4f}', f'Hist={self.last["macd_hist"]:.4f}']
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['low'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'MACD Reversal', confidence, reasons, self.df)
    
    def stoch_oversold_bounce(self) -> Optional[SignalResult]:
        if not (self.prev['stoch_k'] < 20 and self.last['stoch_k'] > self.prev['stoch_k']):
            return None
        
        if not self.last['stoch_k'] > self.last['stoch_d']:
            return None
        
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 55
        reasons = [f'Stochastic Bounce', f'%K={self.last["stoch_k"]:.1f} %D={self.last["stoch_d"]:.1f}']
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['low'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Stochastic Bounce', confidence, reasons, self.df)
    
    def fib_bounce_long(self) -> Optional[SignalResult]:
        dist_618 = abs(self.last['close'] - self.last['fib_618']) / self.last['close'] * 100
        dist_500 = abs(self.last['close'] - self.last['fib_500']) / self.last['close'] * 100
        
        if min(dist_618, dist_500) > 1.5:
            return None
        
        if not self.last['close'] > self.last['open']:
            return None
        
        level = '0.618' if dist_618 < dist_500 else '0.500'
        price = self.last['fib_618'] if dist_618 < dist_500 else self.last['fib_500']
        
        confidence = 60
        reasons = [f'Fibonacci {level} Bounce', f'قیمت: {self.last["close"]:.4f}', f'سطح: {price:.4f}']
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['low'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = self.last['fib_382']
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Fibonacci Bounce', confidence, reasons, self.df)
    
    def sr_flip_long(self) -> Optional[SignalResult]:
        recent_highs = self.df[self.df['swing_high']]['high']
        if len(recent_highs) < 2:
            return None
        
        resistance = recent_highs.iloc[-2]
        
        if not self.last['close'] > resistance * 1.01:
            return None
        
        if not self.prev['close'] < resistance:
            return None
        
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 65
        reasons = [f'S/R Flip', f'مقاومت: {resistance:.4f}', f'قیمت: {self.last["close"]:.4f}']
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = resistance - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'S/R Flip', confidence, reasons, self.df)
    
    # ═══════════════════════════════════════════════════════════
    # استراتژی‌های SHORT
    # ═══════════════════════════════════════════════════════════
    
    def ema_death_cross(self) -> Optional[SignalResult]:
        cross = (self.prev['ema8'] >= self.prev['ema21'] and 
                 self.last['ema8'] < self.last['ema21'])
        if not cross:
            return None
        
        ema55_ok = self.last['ema21'] < self.last['ema55']
        rsi_ok = 35 < self.last['rsi'] < 70
        vol_ok = self.last['vol_ratio'] > 0.8
        macd_ok = self.last['macd'] < self.last['macd_signal']
        
        confidence = 50
        reasons = ['EMA8 کراس پایین EMA21']
        
        if ema55_ok: confidence += 15; reasons.append('EMA21 < EMA55')
        if rsi_ok: confidence += 10; reasons.append(f'RSI={self.last["rsi"]:.1f}')
        if vol_ok: confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        if macd_ok: confidence += 10; reasons.append('MACD نزولی')
        
        if confidence < 60:
            return None
        
        entry = self.last['close']
        sl = max(self.last['ema21'], self.last['high']) + self.last['atr'] * 0.5
        risk = sl - entry
        tp1 = entry - risk * 2
        tp2 = entry - risk * 3.5
        
        return SignalResult('SHORT', entry, sl, tp1, tp2, 'EMA Death Cross', confidence, reasons, self.df)
    
    def rsi_divergence_short(self) -> Optional[SignalResult]:
        highs = self.df['high'].iloc[-30:-1]
        rsi_vals = self.df['rsi'].iloc[-30:-1]
        recent_highs = highs[highs == highs.rolling(5, center=True).max()]
        recent_rsi = rsi_vals[highs == highs.rolling(5, center=True).max()]
        
        if len(recent_highs) < 2:
            return None
        
        high1, high2 = recent_highs.iloc[-2], recent_highs.iloc[-1]
        rsi1, rsi2 = recent_rsi.iloc[-2], recent_rsi.iloc[-1]
        
        if not (high2 > high1 * 1.005 and rsi2 < rsi1 * 0.99):
            return None
        
        if not self.last['close'] < self.last['open']:
            return None
        
        confidence = 70
        reasons = [f'RSI Divergence نزولی', f'قیمت: {high1:.4f} → {high2:.4f}', f'RSI: {rsi1:.1f} → {rsi2:.1f}']
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = high2 + self.last['atr'] * 0.5
        risk = sl - entry
        tp1 = entry - risk * 2.5
        tp2 = entry - risk * 4
        
        return SignalResult('SHORT', entry, sl, tp1, tp2, 'RSI Divergence Short', confidence, reasons, self.df)
    
    def bb_squeeze_breakdown(self) -> Optional[SignalResult]:
        bb_width_before = self.df['bb_width'].iloc[-6:-1].mean()
        if bb_width_before > 3:
            return None
        
        if not (self.last['close'] < self.last['bb_mid'] and 
                self.last['close'] < self.prev['close'] and
                self.last['close'] < self.last['open']):
            return None
        
        if self.last['vol_ratio'] < 1.5:
            return None
        
        confidence = 65
        reasons = [f'BB Squeeze Breakdown', f'Breakdown {self.last["close"]:.4f}', f'Vol={self.last["vol_ratio"]:.1f}x']
        
        entry = self.last['close']
        sl = self.last['bb_upper'] + self.last['atr'] * 0.5
        risk = sl - entry
        tp1 = entry - risk * 2
        tp2 = self.last['bb_lower']
        
        return SignalResult('SHORT', entry, sl, tp1, tp2, 'BB Squeeze Breakdown', confidence, reasons, self.df)
    
    def macd_histogram_reversal_short(self) -> Optional[SignalResult]:
        hist_trend = (self.prev3['macd_hist'] < self.prev2['macd_hist'] < 
                      self.prev['macd_hist'])
        hist_now = self.last['macd_hist'] < self.prev['macd_hist']
        
        if not (hist_trend and hist_now):
            return None
        
        if self.last['macd'] < 0:
            return None
        
        if not self.last['close'] < self.last['open']:
            return None
        
        confidence = 60
        reasons = ['MACD Histogram Reversal نزولی', f'MACD={self.last["macd"]:.4f}', f'Hist={self.last["macd_hist"]:.4f}']
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['high'] + self.last['atr'] * 0.5
        risk = sl - entry
        tp1 = entry - risk * 2
        tp2 = entry - risk * 3.5
        
        return SignalResult('SHORT', entry, sl, tp1, tp2, 'MACD Reversal Short', confidence, reasons, self.df)
    
    def stoch_overbought_fall(self) -> Optional[SignalResult]:
        if not (self.prev['stoch_k'] > 80 and self.last['stoch_k'] < self.prev['stoch_k']):
            return None
        
        if not self.last['stoch_k'] < self.last['stoch_d']:
            return None
        
        if not self.last['close'] < self.last['open']:
            return None
        
        confidence = 55
        reasons = [f'Stochastic Overbought Fall', f'%K={self.last["stoch_k"]:.1f} %D={self.last["stoch_d"]:.1f}']
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['high'] + self.last['atr'] * 0.5
        risk = sl - entry
        tp1 = entry - risk * 2
        tp2 = entry - risk * 3
        
        return SignalResult('SHORT', entry, sl, tp1, tp2, 'Stochastic Fall', confidence, reasons, self.df)
    
    def fib_reject_short(self) -> Optional[SignalResult]:
        dist_382 = abs(self.last['close'] - self.last['fib_382']) / self.last['close'] * 100
        dist_500 = abs(self.last['close'] - self.last['fib_500']) / self.last['close'] * 100
        
        if min(dist_382, dist_500) > 1.5:
            return None
        
        if not self.last['close'] < self.last['open']:
            return None
        
        level = '0.382' if dist_382 < dist_500 else '0.500'
        price = self.last['fib_382'] if dist_382 < dist_500 else self.last['fib_500']
        
        confidence = 60
        reasons = [f'Fibonacci {level} Reject', f'قیمت: {self.last["close"]:.4f}', f'سطح: {price:.4f}']
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['high'] + self.last['atr'] * 0.5
        risk = sl - entry
        tp1 = entry - risk * 2
        tp2 = self.last['fib_618']
        
        return SignalResult('SHORT', entry, sl, tp1, tp2, 'Fibonacci Reject', confidence, reasons, self.df)
    
    def sr_flip_short(self) -> Optional[SignalResult]:
        recent_lows = self.df[self.df['swing_low']]['low']
        if len(recent_lows) < 2:
            return None
        
        support = recent_lows.iloc[-2]
        
        if not self.last['close'] < support * 0.99:
            return None
        
        if not self.prev['close
