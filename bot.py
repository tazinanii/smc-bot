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

# تنظیم لاگ
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
        
        # EMA
        for p in [8, 13, 21, 34, 55, 89, 200]:
            df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        
        # SMA
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        
        # ATR
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['close'].shift())
        lc = abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        df['rsi_ma'] = df['rsi'].rolling(14).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
        
        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # VWAP
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Ichimoku
        df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        df['senkou_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        
        # Fibonacci levels
        recent_high = df['high'].rolling(50).max()
        recent_low = df['low'].rolling(50).min()
        diff = recent_high - recent_low
        df['fib_382'] = recent_high - 0.382 * diff
        df['fib_500'] = recent_high - 0.500 * diff
        df['fib_618'] = recent_high - 0.618 * diff
        
        # Swing points
        df['swing_high'] = df['high'].rolling(5, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(5, center=True).min() == df['low']
        
        # Distance from EMA
        df['dist_ema8'] = (df['close'] - df['ema8']) / df['ema8'] * 100
        df['dist_ema21'] = (df['close'] - df['ema21']) / df['ema21'] * 100
        
        return df

# ═════════════════════════════════════════════════════════════════
# سیگنال
# ═════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    signal: str
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
        risk = abs(self.entry - self.sl)
        reward = abs(target - self.entry)
        return round(reward / risk, 2) if risk > 0 else 0

# ═════════════════════════════════════════════════════════════════
# استراتژی‌ها
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
    def common_filters(self) -> Tuple[bool, List[str]]:
        """فیلترهایی که همه استراتژی‌ها باید رد کنن"""
        reasons = []
        
        # قیمت نباید خیلی دور از EMA8 باشه (پامپ نشده باشه)
        if self.last['dist_ema8'] > 3:
            return False, [f"❌ پامپ شدید ({self.last['dist_ema8']:.1f}%)"]
        
        # ATR باید معقول باشه
        if self.last['atr_pct'] < 0.1:
            return False, ["❌ ATR خیلی کم"]
        
        return True, []
    
    # ─── استراتژی ۱: EMA Golden Cross ─────────────────────────
    def ema_golden_cross(self) -> Optional[SignalResult]:
        """EMA8 کراس EMA21 با تایید EMA55"""
        
        # کراس در ۲ کندل اخیر
        cross = (self.prev['ema8'] <= self.prev['ema21'] and 
                 self.last['ema8'] > self.last['ema21'])
        
        if not cross:
            return None
        
        # تاییدها
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
    
    # ─── استراتژی ۲: RSI Divergence ──────────────────────────
    def rsi_divergence(self) -> Optional[SignalResult]:
        """واگرایی RSI صعودی"""
        
        # پیدا کردن دو کف قیمت
        lows = self.df['low'].iloc[-30:-1]
        rsi_vals = self.df['rsi'].iloc[-30:-1]
        
        # دو کف اخیر
        recent_lows = lows[lows == lows.rolling(5, center=True).min()]
        recent_rsi = rsi_vals[lows == lows.rolling(5, center=True).min()]
        
        if len(recent_lows) < 2:
            return None
        
        low1, low2 = recent_lows.iloc[-2], recent_lows.iloc[-1]
        rsi1, rsi2 = recent_rsi.iloc[-2], recent_rsi.iloc[-1]
        
        # واگرایی: قیمت پایین‌تر، RSI بالاتر
        if not (low2 < low1 * 0.995 and rsi2 > rsi1 * 1.01):
            return None
        
        # تایید
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 70
        reasons = [
            f'RSI Divergence',
            f'قیمت: {low1:.4f} → {low2:.4f}',
            f'RSI: {rsi1:.1f} → {rsi2:.1f}'
        ]
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10
            reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = low2 - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2.5
        tp2 = entry + risk * 4
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'RSI Divergence', confidence, reasons, self.df)
    
    # ─── استراتژی ۳: Bollinger Squeeze + Breakout ───────────
    def bb_squeeze_breakout(self) -> Optional[SignalResult]:
        """باند تنگ + شکست"""
        
        # باند تنگ در ۵ کندل قبل
        bb_width_before = self.df['bb_width'].iloc[-6:-1].mean()
        if bb_width_before > 3:
            return None
        
        # شکست به سمت بالا
        if not (self.last['close'] > self.last['bb_mid'] and 
                self.last['close'] > self.prev['close'] and
                self.last['close'] > self.last['open']):
            return None
        
        # حجم بالا
        if self.last['vol_ratio'] < 1.5:
            return None
        
        confidence = 65
        reasons = [
            f'BB Squeeze ({bb_width_before:.1f}%)',
            f'Breakout {self.last["close"]:.4f}',
            f'Vol={self.last["vol_ratio"]:.1f}x'
        ]
        
        entry = self.last['close']
        sl = self.last['bb_lower'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = self.last['bb_upper']
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'BB Squeeze Breakout', confidence, reasons, self.df)
    
    # ─── استراتژی ۴: MACD Histogram Reversal ─────────────────
    def macd_histogram_reversal(self) -> Optional[SignalResult]:
        """برگشت هیستوگرام MACD"""
        
        # ۳ کندل قبل هیستوگرام نزولی
        hist_trend = (self.prev3['macd_hist'] > self.prev2['macd_hist'] > 
                      self.prev['macd_hist'])
        
        # کندل فعلی هیستوگرام صعودی
        hist_now = self.last['macd_hist'] > self.prev['macd_hist']
        
        if not (hist_trend and hist_now):
            return None
        
        # MACD زیر صفر باشه (نه بالای بازار)
        if self.last['macd'] > 0:
            return None
        
        # کندل صعودی
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 60
        reasons = [
            'MACD Histogram Reversal',
            f'MACD={self.last["macd"]:.4f}',
            f'Hist={self.last["macd_hist"]:.4f}'
        ]
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10
            reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['low'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'MACD Reversal', confidence, reasons, self.df)
    
    # ─── استراتژی ۵: Stochastic Oversold Bounce ────────────
    def stoch_oversold_bounce(self) -> Optional[SignalResult]:
        """برگشت از oversold استوکاستیک"""
        
        # %K از زیر ۲۰ برگشته
        if not (self.prev['stoch_k'] < 20 and self.last['stoch_k'] > self.prev['stoch_k']):
            return None
        
        # %K بالاتر از %D
        if not self.last['stoch_k'] > self.last['stoch_d']:
            return None
        
        # کندل صعودی
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 55
        reasons = [
            f'Stochastic Bounce',
            f'%K={self.last["stoch_k"]:.1f} %D={self.last["stoch_d"]:.1f}'
        ]
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10
            reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['low'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Stochastic Bounce', confidence, reasons, self.df)
    
    # ─── استراتژی ۶: Ichimoku Cloud Breakout ─────────────────
    def ichimoku_cloud_breakout(self) -> Optional[SignalResult]:
        """شکست ابر ایچیموکو"""
        
        # قیمت بالای ابر
        cloud_top = max(self.last['senkou_a'], self.last['senkou_b'])
        if not self.last['close'] > cloud_top:
            return None
        
        # Tenkan بالای Kijun
        if not self.last['tenkan'] > self.last['kijun']:
            return None
        
        # کندل صعودی
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 65
        reasons = [
            'Ichimoku Cloud Breakout',
            f'Tenkan={self.last["tenkan"]:.4f} > Kijun={self.last["kijun"]:.4f}'
        ]
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10
            reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = cloud_top - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Ichimoku Breakout', confidence, reasons, self.df)
    
    # ─── استراتژی ۷: Fibonacci Retracement Bounce ──────────
    def fib_bounce(self) -> Optional[SignalResult]:
        """برخورد از سطح فیبوناچی"""
        
        # قیمت باید نزدیک ۰.۶۱۸ یا ۰.۵۰۰ باشه
        dist_618 = abs(self.last['close'] - self.last['fib_618']) / self.last['close'] * 100
        dist_500 = abs(self.last['close'] - self.last['fib_500']) / self.last['close'] * 100
        
        if min(dist_618, dist_500) > 1.5:
            return None
        
        # کندل صعودی
        if not self.last['close'] > self.last['open']:
            return None
        
        level = '0.618' if dist_618 < dist_500 else '0.500'
        price = self.last['fib_618'] if dist_618 < dist_500 else self.last['fib_500']
        
        confidence = 60
        reasons = [
            f'Fibonacci {level} Bounce',
            f'قیمت: {self.last["close"]:.4f}',
            f'سطح: {price:.4f}'
        ]
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10
            reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['low'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = self.last['fib_382']
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Fibonacci Bounce', confidence, reasons, self.df)
    
    # ─── استراتژی ۸: Volume Profile Breakout ─────────────────
    def volume_profile_breakout(self) -> Optional[SignalResult]:
        """شکست با حجم قوی"""
        
        # سقف ۲۰ کندل اخیر
        recent_high = self.df['high'].iloc[-20:-2].max()
        
        # شکست
        if not (self.last['close'] > recent_high and 
                self.last['close'] > self.last['open']):
            return None
        
        # حجم خیلی بالا
        if self.last['vol_ratio'] < 2.0:
            return None
        
        # قبلش consolidation
        range_before = (self.df['high'].iloc[-10:-2].max() - 
                       self.df['low'].iloc[-10:-2].min()) / self.last['close'] * 100
        if range_before > 5:
            return None
        
        confidence = 70
        reasons = [
            f'Volume Breakout',
            f'سقف شکسته: {recent_high:.4f}',
            f'Vol={self.last["vol_ratio"]:.1f}x',
            f'Consolidation: {range_before:.1f}%'
        ]
        
        entry = self.last['close']
        sl = recent_high - self.last['atr'] * 1.0
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Volume Breakout', confidence, reasons, self.df)
    
    # ─── استراتژی ۹: VWAP Bounce ────────────────────────────
    def vwap_bounce(self) -> Optional[SignalResult]:
        """برخورد از VWAP"""
        
        # قیمت باید نزدیک VWAP باشه
        dist_vwap = abs(self.last['close'] - self.last['vwap']) / self.last['vwap'] * 100
        if dist_vwap > 1.0:
            return None
        
        # قبلش زیر VWAP بوده
        if not self.prev['close'] < self.prev['vwap']:
            return None
        
        # کندل صعودی
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 55
        reasons = [
            'VWAP Bounce',
            f'VWAP={self.last["vwap"]:.4f}',
            f'قیمت={self.last["close"]:.4f}'
        ]
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10
            reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = self.last['vwap'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'VWAP Bounce', confidence, reasons, self.df)
    
    # ─── استراتژی ۱۰: Support/Resistance Flip ────────────────
    def sr_flip(self) -> Optional[SignalResult]:
        """تبدیل مقاومت به حمایت"""
        
        # پیدا کردن سطح مقاومت اخیر
        recent_highs = self.df[self.df['swing_high']]['high']
        if len(recent_highs) < 2:
            return None
        
        resistance = recent_highs.iloc[-2]
        
        # قیمت باید بالای مقاومت باشه
        if not self.last['close'] > resistance * 1.01:
            return None
        
        # قبلش زیر مقاومت بوده
        if not self.prev['close'] < resistance:
            return None
        
        # کندل صعودی
        if not self.last['close'] > self.last['open']:
            return None
        
        confidence = 65
        reasons = [
            f'S/R Flip',
            f'مقاومت: {resistance:.4f}',
            f'قیمت: {self.last["close"]:.4f}'
        ]
        
        if self.last['vol_ratio'] > 1.0:
            confidence += 10
            reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = resistance - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'S/R Flip', confidence, reasons, self.df)
    
    # ─── تحلیل نهایی ────────────────────────────────────────
    def analyze(self) -> Optional[SignalResult]:
        """اجرای همه استراتژی‌ها و انتخاب بهترین"""
        
        # فیلتر مشترک
        ok, filter_reasons = self.common_filters()
        if not ok:
            return None
        
        strategies = [
            self.ema_golden_cross,
            self.rsi_divergence,
            self.bb_squeeze_breakout,
            self.macd_histogram_reversal,
            self.stoch_oversold_bounce,
            self.ichimoku_cloud_breakout,
            self.fib_bounce,
            self.volume_profile_breakout,
            self.vwap_bounce,
            self.sr_flip,
        ]
        
        signals = []
        for strategy in strategies:
            try:
                sig = strategy()
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"خطا در {strategy.__name__}: {e}")
        
        if not signals:
            return None
        
        # انتخاب بهترین (بالاترین confidence)
        best = max(signals, key=lambda x: x.confidence)
        
        # اگه چندتا استراتژی همزمان فعال باشن، confidence بیشتر
        if len(signals) > 1:
            best.confidence = min(100, best.confidence + 10)
            best.reasons.append(f'✅ {len(signals)} استراتژی همزمان')
        
        return best

# ═════════════════════════════════════════════════════════════════
# فرمت پیام
# ═════════════════════════════════════════════════════════════════

def format_signal(symbol: str, sig: SignalResult) -> str:
    emoji = "🟢" if sig.signal == 'LONG' else "🔴"
    stars = "⭐" * (sig.confidence // 20)
    
    return f"""
{emoji} <b>سیگنال {sig.signal} - {symbol}</b> {stars}

📊 <b>استراتژی:</b> {sig.strategy}
💯 <b>اعتماد:</b> {sig.confidence}%

━━━━━━━━━━━━━━━━━━━━━━
💰 <b>ورود:</b> <code>{sig.entry:.4f}</code>
⛔ <b>استاپ:</b> <code>{sig.sl:.4f}</code>
🎯 <b>TP1:</b> <code>{sig.tp1:.4f}</code> (R:R 1:{sig.rr(sig.tp1)})
🎯 <b>TP2:</b> <code>{sig.tp2:.4f}</code> (R:R 1:{sig.rr(sig.tp2)})
━━━━━━━━━━━━━━━━━━━━━━

📈 RSI: {sig.df.iloc[-1]['rsi']:.1f} | MACD: {sig.df.iloc[-1]['macd']:.4f}
📊 حجم: {sig.df.iloc[-1]['vol_ratio']:.1f}x | ATR: {sig.df.iloc[-1]['atr_pct']:.2f}%
📝 دلایل:
{chr(10).join('  • ' + r for r in sig.reasons)}

⏰ {sig.time.strftime('%Y-%m-%d %H:%M')} UTC

💡 50% TP1 | 30% TP2 | 20% Trailing
⚠️ ریسک خودتونو مدیریت کنید
"""

# ═════════════════════════════════════════════════════════════════
# مدیریت زمان
# ═════════════════════════════════════════════════════════════════

def get_tf_minutes(tf: str) -> int:
    return {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}.get(tf, 15)

def wait_next_candle(tf: str):
    minutes = get_tf_minutes(tf)
    now = datetime.now(timezone.utc)
    
    next_m = ((now.minute // minutes) + 1) * minutes
    if next_m >= 60:
        next_candle = (now + timedelta(hours=1)).replace(minute=next_m % 60, second=5, microsecond=0)
    else:
        next_candle = now.replace(minute=next_m, second=5, microsecond=0)
    
    wait = (next_candle - now).total_seconds()
    if wait > 0:
        logger.info(f"⏳ صبر {wait:.0f}s...")
        time.sleep(wait)

# ═════════════════════════════════════════════════════════════════
# اجرا
# ═════════════════════════════════════════════════════════════════

def main():
    telegram = TelegramNotifier(TOKEN, CHAT_ID)
    fetcher = DataFetcher()
    sent_signals: Dict[str, bool] = {}
    
    logger.info(f"🚀 شروع | {len(SYMBOLS)} جفت‌ارز | {TIMEFRAME}")
    
    while True:
        cycle_start = time.time()
        found = 0
        
        for symbol in SYMBOLS:
            try:
                df = fetcher.fetch(symbol, TIMEFRAME)
                if df is None or len(df) < 60:
                    continue
                
                df = Indicators.add_all(df)
                
                # دیباگ
                last = df.iloc[-1]
                logger.info(
                    f"🔍 {symbol} | "
                    f"قیمت:{last['close']:.4f} | "
                    f"RSI:{last['rsi']:.1f} | "
                    f"MACD:{last['macd']:.4f} | "
                    f"Vol:{last['vol_ratio']:.1f}x | "
                    f"BB:{last['bb_width']:.1f}%"
                )
                
                engine = StrategyEngine(df)
                signal = engine.analyze()
                
                if signal:
                    # جلوگیری از تکرار (۱۲ ساعت)
                    key = f"{symbol}_{signal.time.strftime('%Y%m%d%H')}"
                    if key in sent_signals:
                        continue
                    
                    msg = format_signal(symbol, signal)
                    if telegram.send(msg):
                        sent_signals[key] = True
                        found += 1
                        logger.info(f"✅ سیگنال {symbol}: {signal.strategy} ({signal.confidence}%)")
                else:
                    logger.info(f"❌ {symbol}: بدون سیگنال")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
        
        # پاک کردن سیگنال‌های قدیمی
        cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
        sent_signals = {k: v for k, v in sent_signals.items() 
                       if datetime.strptime(k.split('_')[1], '%Y%m%d%H').replace(tzinfo=timezone.utc) > cutoff}
        
        cycle_time = time.time() - cycle_start
        logger.info(f"🏁 دور: {cycle_time:.1f}s | سیگنال: {found}")
        wait_next_candle(TIMEFRAME)

if __name__ == '__main__':
    main()
