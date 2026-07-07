import ccxt
import pandas as pd
import numpy as np
import requests
import time
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

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
    handlers=[logging.FileHandler('bot.log'), logging.StreamHandler()]
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
                json={'chat_id': self.chat_id, 'text': text, 'parse_mode': parse_mode},
                timeout=15
            )
            return r.status_code == 200
        except Exception as e:
            logger.error(f"❌ تلگرام: {e}")
            return False
    
    def send_startup(self):
        self.send(f"🚀 <b>ربات فعال شد!</b>\n📊 {len(SYMBOLS)} جفت‌ارز | ⏱️ {TIMEFRAME}")

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
                wait = 3 ** attempt
                logger.warning(f"⏳ Rate limit {symbol}, صبر {wait}s...")
                time.sleep(wait)
            except Exception as e:
                logger.warning(f"⚠️ {symbol}: {str(e)[:80]}")
                time.sleep(1)
        
        logger.error(f"❌ {symbol}: دیتا دریافت نشد")
        return None

# ═════════════════════════════════════════════════════════════════
# اندیکاتورها
# ═════════════════════════════════════════════════════════════════

class Indicators:
    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        for p in [8, 21, 55, 200]:
            df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        
        df['sma50'] = df['close'].rolling(50).mean()
        
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
        
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        # فاصله از EMA
        df['dist_ema8'] = (df['close'] - df['ema8']) / df['ema8'] * 100
        df['dist_ema21'] = (df['close'] - df['ema21']) / df['ema21'] * 100
        
        # Swing High/Low
        df['swing_high'] = df['high'].rolling(5, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(5, center=True).min() == df['low']
        
        return df

# ═════════════════════════════════════════════════════════════════
# استراتژی‌ها — فقط سیگنال روی کف
# ═════════════════════════════════════════════════════════════════

class SignalResult:
    def __init__(self, signal, entry, sl, tp1, tp2, strategy, confidence, reasons, df):
        self.signal = signal
        self.entry = entry
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.strategy = strategy
        self.confidence = confidence
        self.reasons = reasons
        self.df = df
        self.time = df.index[-1]
    
    def rr(self, target):
        risk = abs(self.entry - self.sl)
        reward = abs(target - self.entry)
        return round(reward / risk, 2) if risk > 0 else 0

class StrategyEngine:
    def __init__(self, df):
        self.df = df
        self.last = df.iloc[-1]
        self.prev = df.iloc[-2]
        self.prev2 = df.iloc[-3]
    
    def is_uptrend(self) -> bool:
        """روند صعودی بلندمدت"""
        return self.last['ema21'] > self.last['ema55'] and self.last['close'] > self.last['ema200']
    
    def is_downtrend(self) -> bool:
        """روند نزولی"""
        return self.last['ema21'] < self.last['ema55']
    
    def find_support(self, lookback: int = 30) -> float:
        """پیدا کردن سطح حمایت اخیر"""
        recent = self.df.iloc[-lookback:]
        lows = recent[recent['swing_low']]['low']
        if len(lows) > 0:
            return lows.iloc[-1]
        return recent['low'].min()
    
    def find_resistance(self, lookback: int = 30) -> float:
        """پیدا کردن سطح مقاومت اخیر"""
        recent = self.df.iloc[-lookback:]
        highs = recent[recent['swing_high']]['high']
        if len(highs) > 0:
            return highs.iloc[-1]
        return recent['high'].max()
    
    def support_bounce(self) -> Optional[SignalResult]:
        """
        🔥 استراتژی ۱: برخورد از سطح حمایت
        قیمت باید به سطح حمایت برخورد کرده و برگشته باشه
        """
        if not self.is_uptrend():
            return None
        
        support = self.find_support(30)
        
        # آیا قیمت به ساپورت نزدیک شده؟
        dist_to_support = (self.last['close'] - support) / support * 100
        
        # باید نزدیک ساپورت باشه (نه خیلی دور)
        if dist_to_support > 1.5:
            return None
        
        # باید از ساپورت برگشته باشه (کندل صعودی)
        if not (self.last['close'] > self.last['open'] and self.last['low'] <= support * 1.005):
            return None
        
        # حجم باید بالا باشه
        if self.last['vol_ratio'] < 1.2:
            return None
        
        confidence = 70
        reasons = [f'Support Bounce @ {support:.4f}']
        
        if self.last['rsi'] < 50:
            confidence += 10
            reasons.append(f'RSI={self.last["rsi"]:.1f} (oversold)')
        
        entry = self.last['close']
        sl = support - self.last['atr'] * 1.0
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Support Bounce', confidence, reasons, self.df)
    
    def double_bottom(self) -> Optional[SignalResult]:
        """
        🔥 استراتژی ۲: دو کف
        دو کف تقریباً هم‌سطح با کندل صعودی
        """
        if not self.is_uptrend():
            return None
        
        # پیدا کردن دو کف اخیر
        lows = self.df[self.df['swing_low']]['low']
        if len(lows) < 2:
            return None
        
        low1 = lows.iloc[-2]
        low2 = lows.iloc[-1]
        
        # دو کف باید نزدیک باشن (تفاوت < 1%)
        if abs(low1 - low2) / low1 > 0.01:
            return None
        
        # کف دوم باید بالاتر یا مساوی باشه
        if low2 > low1 * 1.005:
            return None
        
        # کندل فعلی باید صعودی باشه
        if not (self.last['close'] > self.last['open'] and self.last['close'] > low2):
            return None
        
        confidence = 75
        reasons = [f'Double Bottom ({low1:.4f} / {low2:.4f})']
        
        if self.last['vol_ratio'] > 1.5:
            confidence += 10
            reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        
        entry = self.last['close']
        sl = low2 - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2.5
        tp2 = entry + risk * 4
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Double Bottom', confidence, reasons, self.df)
    
    def higher_low_pullback(self) -> Optional[SignalResult]:
        """
        🔥 استراتژی ۳: کف بالاتر در روند صعودی
        Pullback به EMA21 در روند صعودی
        """
        if not self.is_uptrend():
            return None
        
        # قیمت باید به EMA21 نزدیک شده باشه
        dist_to_ema21 = abs(self.last['close'] - self.last['ema21']) / self.last['ema21'] * 100
        if dist_to_ema21 > 1.0:
            return None
        
        # کندل صعودی
        if not self.last['close'] > self.last['open']:
            return None
        
        # کف بالاتر
        recent_lows = self.df[self.df['swing_low']]['low']
        if len(recent_lows) < 2:
            return None
        
        if recent_lows.iloc[-1] <= recent_lows.iloc[-2]:
            return None
        
        confidence = 65
        reasons = [f'Higher Low Pullback to EMA21']
        
        if self.last['rsi'] < 55:
            confidence += 10
            reasons.append(f'RSI={self.last["rsi"]:.1f}')
        
        entry = self.last['close']
        sl = self.last['ema21'] - self.last['atr'] * 1.0
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'HL Pullback', confidence, reasons, self.df)
    
    def ema_bounce(self) -> Optional[SignalResult]:
        """
        🔥 استراتژی ۴: برخورد از EMA در روند صعودی
        """
        if not self.is_uptrend():
            return None
        
        # قیمت باید به EMA55 نزدیک شده باشه
        if self.last['close'] > self.last['ema55'] * 1.02:
            return None
        
        if self.last['low'] > self.last['ema55']:
            return None
        
        # برخورد کرده و برگشته
        if not (self.last['close'] > self.last['open'] and self.last['close'] > self.last['ema55']):
            return None
        
        confidence = 70
        reasons = [f'EMA55 Bounce ({self.last["ema55"]:.4f})']
        
        if self.last['vol_ratio'] > 1.2:
            confidence += 10
        
        entry = self.last['close']
        sl = self.last['ema55'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'EMA Bounce', confidence, reasons, self.df)
    
    def analyze(self):
        signals = []
        for strategy in [self.support_bounce, self.double_bottom, self.higher_low_pullback, self.ema_bounce]:
            try:
                sig = strategy()
                if sig:
                    signals.append(sig)
            except Exception as e:
                pass
        
        if not signals:
            return None
        return max(signals, key=lambda x: x.confidence)

# ═════════════════════════════════════════════════════════════════
# فرمت پیام
# ═════════════════════════════════════════════════════════════════

def format_signal(symbol, sig):
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

📈 RSI: {sig.df.iloc[-1]['rsi']:.1f} | ATR: {sig.df.iloc[-1]['atr_pct']:.2f}%
📊 حجم: {sig.df.iloc[-1]['vol_ratio']:.1f}x میانگین
📝 دلایل:
{chr(10).join('  • ' + r for r in sig.reasons)}

⏰ {sig.time.strftime('%Y-%m-%d %H:%M')} UTC

💡 50% TP1 | 30% TP2 | 20% Trailing
⚠️ ریسک خودتونو مدیریت کنید
"""

# ═════════════════════════════════════════════════════════════════
# مدیریت زمان
# ═════════════════════════════════════════════════════════════════

def get_tf_minutes(tf):
    return {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}.get(tf, 15)

def wait_next_candle(tf):
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
    sent = {}
    
    telegram.send_startup()
    logger.info(f"🚀 شروع | {len(SYMBOLS)} جفت‌ارز | {TIMEFRAME}")
    
    while True:
        start = time.time()
        found = 0
        
        for symbol in SYMBOLS:
            try:
                df = fetcher.fetch(symbol, TIMEFRAME)
                if df is None or len(df) < 60:
                    continue
                
                df = Indicators.add_all(df)
                
                # دیباگ
                last = df.iloc[-1]
                support = StrategyEngine(df).find_support(30)
                logger.info(
                    f"🔍 {symbol} | "
                    f"قیمت:{last['close']:.4f} "
                    f"EMA21:{last['ema21']:.4f} "
                    f"EMA55:{last['ema55']:.4f} "
                    f"Support:{support:.4f} "
                    f"RSI:{last['rsi']:.1f} "
                    f"Uptrend:{last['ema21'] > last['ema55']}"
                )
                
                engine = StrategyEngine(df)
                signal = engine.analyze()
                
                if signal:
                    key = f"{symbol}_{signal.time.strftime('%Y%m%d%H%M')}"
                    if key in sent:
                        continue
                    
                    msg = format_signal(symbol, signal)
                    if telegram.send(msg):
                        sent[key] = True
                        found += 1
                        logger.info(f"✅ سیگنال {symbol}: {signal.strategy} ({signal.confidence}%)")
                else:
                    logger.info(f"❌ {symbol}: بدون سیگنال")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
        sent = {k: v for k, v in sent.items() 
                if datetime.strptime(k.split('_')[1], '%Y%m%d%H%M').replace(tzinfo=timezone.utc) > cutoff}
        
        logger.info(f"🏁 دور: {time.time()-start:.1f}s | سیگنال: {found}")
        wait_next_candle(TIMEFRAME)

if __name__ == '__main__':
    main()
