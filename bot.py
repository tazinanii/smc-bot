import ccxt
import pandas as pd
import numpy as np
import requests
import time
import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

# ═════════════════════════════════════════════════════════════════
# تنظیمات
# ═════════════════════════════════════════════════════════════════

TOKEN = os.environ.get('TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')
TIMEFRAME = os.environ.get('TIMEFRAME', '15m')

# ⚠️ MATIC به POL تغییر نام داده
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 
    'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT',
    'LINK/USDT', 'POL/USDT'  # ← تغییر از MATIC
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
            logger.warning("⚠️ توکن یا CHAT_ID تنظیم نشده")
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': parse_mode,
                    'disable_web_page_preview': True
                },
                timeout=15
            )
            if response.status_code == 200:
                logger.info("📨 پیام ارسال شد")
                return True
            else:
                logger.error(f"❌ خطای تلگرام: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ خطا در ارسال تلگرام: {e}")
            return False
    
    def send_startup(self):
        msg = f"""
🚀 <b>ربات تریدینگ فعال شد!</b>

📊 جفت‌ارزها: {len(SYMBOLS)} عدد
⏱️ تایم‌فریم: {TIMEFRAME}
🔍 استراتژی‌ها: EMA Cross + RSI Div + Breakout + BB Bounce

⏳ اسکن شروع شد...
"""
        self.send(msg)

# ═════════════════════════════════════════════════════════════════
# دیتا - فقط KuCoin چون Binance/Bybit از IP Render بلاک شدن
# ═════════════════════════════════════════════════════════════════

class DataFetcher:
    def __init__(self):
        # فقط KuCoin — Binance و Bybit از IP Render بلاک هستن
        self.exchange = ccxt.kucoin({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
    
    def fetch(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 50:
                df = pd.DataFrame(ohlcv, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                logger.info(f"✅ {symbol} از KuCoin ({len(df)} کندل)")
                return df.set_index('timestamp')
        except Exception as e:
            logger.warning(f"⚠️ KuCoin برای {symbol} خطا: {str(e)[:80]}")
        
        logger.error(f"❌ هیچ صرافی‌ای برای {symbol} کار نکرد")
        return None

# ═════════════════════════════════════════════════════════════════
# اندیکاتورها
# ═════════════════════════════════════════════════════════════════

class Indicators:
    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMA
        for period in [8, 21, 55, 200]:
            df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # SMA
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        
        # ATR
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['close'].shift())
        lc = abs(df['low'] - df['close'].shift())
        df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
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
        
        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        return df

# ═════════════════════════════════════════════════════════════════
# استراتژی‌ها
# ═════════════════════════════════════════════════════════════════

class SignalResult:
    def __init__(self, signal: str, entry: float, sl: float, tp1: float, tp2: float,
                 strategy: str, confidence: int, reasons: List[str], df: pd.DataFrame):
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
    
    def calculate_rr(self, target: float) -> float:
        if self.signal == 'LONG':
            risk = self.entry - self.sl
            reward = target - self.entry
        else:
            risk = self.sl - self.entry
            reward = self.entry - target
        return round(reward / risk, 2) if risk > 0 else 0

class StrategyEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.last = df.iloc[-1]
        self.prev = df.iloc[-2]
        self.prev2 = df.iloc[-3]
    
    # ─── استراتژی ۱: EMA Crossover ─────────────────────────────
    def ema_cross(self) -> Optional[SignalResult]:
        prev_cross = self.prev['ema8'] <= self.prev['ema21']
        now_cross = self.last['ema8'] > self.last['ema21']
        
        if not (prev_cross and now_cross):
            return None
        
        ema_trend = self.last['ema21'] > self.last['ema55']
        rsi_ok = 40 < self.last['rsi'] < 70
        vol_ok = self.last['vol_ratio'] > 1.1
        atr_ok = self.last['atr_pct'] > 0.2
        
        confidence = 50
        reasons = ['EMA8 کراس EMA21']
        
        if ema_trend: 
            confidence += 15
            reasons.append('EMA21 > EMA55')
        if rsi_ok: 
            confidence += 10
            reasons.append(f'RSI={self.last["rsi"]:.1f}')
        if vol_ok: 
            confidence += 10
            reasons.append(f'حجم {self.last["vol_ratio"]:.1f}x')
        if atr_ok: 
            confidence += 10
        
        if confidence < 65:
            return None
        
        entry = self.last['close']
        sl = min(self.last['ema21'], self.last['low']) - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'EMA Cross', confidence, reasons, self.df)
    
    # ─── استراتژی ۲: RSI Divergence ────────────────────────────
    def rsi_divergence(self) -> Optional[SignalResult]:
        lows = self.df['low'].iloc[-20:-1]
        rsi_vals = self.df['rsi'].iloc[-20:-1]
        
        recent_lows = lows[lows == lows.rolling(5, center=True).min()]
        recent_rsi = rsi_vals[lows == lows.rolling(5, center=True).min()]
        
        if len(recent_lows) < 2:
            return None
        
        low1, low2 = recent_lows.iloc[-2], recent_lows.iloc[-1]
        rsi1, rsi2 = recent_rsi.iloc[-2], recent_rsi.iloc[-1]
        
        if not (low2 < low1 and rsi2 > rsi1):
            return None
        
        ema_ok = self.last['close'] > self.last['ema55']
        vol_ok = self.last['vol_ratio'] > 1.0
        
        confidence = 60
        reasons = [f'RSI Divergence ({rsi1:.1f}→{rsi2:.1f})']
        
        if ema_ok: 
            confidence += 15
            reasons.append('قیمت بالای EMA55')
        if vol_ok: 
            confidence += 10
        
        entry = self.last['close']
        sl = low2 - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 4
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'RSI Div', confidence, reasons, self.df)
    
    # ─── استراتژی ۳: Breakout ─────────────────────────────────
    def breakout(self) -> Optional[SignalResult]:
        recent_high = self.df['high'].iloc[-20:-2].max()
        recent_low = self.df['low'].iloc[-20:-2].min()
        
        if self.last['close'] > recent_high and self.last['close'] > self.last['open']:
            vol_ok = self.last['vol_ratio'] > 1.3
            ema_ok = self.last['close'] > self.last['ema55']
            
            confidence = 55
            reasons = [f'Breakout بالای {recent_high:.2f}']
            
            if vol_ok: 
                confidence += 20
                reasons.append(f'حجم قوی {self.last["vol_ratio"]:.1f}x')
            if ema_ok: 
                confidence += 15
            
            if confidence < 70:
                return None
            
            entry = self.last['close']
            sl = recent_high - self.last['atr'] * 1.0
            risk = entry - sl
            tp1 = entry + risk * 2
            tp2 = entry + risk * 3
            
            return SignalResult('LONG', entry, sl, tp1, tp2, 'Breakout', confidence, reasons, self.df)
        
        return None
    
    # ─── استراتژی ۴: Bollinger Bounce ──────────────────────────
    def bollinger_bounce(self) -> Optional[SignalResult]:
        touched_lower = (self.df['low'].iloc[-5:-1] <= self.df['bb_lower'].iloc[-5:-1]).any()
        now_bounce = self.last['close'] > self.last['bb_lower'] and self.last['close'] > self.last['open']
        
        if not (touched_lower and now_bounce):
            return None
        
        ema_ok = self.last['ema8'] > self.last['ema21']
        rsi_ok = self.last['rsi'] > 35
        
        confidence = 55
        reasons = ['Bollinger Bounce']
        
        if ema_ok: 
            confidence += 15
        if rsi_ok: 
            confidence += 10
        
        if confidence < 65:
            return None
        
        entry = self.last['close']
        sl = self.last['bb_lower'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = self.last['bb_mid']
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'BB Bounce', confidence, reasons, self.df)
    
    def analyze(self) -> Optional[SignalResult]:
        signals = []
        
        for strategy in [self.ema_cross, self.rsi_divergence, self.breakout, self.bollinger_bounce]:
            try:
                sig = strategy()
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"خطا در {strategy.__name__}: {e}")
        
        if not signals:
            return None
        
        best = max(signals, key=lambda x: x.confidence)
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
🎯 <b>TP1:</b> <code>{sig.tp1:.4f}</code> (R:R 1:{sig.calculate_rr(sig.tp1)})
🎯 <b>TP2:</b> <code>{sig.tp2:.4f}</code> (R:R 1:{sig.calculate_rr(sig.tp2)})
━━━━━━━━━━━━━━━━━━━━━━

📈 RSI: {sig.df.iloc[-1]['rsi']:.1f} | ATR: {sig.df.iloc[-1]['atr_pct']:.2f}%
📊 حجم: {sig.df.iloc[-1]['vol_ratio']:.1f}x میانگین
📝 دلایل: {', '.join(sig.reasons)}

⏰ {sig.time.strftime('%Y-%m-%d %H:%M')} UTC

💡 50% TP1 | 30% TP2 | 20% Trailing
⚠️ ریسک خودتونو مدیریت کنید
"""

# ═════════════════════════════════════════════════════════════════
# مدیریت زمان
# ═════════════════════════════════════════════════════════════════

def get_timeframe_minutes(tf: str) -> int:
    mapping = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
    return mapping.get(tf, 15)

def wait_for_next_candle(tf: str):
    minutes = get_timeframe_minutes(tf)
    now = datetime.now(timezone.utc)  # ← اصلاح utcnow()
    
    next_minute = ((now.minute // minutes) + 1) * minutes
    if next_minute >= 60:
        next_candle = (now + timedelta(hours=1)).replace(
            minute=next_minute % 60, second=5, microsecond=0
        )
    else:
        next_candle = now.replace(minute=next_minute, second=5, microsecond=0)
    
    wait = (next_candle - now).total_seconds()
    if wait > 0:
        logger.info(f"⏳ صبر {wait:.0f} ثانیه تا کندل بعدی...")
        time.sleep(wait)

# ═════════════════════════════════════════════════════════════════
# اجرا
# ═════════════════════════════════════════════════════════════════

def main():
    telegram = TelegramNotifier(TOKEN, CHAT_ID)
    fetcher = DataFetcher()
    sent_signals = {}
    
    telegram.send_startup()
    logger.info(f"🚀 ربات شروع شد | {len(SYMBOLS)} جفت‌ارز | تایم‌فریم {TIMEFRAME}")
    
    while True:
        cycle_start = time.time()
        found_signals = 0
        
        for symbol in SYMBOLS:
            try:
                df = fetcher.fetch(symbol, TIMEFRAME)
                if df is None or len(df) < 60:
                    continue
                
                df = Indicators.add_all(df)
                
                # 🔍 دیباگ: چاپ وضعیت فعلی
                last = df.iloc[-1]
                logger.info(f"🔍 {symbol} | EMA8:{last['ema8']:.1f} EMA21:{last['ema21']:.1f} "
                           f"RSI:{last['rsi']:.1f} VOL:{last['vol_ratio']:.1f}x")
                
                engine = StrategyEngine(df)
                signal = engine.analyze()
                
                if signal:
                    key = f"{symbol}_{signal.time.strftime('%Y%m%d%H')}"
                    if key in sent_signals:
                        continue
                    
                    msg = format_signal(symbol, signal)
                    if telegram.send(msg):
                        sent_signals[key] = True
                        found_signals += 1
                        logger.info(f"✅ سیگنال {symbol}: {signal.strategy} ({signal.confidence}%)")
                else:
                    logger.info(f"❌ {symbol}: هیچ سیگنالی")
                
            except Exception as e:
                logger.error(f"❌ خطا در {symbol}: {e}")
                continue
        
        # پاک کردن سیگنال‌های قدیمی
        cutoff = datetime.now(timezone.utc) - timedelta(hours=12)  # ← اصلاح utcnow()
        sent_signals = {k: v for k, v in sent_signals.items() 
                       if datetime.strptime(k.split('_')[1], '%Y%m%d%H').replace(tzinfo=timezone.utc) > cutoff}
        
        cycle_time = time.time() - cycle_start
        logger.info(f"🏁 دور کامل: {cycle_time:.1f}s | سیگنال: {found_signals}")
        
        wait_for_next_candle(TIMEFRAME)

if __name__ == '__main__':
    main()
