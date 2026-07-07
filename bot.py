import ccxt
import pandas as pd
import numpy as np
import requests
import time
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List

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
        
        # 🔥 جدید: فاصله از EMA8 (برای تشخیص پامپ)
        df['dist_ema8'] = (df['close'] - df['ema8']) / df['ema8'] * 100
        # فاصله از EMA21
        df['dist_ema21'] = (df['close'] - df['ema21']) / df['ema21'] * 100
        
        # 🔥 جدید: ADX برای تشخیص روند قوی
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        tr_smooth = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * plus_dm.where(plus_dm > minus_dm, 0).ewm(span=14).mean() / tr_smooth
        minus_di = 100 * minus_dm.where(minus_dm > plus_dm, 0).ewm(span=14).mean() / tr_smooth
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).fillna(0)
        df['adx'] = dx.ewm(span=14, adjust=False).mean()
        
        return df

# ═════════════════════════════════════════════════════════════════
# استراتژی‌ها — با فیلترهای ضد-استاپ
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
        self.prev3 = df.iloc[-4]
    
    def anti_stop_filters(self) -> tuple[bool, List[str]]:
        """
        🔥 فیلترهای ضد-استاپ
        برمی‌گردونه: (ok, reasons)
        """
        reasons = []
        
        # ۱. قیمت نباید خیلی بالای EMA8 باشه (پامپ نکرده باشه)
        dist_ema8 = self.last['dist_ema8']
        if dist_ema8 > 2.0:
            return False, [f"❌ پامپ شدید ({dist_ema8:.1f}% بالای EMA8)"]
        if dist_ema8 > 1.0:
            reasons.append(f"⚠️ فاصله از EMA8: {dist_ema8:.1f}%")
        
        # ۲. قیمت باید بالای EMA200 باشه (روند صعودی بلندمدت)
        if self.last['close'] < self.last['ema200']:
            return False, ["❌ زیر EMA200"]
        reasons.append("✅ بالای EMA200")
        
        # ۳. ADX باید بالا باشه (روند قوی)
        if self.last['adx'] < 20:
            return False, [f"❌ ADX ضعیف ({self.last['adx']:.1f})"]
        reasons.append(f"✅ ADX قوی ({self.last['adx']:.1f})")
        
        # ۴. آخر ۵ کندل نباید همه صعودی باشن ( exhaustion )
        last_5 = self.df.iloc[-5:]
        green_candles = (last_5['close'] > last_5['open']).sum()
        if green_candles >= 4:
            return False, [f"❌ Exhaustion ({green_candles}/5 صعودی)"]
        
        # ۵. حجم نباید خیلی بالا باشه (FOMO)
        if self.last['vol_ratio'] > 5:
            return False, [f"❌ حجم FOMO ({self.last['vol_ratio']:.1f}x)"]
        
        return True, reasons
    
    def ema_cross(self):
        """EMA Cross با فیلتر Pullback"""
        # کراس باید توی ۳ کندل اخیر باشه
        cross_recent = False
        for i in range(1, 4):
            prev = self.df.iloc[-i-1]
            curr = self.df.iloc[-i]
            if prev['ema8'] <= prev['ema21'] and curr['ema8'] > curr['ema21']:
                cross_recent = True
                break
        
        if not cross_recent:
            return None
        
        # 🔥 Pullback: قیمت باید یه بار به EMA21 نزدیک شده باشه
        pullback = False
        for i in range(1, 10):
            if self.df.iloc[-i]['low'] <= self.df.iloc[-i]['ema21'] * 1.005:
                pullback = True
                break
        
        if not pullback:
            return None
        
        # فیلترهای ضد-استاپ
        ok, filter_reasons = self.anti_stop_filters()
        if not ok:
            logger.info(f"🚫 EMA Cross رد شد: {filter_reasons}")
            return None
        
        rsi_ok = 40 < self.last['rsi'] < 70
        vol_ok = 1.0 < self.last['vol_ratio'] < 3.0
        atr_ok = self.last['atr_pct'] > 0.15
        
        confidence = 60
        reasons = ['EMA Cross + Pullback'] + filter_reasons
        
        if rsi_ok: confidence += 10; reasons.append(f'RSI={self.last["rsi"]:.1f}')
        if vol_ok: confidence += 10; reasons.append(f'Vol={self.last["vol_ratio"]:.1f}x')
        if atr_ok: confidence += 10
        
        entry = self.last['close']
        sl = self.last['ema21'] - self.last['atr'] * 1.0  # SL پایین‌تر = امن‌تر
        risk = entry - sl
        tp1 = entry + risk * 1.5  # TP نزدیک‌تر = احتمال بیشتر
        tp2 = entry + risk * 2.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'EMA Pullback', confidence, reasons, self.df)
    
    def rsi_divergence(self):
        """RSI Divergence واقعی روی کف"""
        lows = self.df['low'].iloc[-30:-1]
        rsi_vals = self.df['rsi'].iloc[-30:-1]
        
        # پیدا کردن دو کف واقعی
        min_idx = lows.rolling(5, center=True).min() == lows
        recent_lows = lows[min_idx]
        recent_rsi = rsi_vals[min_idx]
        
        if len(recent_lows) < 2:
            return None
        
        low1, low2 = recent_lows.iloc[-2], recent_lows.iloc[-1]
        rsi1, rsi2 = recent_rsi.iloc[-2], recent_rsi.iloc[-1]
        
        # واگرایی صعودی: قیمت پایین‌تر، RSI بالاتر
        if not (low2 < low1 * 0.99 and rsi2 > rsi1 * 1.01):
            return None
        
        # 🔥 کف دوم باید بالاتر از EMA200 باشه
        if low2 < self.last['ema200']:
            return None
        
        # فیلترهای ضد-استاپ
        ok, filter_reasons = self.anti_stop_filters()
        if not ok:
            return None
        
        confidence = 70
        reasons = [f'RSI Div ({rsi1:.1f}→{rsi2:.1f})'] + filter_reasons
        
        entry = self.last['close']
        sl = low2 - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'RSI Div', confidence, reasons, self.df)
    
    def breakout(self):
        """Breakout با فیلترهای سخت‌گیرانه"""
        recent_high = self.df['high'].iloc[-20:-5].max()  # ۱۵ کندل قبل نه ۲۰
        
        # شکست باید قوی باشه
        if not (self.last['close'] > recent_high * 1.005 and self.last['close'] > self.last['open']):
            return None
        
        # 🔥 حجم باید خیلی بالا باشه
        if self.last['vol_ratio'] < 2.0:
            return None
        
        # 🔥 باید ۳ کندل قبل consolidation داشته باشه
        range_before = (self.df['high'].iloc[-5:-2].max() - self.df['low'].iloc[-5:-2].min()) / self.last['close'] * 100
        if range_before > 3:
            return None  # قبلش رنج نبوده
        
        # فیلترهای ضد-استاپ
        ok, filter_reasons = self.anti_stop_filters()
        if not ok:
            return None
        
        confidence = 75
        reasons = [f'Breakout {recent_high:.2f}'] + filter_reasons
        reasons.append(f'Consolidation {range_before:.1f}%')
        
        entry = self.last['close']
        sl = recent_high - self.last['atr'] * 1.5  # SL امن‌تر
        risk = entry - sl
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 2.5
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Breakout', confidence, reasons, self.df)
    
    def bollinger_squeeze(self):
        """Bollinger Squeeze — بهتر از Bounce"""
        # باند باید تنگ باشه
        bb_width = (self.last['bb_upper'] - self.last['bb_lower']) / self.last['bb_mid'] * 100
        if bb_width > 5:
            return None
        
        # شکست به سمت بالا
        if not (self.last['close'] > self.last['bb_mid'] and self.last['close'] > self.last['open']):
            return None
        
        # حجم بالا
        if self.last['vol_ratio'] < 1.5:
            return None
        
        # فیلترهای ضد-استاپ
        ok, filter_reasons = self.anti_stop_filters()
        if not ok:
            return None
        
        confidence = 65
        reasons = [f'BB Squeeze ({bb_width:.2f}%)'] + filter_reasons
        
        entry = self.last['close']
        sl = self.last['bb_lower'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = self.last['bb_upper']
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'BB Squeeze', confidence, reasons, self.df)
    
    def analyze(self):
        signals = []
        for strategy in [self.ema_cross, self.rsi_divergence, self.breakout, self.bollinger_squeeze]:
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

📈 RSI: {sig.df.iloc[-1]['rsi']:.1f} | ADX: {sig.df.iloc[-1]['adx']:.1f}
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
                logger.info(
                    f"🔍 {symbol} | "
                    f"قیمت:{last['close']:.2f} "
                    f"EMA8dist:{last['dist_ema8']:.2f}% "
                    f"ADX:{last['adx']:.1f} "
                    f"RSI:{last['rsi']:.1f} "
                    f"Vol:{last['vol_ratio']:.1f}x"
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
