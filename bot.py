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
        
        # 🔥 جدید: فاصله از سقف اخیر
        df['dist_from_high'] = (df['close'] - df['high'].rolling(50).max()) / df['high'].rolling(50).max() * 100
        
        return df

# ═════════════════════════════════════════════════════════════════
# استراتژی: فقط "Buy the Dip"
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
    
    def buy_the_dip(self) -> Optional[SignalResult]:
        """
        🔥 استراتژی نهایی: Buy the Dip
        فقط وقتی قیمت افتاده و oversold شده
        """
        
        # ۱. قیمت باید حداقل ۳% از سقف ۵۰ کندل اخیر افتاده باشه
        recent_high = self.df['high'].iloc[-50:].max()
        dip_pct = (recent_high - self.last['close']) / recent_high * 100
        
        if dip_pct < 3.0:
            logger.info(f"🚫 {self.last.name}: Dip کافی نیست ({dip_pct:.1f}%)")
            return None
        
        # ۲. RSI باید < ۴۰ باشه (oversold)
        if self.last['rsi'] > 40:
            logger.info(f"🚫 {self.last.name}: RSI بالاست ({self.last['rsi']:.1f})")
            return None
        
        # ۳. قیمت باید بالای EMA200 باشه (روند صعودی بلندمدت)
        if self.last['close'] < self.last['ema200']:
            logger.info(f"🚫 {self.last.name}: زیر EMA200")
            return None
        
        # ۴. کندل فعلی باید صعودی باشه (برگشت)
        if not self.last['close'] > self.last['open']:
            logger.info(f"🚫 {self.last.name}: کندل نزولی")
            return None
        
        # ۵. حجم باید بالا باشه (پانیک سل)
        if self.last['vol_ratio'] < 1.5:
            logger.info(f"🚫 {self.last.name}: حجم کم ({self.last['vol_ratio']:.1f}x)")
            return None
        
        # ۶. کف کندل فعلی باید از کف قبلی بالاتر باشه (Higher Low)
        if self.last['low'] <= self.prev['low']:
            logger.info(f"🚫 {self.last.name}: Higher Low نیست")
            return None
        
        # ✅ همه شرایط برقرار
        confidence = 70
        reasons = [
            f'Dip {dip_pct:.1f}% از سقف',
            f'RSI={self.last["rsi"]:.1f} (oversold)',
            f'Vol={self.last["vol_ratio"]:.1f}x',
            'Higher Low',
            'کندل صعودی'
        ]
        
        entry = self.last['close']
        sl = self.last['low'] - self.last['atr'] * 0.5
        risk = entry - sl
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3
        
        return SignalResult('LONG', entry, sl, tp1, tp2, 'Buy the Dip', confidence, reasons, self.df)
    
    def analyze(self):
        return self.buy_the_dip()

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
                recent_high = df['high'].iloc[-50:].max()
                dip_pct = (recent_high - last['close']) / recent_high * 100
                logger.info(
                    f"🔍 {symbol} | "
                    f"قیمت:{last['close']:.4f} | "
                    f"سقف:{recent_high:.4f} | "
                    f"Dip:{dip_pct:.1f}% | "
                    f"RSI:{last['rsi']:.1f} | "
                    f"Vol:{last['vol_ratio']:.1f}x | "
                    f"EMA200:{last['ema200']:.4f}"
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
