import ccxt
import pandas as pd
import requests
import time
import os
from datetime import datetime

# ═════════════════════════════════════════════════════════════════
# تنظیمات از متغیرهای محیطی (Render Variables)
# ═════════════════════════════════════════════════════════════════

TOKEN = os.environ.get('8871815584:AAEqHlkmVB75GwHbjWKcnLQNG53zMA19Udc', '')
CHAT_ID = os.environ.get('1310655410', '')
SYMBOL = os.environ.get('SYMBOL', 'BTC/USDT')
TIMEFRAME = os.environ.get('TIMEFRAME', '15m')

# ═════════════════════════════════════════════════════════════════
# توابع تلگرام
# ═════════════════════════════════════════════════════════════════

def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, json={
            'chat_id': CHAT_ID,
            'text': text,
            'parse_mode': 'HTML'
        }, timeout=10)
        print(f"📨 ارسال شد")
    except Exception as e:
        print(f"❌ خطا: {e}")

# ═════════════════════════════════════════════════════════════════
# TOOBIT API
# ═════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════
# TOOBIT API KEY (اختیاری)
# ═════════════════════════════════════════════════════════════════

TOOBIT_API_KEY = os.environ.get('fFu7e9T98nmQl9leDejtsTOdKyKjcQyjQUbkuLGhcJSuqSJEB7y0I3Nmq3pe3OWP', '')
TOOBIT_API_SECRET = os.environ.get('TOOBIT_API_SECRET', '')
def get_data_toobit():
    """دریافت داده از Toobit API"""
    try:
        print("🔄 تلاش با Toobit...")
        
        # Toobit API endpoint
        base_url = "https://api.toobit.com"
        
        # تبدیل SYMBOL به فرمت Toobit (BTC/USDT → BTCUSDT)
        symbol_toobit = SYMBOL.replace('/', '')
        
        # تبدیل TIMEFRAME
        interval_map = {
            '1m': '1',
            '5m': '5',
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '4h': '240',
            '1d': 'D'
        }
        interval = interval_map.get(TIMEFRAME, '15')
        
        # Kline/Candlestick data
        endpoint = "/api/v1/market/kline"
        
        params = {
            'symbol': symbol_toobit,
            'interval': interval,
            'limit': 100
        }
        
        # اگه API Key داری، اضافه کن
        headers = {}
        if TOOBIT_API_KEY:
            import hashlib
            import hmac
            
            timestamp = str(int(time.time() * 1000))
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            
            # امضا
            signature = hmac.new(
                TOOBIT_API_SECRET.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers = {
                'X-API-KEY': TOOBIT_API_KEY,
                'X-TIMESTAMP': timestamp,
                'X-SIGNATURE': signature
            }
            print("🔑 Using Toobit API Key")
        
        response = requests.get(f"{base_url}{endpoint}", params=params, headers=headers, timeout=10)
        data = response.json()
        
        if data.get('code') == '0' and data.get('data'):
            klines = data['data']
            
            # تبدیل به DataFrame
            df_data = []
            for k in klines:
                df_data.append({
                    'timestamp': int(k[0]),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            
            df = pd.DataFrame(df_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"✅ متصل به Toobit")
            return df.set_index('timestamp')
        else:
            print(f"❌ Toobit خطا: {data}")
            return None
            
    except Exception as e:
        print(f"❌ Toobit خطا: {e}")
        return None

# ═════════════════════════════════════════════════════════════════
# CCXT EXCHANGES
# ═════════════════════════════════════════════════════════════════

def get_data_ccxt():
    """دریافت داده از صرافی‌های CCXT"""
    exchanges = [
        ccxt.kucoin({'enableRateLimit': True}),
        ccxt.bybit({'enableRateLimit': True}),
        ccxt.binance({'enableRateLimit': True}),
    ]
    
    for ex in exchanges:
        try:
            print(f"🔄 تلاش با {ex.id}...")
            ohlcv = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"✅ متصل به {ex.id}")
                return df.set_index('timestamp')
        except Exception as e:
            print(f"❌ {ex.id} خطا: {e}")
            continue
    
    return None

# ═════════════════════════════════════════════════════════════════
# دریافت داده (همه صرافی‌ها)
# ═════════════════════════════════════════════════════════════════

def get_data():
    """تلاش با همه صرافی‌ها"""
    
    # اول Toobit
    df = get_data_toobit()
    if df is not None:
        return df
    
    # بعد CCXT
    df = get_data_ccxt()
    if df is not None:
        return df
    
    print("❌ هیچ صرافی‌ای کار نکرد")
    return None

# ═════════════════════════════════════════════════════════════════
# اندیکاتورها
# ═════════════════════════════════════════════════════════════════

def calculate(df):
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema55'] = df['close'].ewm(span=55, adjust=False).mean()
    
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    return df

# ═════════════════════════════════════════════════════════════════
# سیگنال
# ═════════════════════════════════════════════════════════════════

def find_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    ema_bull = last['ema8'] > last['ema21'] > last['ema55']
    ema_bear = last['ema8'] < last['ema21'] < last['ema55']
    
    ob_bull = (prev2['close'] < prev2['open'] and 
               prev['close'] > prev['open'] and 
               prev['close'] > prev2['high'])
    
    ob_bear = (prev2['close'] > prev2['open'] and 
               prev['close'] < prev['open'] and 
               prev['close'] < prev2['low'])
    
    recent_low = df['low'].iloc[-10:-2].min()
    recent_high = df['high'].iloc[-10:-2].max()
    
    sweep_bull = (last['low'] < recent_low and 
                  last['close'] > recent_low and 
                  last['close'] > last['open'])
    
    sweep_bear = (last['high'] > recent_high and 
                  last['close'] < recent_high and 
                  last['close'] < last['open'])
    
    high_vol = last['volume'] > last['vol_ma'] * 1.3
    atr_pct = last['atr'] / last['close'] * 100
    atr_ok = atr_pct > 0.25
    rsi = last['rsi']
    
    signal = None
    entry = last['close']
    sl = tp1 = tp2 = tp3 = None
    reasons = []
    
    if (ob_bull or sweep_bull) and ema_bull and high_vol and atr_ok and 30 < rsi < 65:
        signal = 'LONG'
        if ob_bull: reasons.append('Order Block')
        if sweep_bull: reasons.append('Liquidity Sweep')
        
        sl = (prev2['low'] if ob_bull else recent_low) - last['atr'] * 0.5
        risk = entry - sl
        risk = max(risk, last['atr'] * 0.5)
        
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 2.5
        tp3 = entry + risk * 4.0
        
    elif (ob_bear or sweep_bear) and ema_bear and high_vol and atr_ok and 35 < rsi < 70:
        signal = 'SHORT'
        if ob_bear: reasons.append('Order Block')
        if sweep_bear: reasons.append('Liquidity Sweep')
        
        sl = (prev2['high'] if ob_bear else recent_high) + last['atr'] * 0.5
        risk = sl - entry
        risk = max(risk, last['atr'] * 0.5)
        
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 2.5
        tp3 = entry - risk * 4.0
    
    return {
        'signal': signal,
        'entry': entry,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'rsi': rsi,
        'atr': last['atr'],
        'time': last.name,
        'reasons': reasons
    }

# ═════════════════════════════════════════════════════════════════
# فرمت پیام
# ═════════════════════════════════════════════════════════════════

def format_msg(s):
    if not s['signal']:
        return None
    
    emoji = "🟢" if s['signal'] == 'LONG' else "🔴"
    reasons = " + ".join(s['reasons'])
    
    if s['signal'] == 'LONG':
        rr1 = (s['tp1'] - s['entry']) / (s['entry'] - s['sl'])
        rr2 = (s['tp2'] - s['entry']) / (s['entry'] - s['sl'])
        rr3 = (s['tp3'] - s['entry']) / (s['entry'] - s['sl'])
    else:
        rr1 = (s['entry'] - s['tp1']) / (s['sl'] - s['entry'])
        rr2 = (s['entry'] - s['tp2']) / (s['sl'] - s['entry'])
        rr3 = (s['entry'] - s['tp3']) / (s['sl'] - s['entry'])
    
    return f"""
{emoji} <b>سیگنال {s['signal']} - {SYMBOL}</b> {emoji}

⏰ <b>زمان:</b> {s['time'].strftime('%Y-%m-%d %H:%M')}

━━━━━━━━━━━━━━━━━━━━━━
💰 <b>ورود:</b> <code>{s['entry']:.2f}</code> USDT
⛔ <b>استاپ لاس:</b> <code>{s['sl']:.2f}</code> USDT
🎯 <b>تارگت 1:</b> <code>{s['tp1']:.2f}</code> (R:R 1:{rr1:.1f})
🎯 <b>تارگت 2:</b> <code>{s['tp2']:.2f}</code> (R:R 1:{rr2:.1f})
🎯 <b>تارگت 3:</b> <code>{s['tp3']:.2f}</code> (R:R 1:{rr3:.1f})
━━━━━━━━━━━━━━━━━━━━━━

📊 RSI: {s['rsi']:.1f} | ATR: {s['atr']:.2f}
📝 دلایل: {reasons}

💡 50% در TP1 | 30% در TP2 | 20% با Trailing
⚠️ مسئولیت با شماست
"""

# ═════════════════════════════════════════════════════════════════
# اجرا
# ═════════════════════════════════════════════════════════════════

print("🚀 ربات SMC شروع شد!")
print(f"💱 {SYMBOL} | ⏱️ {TIMEFRAME}")
print("💱 صرافی‌ها: Toobit, KuCoin, Bybit, Binance")

send_telegram("🚀 <b>ربات SMC فعال شد!</b>\n"
              f"💱 {SYMBOL} | ⏱️ {TIMEFRAME}\n"
              "💱 صرافی‌ها: Toobit, KuCoin, Bybit, Binance\n"
              "⏳ منتظر سیگنال...")

last_time = None

while True:
    try:
        df = get_data()
        if df is None:
            print("⏳ 60 ثانیه صبر می‌کنم...")
            time.sleep(60)
            continue
        
        df = calculate(df)
        sig = find_signal(df)
        
        if sig['signal'] and sig['time'] != last_time:
            msg = format_msg(sig)
            if msg:
                send_telegram(msg)
                print(f"\n✅ سیگنال {sig['signal']}!")
                print(f"💰 Entry: {sig['entry']:.2f}")
                print(f"⛔ SL: {sig['sl']:.2f}")
                print(f"🎯 TP1: {sig['tp1']:.2f}")
                print(f"🎯 TP2: {sig['tp2']:.2f}")
                print(f"🎯 TP3: {sig['tp3']:.2f}\n")
                last_time = sig['time']
        else:
            print(f"⏳ {datetime.now().strftime('%H:%M:%S')} | "
                  f"قیمت: {sig['entry']:.2f} | "
                  f"RSI: {sig['rsi']:.1f} | "
                  "سیگنالی نیست")
        
        time.sleep(900)
        
    except KeyboardInterrupt:
        send_telegram("👋 ربات متوقف شد")
        print("\n👋 متوقف شد")
        break
    except Exception as e:
        print(f"❌ خطا: {e}")
        time.sleep(60)
