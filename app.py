from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import logging
import time
import requests
import re 
from datetime import datetime
from typing import List, Dict
from functools import wraps
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import feedparser

# ==========================================
# 🔧 1. 系統配置
# ==========================================
@dataclass
class Config:
    DEFAULT_EXCHANGE_RATE: float = 32.5
    CACHE_TTL: int = 60 
    
    COIN_META = {
        'BTC': {'cn_name': '比特幣'}, 'ETH': {'cn_name': '以太幣'}, 'BNB': {'cn_name': '幣安幣'},
        'SOL': {'cn_name': '索拉納'}, 'XRP': {'cn_name': '瑞波幣'}, 'DOGE': {'cn_name': '狗狗幣'},
        'ADA': {'cn_name': '艾達幣'}, 'TRX': {'cn_name': '波場幣'}, 'AVAX': {'cn_name': '雪崩幣'},
        'DOT': {'cn_name': '波卡幣'}, 'LINK': {'cn_name': '連鎖幣'}, 'LTC': {'cn_name': '萊特幣'},
        'USDC': {'cn_name': 'USD Coin', 'is_stable': True},
        'FDUSD': {'cn_name': 'FDUSD', 'is_stable': True},
        'USDT': {'cn_name': '泰達幣', 'is_stable': True}
    }
    STABLE_COINS = {'USDC', 'FDUSD', 'USDT', 'DAI', 'TUSD', 'USDE'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
np.seterr(divide='ignore', invalid='ignore')

# ==========================================
# 🧬 2. 快取裝飾器
# ==========================================
def ttl_cache(ttl_seconds: int):
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = str(args) + str(kwargs)
            if key in cache and 'timestamp' in cache[key]:
                if now - cache[key]['timestamp'] < ttl_seconds:
                    return cache[key]['data']
            result = func(*args, **kwargs)
            cache[key] = {'data': result, 'timestamp': now}
            return result
        return wrapper
    return decorator

# ==========================================
# 🌐 3. 資料管理 (直接抓取原始 API，不佔用連線池)
# ==========================================
class DataManager:

    @staticmethod
    def get_all_tickers() -> List[Dict]:
        try:
            # 🚀 直接要資料，不透過官方套件，絕不塞車
            res = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=5)
            tickers = res.json()
            valid_tickers = []
            for t in tickers:
                symbol = t['symbol']
                if symbol.endswith('USDT') and 'UP' not in symbol and 'DOWN' not in symbol:
                    quote_vol = float(t['quoteVolume'])
                    if quote_vol > 500000: 
                        valid_tickers.append({
                            "symbol": symbol.replace('USDT', ''),
                            "price_usd": float(t['lastPrice']),
                            "change": float(t['priceChangePercent']),
                            "vol": quote_vol
                        })
            valid_tickers.sort(key=lambda x: x['vol'], reverse=True)
            
            final_list = []
            for idx, item in enumerate(valid_tickers):
                sym = item['symbol']
                meta = Config.COIN_META.get(sym, {})
                item['rank'] = idx + 1
                item['name'] = sym
                item['cn_name'] = meta.get('cn_name', sym)
                item['is_stable'] = meta.get('is_stable', sym in Config.STABLE_COINS)
                item['risk'] = {} 
                final_list.append(item)
            return final_list
        except Exception as e: 
            print(f"⚠️ 幣安大盤 API 抓取失敗: {e}")
            return []

    @staticmethod
    def get_kline_safe(symbol: str):
        try:
            # 🚀 自己拼網址抓 K 線，抓完就跑！
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=120"
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                data = res.json()
                return symbol, [float(k[4]) for k in data]
            return symbol, []
        except Exception as e:
            print(f"⚠️ K線抓取失敗 {symbol}: {e}")
            return symbol, []

    @staticmethod
    @ttl_cache(ttl_seconds=300) 
    def get_historical_df_parallel(top_symbols: List[str]) -> pd.DataFrame:
        data_dict = {}
        # 🔥 降壓秘訣：只抓前 10 名的熱門幣，保護免費主機
        target_pairs = [s + 'USDT' for s in top_symbols[:10]]
        if 'BTCUSDT' not in target_pairs:
            target_pairs.append('BTCUSDT')
            
        # 🔥 降壓秘訣：乖乖排隊抓資料
        for pair in target_pairs:
            try:
                symbol, prices = DataManager.get_kline_safe(pair)
                if prices: 
                    data_dict[symbol.replace('USDT', '')] = prices
                time.sleep(0.1) 
            except: 
                pass

        if 'BTC' not in data_dict: return pd.DataFrame()
        min_len = min([len(v) for v in data_dict.values()])
        final_dict = {k: v[-min_len:] for k, v in data_dict.items()}
        return pd.DataFrame(final_dict)

    @staticmethod
    def get_realtime_exchange_rate() -> float:
        try: return float(requests.get("https://tw.rter.info/capi.php", timeout=1).json()["USDTWD"]["Exrate"])
        except: return Config.DEFAULT_EXCHANGE_RATE

# ==========================================
# 🤖 4. AI 翻譯官 (AIAssistant)
# ==========================================
class AIAssistant:
    @staticmethod
    def generate_sfi_insight(score: int) -> str:
        if score >= 65:
            return "🔴 <b>AI 警告：溫室裡的花朵！</b><br>它對大盤的抵抗力極差。只要老大哥（比特幣）稍微跌倒，它絕對會跟著大跳水！若市場氣氛不佳，千萬別買來避險。"
        elif score >= 40:
            return "🟡 <b>AI 判斷：正常的跟屁蟲。</b><br>表現中規中矩，大盤漲它就漲，大盤跌它就跌，沒有特別突出的避險防禦能力。"
        else:
            return "🟢 <b>AI 提示：獨立的孤狼！</b><br>它有自己的護城河，走勢跟大盤脫鉤。當市場恐慌大家逃命時，資金往往會躲進這種幣裡面避風頭。"

    @staticmethod
    def generate_copula_insight(corr: float, lambda_lower: float) -> str:
        insight = ""
        if corr >= 0.6: insight += "📈 <b>【連體嬰】</b>走勢幾乎一模一樣，買它等於買比特幣，無法分散風險。<br>"
        elif corr <= 0.3: insight += "☁️ <b>【各自安好】</b>漲跌不看比特幣臉色，走勢隨機，適合用來分散資產風險。<br>"
        else: insight += "🤝 <b>【普通朋友】</b>平常會跟隨大盤波動，但偶爾會走自己的路。<br>"

        if lambda_lower >= 0.3: insight += "⚠️ <b>嚴重警告：</b>圖表左下角顯示，過去股災時它都是「手牽手一起跳崖」，絕對會被拖下水。"
        elif lambda_lower <= 0.1: insight += "🛡️ <b>防禦屬性：</b>過去大崩盤時，它奇蹟似地沒有跟著摔下去，具備抗跌力。"
        return insight

    @staticmethod
    def generate_mc_insight(current_price: float, mean_path_end: float, volatility: float) -> str:
        if mean_path_end > current_price * 1.02: trend = "向上翹，代表大趨勢看漲 🚀"
        elif mean_path_end < current_price * 0.98: trend = "往下垂，代表趨勢看跌 📉"
        else: trend = "平緩，近期將處於盤整階段 ⚖️"
            
        if volatility > 3.0: vol_insight = "藍線散得極開，未來「不確定性極大」，可能暴賺也可能腰斬，心臟要大顆！ 🎢"
        elif volatility < 1.0: vol_insight = "藍線緊緊擠在一起，代表未來幾天價格安定，適合想安穩睡覺的投資人。 🛌"
        else: vol_insight = "藍線發散程度正常，屬於一般市場波動風險。"
        return f"➖ <b>主力預測：</b>AI 綜合 100 種平行宇宙，預測黃線{trend}<br>📢 <b>風險判讀：</b>{vol_insight}"

# ==========================================
# 🔮 5. 核心演算法 (Risk, MC)
# ==========================================
class MonteCarloEngine:
    @staticmethod
    def simulate_price_paths(prices: List[float], days: int = 7, simulations: int = 100) -> Dict:
        try:
            if len(prices) < 10: return {}
            log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
            drift = log_returns.mean() - (0.5 * log_returns.var())
            stdev = log_returns.std()
            if np.isnan(stdev) or stdev == 0: return {}
            
            simulation_data = []
            last_price = prices[-1]
            for _ in range(simulations):
                prices_path = [last_price]
                for _ in range(days):
                    shock = drift + stdev * np.random.normal()
                    prices_path.append(prices_path[-1] * np.exp(shock))
                simulation_data.append(prices_path)
            
            final_prices = [p[-1] for p in simulation_data]
            return {"paths": simulation_data, "mean_path": np.mean(simulation_data, axis=0).tolist(), "var_95": np.percentile(final_prices, 5), "current_price": last_price, "volatility": stdev * 100}
        except: return {}

class RiskModel:
    @staticmethod
    def calculate_copula_risk(symbol: str, df: pd.DataFrame, is_stable: bool, current_price: float) -> Dict:
        try:
            if is_stable: return {"level": "safe", "msg": "穩定資產", "corr": 0.01, "score": 1, "lambda": 0, "beta": 0, "stress": {"s10": current_price}}
            if symbol not in df.columns or 'BTC' not in df.columns: return {"level": "base", "msg": "資料不足", "corr": 0, "score": 0, "lambda": 0, "beta": 0, "stress": {}}
            
            target_df = df[['BTC', symbol]].dropna()
            returns = target_df.pct_change().dropna()
            if len(target_df) < 10 or returns['BTC'].std() == 0: return {"level": "base", "msg": "資料不足", "corr": 0, "score": 0, "lambda": 0, "beta": 0, "stress": {}}

            # 🔥 移除 spearman，改用內建安全公式防崩潰
            corr = returns['BTC'].corr(returns[symbol])
            
            u, v = returns['BTC'].rank(pct=True), returns[symbol].rank(pct=True)
            crash_together = np.sum((u <= 0.2) & (v <= 0.2))
            crash_btc = np.sum(u <= 0.2)
            lambda_lower = 0 if crash_btc == 0 else crash_together / crash_btc
            
            btc_threshold = returns['BTC'].quantile(0.1) 
            tail_indices = returns['BTC'] <= btc_threshold
            if tail_indices.sum() > 0:
                avg_crash_coin = returns[symbol][tail_indices].mean()
                avg_crash_btc = returns['BTC'][tail_indices].mean()
                tail_beta = avg_crash_coin / avg_crash_btc if avg_crash_btc != 0 else 1.0
            else: tail_beta = 1.0
            
            tail_beta = float(np.clip(tail_beta, -2.0, 5.0))
            beta_factor = min(2.0, max(0.5, tail_beta))
            norm_beta = (beta_factor - 0.5) / 1.5
            raw_score = (lambda_lower * 0.5 + (corr if corr>0 else 0)*0.2 + norm_beta * 0.3) * 100
            sfi_score = int(np.clip(raw_score, 0, 100))
            predicted_price = max(0, current_price * (1 + (-0.10 * tail_beta)))
            
            if sfi_score >= 65: level, msg = "danger", "🔥 高度脆弱"
            elif sfi_score >= 40: level, msg = "warning", "⚠️ 中度連動"
            else: level, msg = "safe", "🟢 走勢獨立"
            return {"level": level, "msg": msg, "corr": round(corr, 2), "lambda": round(lambda_lower, 2), "beta": round(tail_beta, 2), "score": sfi_score, "stress": {"s10": predicted_price}}
        except Exception as e: 
            print(f"⚠️ {symbol} 運算錯誤: {e}")
            return {"level": "base", "msg": "運算錯誤", "corr": 0, "score": 0, "lambda": 0, "beta": 0, "stress": {}}

class NewsEngine:
    @staticmethod
    def generate_sentiment_news(crypto_data: List[Dict]) -> List[Dict]:
        news_feed = []
        top_movers = sorted([c for c in crypto_data if c.get('change') is not None], key=lambda x: abs(x['change']), reverse=True)[:3]
        for c in top_movers:
            t, s = (f"【AI看多】{c['symbol']} 突破阻力", "positive") if c['change']>5 else (f"【風險】{c['symbol']} 賣壓湧現", "negative") if c['change']<-5 else (f"【觀察】{c['symbol']} 盤整中", "neutral")
            news_feed.append({"time": datetime.now().strftime("%H:%M"), "title": t, "sentiment": s})
        return news_feed

# ==========================================
# 📡 6. 社群媒體引擎 (V8: 全網監控)
# ==========================================
class SocialMediaEngine:
    FATAL_NOISE_KEYWORDS = ["閒聊", "好爽", "畢業", "塊陶", "公園", "薯條", "便當", "信仰", "崩盤", "丸子", "蒸的", "睡飽", "財富自由", "睏霸", "韭菜", "舒服", "下去", "這波", "笑死", "甚至", "乾爹", "崩", "噴", "接刀", "水桶", "公告", "版規", "協尋", "詐騙", "入群", "群組", "怎麼看", "大家", "覺得", "是否", "請問", "請益", "新手", "小白", "這隻", "推薦", "？", "?"]
    PREMIUM_KEYWORDS = ["貝萊德", "BlackRock", "富達", "Fidelity", "微策略", "MicroStrategy", "灰度", "Grayscale", "SEC", "聯準會", "Fed", "鮑爾", "Powell", "非農", "CPI", "PCE", "利率", "會議紀要", "幣安", "Coinbase", "Vitalik", "中本聰", "川普", "馬斯克", "Musk"]
    SIGNAL_KEYWORDS = ["ETF", "升息", "降息", "通膨", "監管", "支撐", "壓力", "均線", "鯨魚", "鏈上", "TVL", "質押", "空投", "白皮書", "核准", "通過", "上市", "減半", "現貨", "合約", "回購", "增持", "銷毀", "新高", "Approval", "Surge", "Bull", "Rally"]
    STRONG_HEADERS = ["[新聞]", "[情報]", "[翻譯]", "[數據]", "[分析]"]
    SOFT_NOISE_KEYWORDS = ["請問", "請益", "新手", "小白", "覺得", "是不是", "有沒有", "大家", "怎麼", "推薦", "感覺", "夢到"]

    @staticmethod
    def get_content_summary(url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Cookie": "over18=1"}
            res = requests.get(url, headers=headers, timeout=1.5)
            if res.status_code != 200: return ""
            soup = BeautifulSoup(res.text, "html.parser")
            main_content = soup.find(id="main-content")
            if main_content:
                for tag in main_content.find_all(["div", "span"], class_=["article-metaline", "article-metaline-right", "push"]): tag.extract()
                return re.sub(r'\s+', ' ', main_content.get_text().strip())[:80] + "..."
            return ""
        except: return ""

    @staticmethod
    def process_single_ptt_post(div):
        try:
            title_div = div.find("div", class_="title")
            if not title_div or not title_div.a: return None
            title = title_div.a.text.strip()
            link = "https://www.ptt.cc" + title_div.a["href"]
            
            summary = ""
            if not any(kw in title for kw in SocialMediaEngine.FATAL_NOISE_KEYWORDS):
                summary = SocialMediaEngine.get_content_summary(link)

            nrec = div.find("div", class_="nrec").text
            push_count = 0
            if nrec:
                if nrec == "爆": push_count = 100
                elif nrec.startswith("X"): push_count = 0
                else: 
                    try: push_count = int(nrec)
                    except: push_count = 0

            return {
                "source": "PTT", "title": title, "author": div.find("div", class_="author").text,
                "date": div.find("div", class_="date").text, "push": push_count,
                "link": link, "content": summary if summary else title 
            }
        except: return None

    @staticmethod
    def scrape_ptt() -> List[Dict]:
        url = "https://www.ptt.cc/bbs/DigiCurrency/index.html"
        headers = {"User-Agent": "Mozilla/5.0", "Cookie": "over18=1"}
        try:
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            divs = soup.find_all("div", class_="r-ent")[:20] 
            results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(SocialMediaEngine.process_single_ptt_post, div) for div in divs]
                for future in as_completed(futures):
                    res = future.result()
                    if res: results.append(res)
            return results
        except: return []

    @staticmethod
    def scrape_cnyes() -> List[Dict]:
        mock_news = [{"source": "CNYES", "title": "比特幣現貨ETF淨流入創單日新高", "author": "鉅亨網", "date": datetime.now().strftime("%m/%d"), "push": 99, "link": "#", "content": "市場數據顯示..."}]
        url = "https://news.cnyes.com/news/cat/bc" 
        headers = {"User-Agent": "Mozilla/5.0"}
        posts = []
        try:
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code != 200: return mock_news
            soup = BeautifulSoup(res.text, "html.parser")
            links = soup.find_all("a", href=True)
            count = 0
            for link in links:
                title = link.get_text().strip()
                href = link['href']
                if "/news/id/" in href and len(title) > 15:
                    posts.append({
                        "source": "CNYES", "title": title, "author": "鉅亨網",
                        "date": datetime.now().strftime("%m/%d"), "push": 80,
                        "link": "https://news.cnyes.com" + href if not href.startswith("http") else href,
                        "content": "鉅亨網區塊鏈新聞快訊"
                    })
                    count += 1
                    if count >= 6: break
            return posts if posts else mock_news
        except: return mock_news

    @staticmethod
    def scrape_blocktempo() -> List[Dict]:
        mock_news = [{"source": "BlockTempo", "title": "以太坊升級倒數，Layer 2 幣種噴發", "author": "動區", "date": datetime.now().strftime("%m/%d"), "push": 95, "link": "#", "content": "開發者確認進度..."}]
        url = "https://www.blocktempo.com/category/cryptocurrency-market/"
        headers = {"User-Agent": "Mozilla/5.0"}
        posts = []
        try:
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code != 200: return mock_news
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.find_all("h3", class_="jeg_post_title")
            for art in articles[:6]:
                a_tag = art.find("a")
                if a_tag:
                    posts.append({
                        "source": "BlockTempo", "title": a_tag.get_text().strip(), "author": "動區",
                        "date": datetime.now().strftime("%m/%d"), "push": 90,
                        "link": a_tag['href'], "content": "動區深度報導"
                    })
            return posts if posts else mock_news
        except: return mock_news

    @staticmethod
    def scrape_coindesk() -> List[Dict]:
        mock_news = [{"source": "CoinDesk", "title": "Bitcoin ETF Sees Record Inflows", "author": "CoinDesk", "date": datetime.now().strftime("%m/%d"), "push": 99, "link": "#", "content": "Global markets react..."}]
        posts = []
        try:
            feed = feedparser.parse('https://www.coindesk.com/arc/outboundfeeds/rss/')
            for entry in feed.entries[:6]:
                posts.append({
                    "source": "CoinDesk", "title": entry.title, "author": "CoinDesk",
                    "date": datetime.now().strftime("%m/%d"), "push": 85,
                    "link": entry.link, "content": entry.summary[:80] + "..." if 'summary' in entry else "CoinDesk Global News"
                })
            return posts if posts else mock_news
        except: return mock_news

    @staticmethod
    def calc_quality_score(post: Dict) -> int:
        if post.get('source') in ['CNYES', 'BlockTempo', 'CoinDesk']: return 80 + (len(post['title']) % 15)
        score = 0
        title = post['title']
        content = post['content']
        full_text = title + " " + content
        if "?" in title or "？" in title: return -500
        if any(kw in title for kw in SocialMediaEngine.FATAL_NOISE_KEYWORDS): return -100
        for kw in SocialMediaEngine.SOFT_NOISE_KEYWORDS:
            if kw in title: score -= 15
        if any(header in title for header in SocialMediaEngine.STRONG_HEADERS): score += 40
        for kw in SocialMediaEngine.SIGNAL_KEYWORDS:
            if kw in full_text: score += 10
        if re.search(r'\d+(\.\d+)?%', title): score += 15
        if post['push'] > 20: score += 10
        if len(content) > 50: score += 5
        return score

    @staticmethod
    def analyze_posts(posts: List[Dict]) -> Dict:
        signals, noises = [], []
        sentiment_score = 0
        keyword_counts = {"BTC": 0, "ETH": 0, "SOL": 0, "BNB": 0, "AI": 0, "ETF": 0}
        
        scored_posts = []
        for post in posts:
            title = post['title']
            for key in keyword_counts.keys():
                if key in title.upper(): keyword_counts[key] += 1
            post['quality_score'] = SocialMediaEngine.calc_quality_score(post)
            scored_posts.append(post)

        high_quality_count = sum(1 for p in scored_posts if p['quality_score'] >= 25)
        THRESHOLD = 25 if high_quality_count >= 3 else 0 
        
        for post in scored_posts:
            is_media = post['source'] in ['CNYES', 'BlockTempo', 'CoinDesk']
            if is_media or post['quality_score'] >= THRESHOLD:
                post['type'] = 'signal'
                full_text = (post['title'] + " " + post['content']).lower()
                bullish_kws = ["新高", "突破", "買入", "大漲", "流入", "看多", "surge", "rally", "bull", "inflow", "record", "gain"]
                bearish_kws = ["跌", "崩", "利空", "賣出", "流出", "看空", "plummet", "crash", "hack", "bear", "outflow", "drop", "loss"]
                
                if any(kw in full_text for kw in bullish_kws):
                    post['sentiment'] = "BULLISH"
                    post['ai_summary'] = "AI 識別為高價值利多訊號，建議關注。"
                    sentiment_score += 5
                elif any(kw in full_text for kw in bearish_kws):
                    post['sentiment'] = "BEARISH"
                    post['ai_summary'] = "AI 識別為潛在風險，建議避險。"
                    sentiment_score -= 5
                else:
                    post['sentiment'] = "NEUTRAL"
                    post['ai_summary'] = f"市場關鍵情報：{post['content'][:20]}..."
                
                signals.append(post)
            else:
                post['type'] = 'noise'
                post['sentiment'] = "NEUTRAL"
                post['ai_summary'] = "低資訊密度"
                noises.append(post)

        signals.sort(key=lambda x: x['quality_score'], reverse=True)
        final_sentiment = max(-100, min(100, sentiment_score * 10))
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        
        sentiment_reason = ""
        top_kws = [k for k, v in sorted_keywords if v > 0][:3]
        kw_str = "、".join(top_kws) if top_kws else "總經數據"

        if final_sentiment <= -20: sentiment_reason = f"📉 恐慌主因：市場對 {kw_str} 相關的避險情緒升溫，導致整體信心下滑。"
        elif final_sentiment >= 20: sentiment_reason = f"🚀 貪婪主因：社群對 {kw_str} 討論熱烈，判斷有資金流入跡象。"
        else: sentiment_reason = f"⚖️ 盤整觀望：目前多空訊號互相抵銷，市場主要聚焦於 {kw_str} 等話題。"
        
        return {
            "sentiment_score": int(final_sentiment), "signal_count": len(signals),
            "noise_count": len(noises), "hot_keywords": sorted_keywords,
            "signals": signals, "noises": noises, "sentiment_reason": sentiment_reason
        }

# ==========================================
# 🚦 7. Routes 
# ==========================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/analysis/<symbol>')
def analysis_page(symbol): return render_template('analysis.html', symbol=symbol)

@app.route('/monte-carlo-info')
def monte_carlo_info(): return render_template('monte_carlo_info.html')

@app.route('/social-sentiment')
def social_sentiment_page(): return render_template('social_sentiment.html')

@app.route('/api/social-data')
@ttl_cache(ttl_seconds=60) 
def get_social_data():
    all_posts = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        f1 = executor.submit(SocialMediaEngine.scrape_ptt)
        f2 = executor.submit(SocialMediaEngine.scrape_cnyes)
        f3 = executor.submit(SocialMediaEngine.scrape_blocktempo)
        f4 = executor.submit(SocialMediaEngine.scrape_coindesk)
        try: all_posts.extend(f1.result())
        except: pass
        try: all_posts.extend(f2.result())
        except: pass
        try: all_posts.extend(f3.result())
        except: pass
        try: all_posts.extend(f4.result())
        except: pass

    if not all_posts:
        return jsonify({
            "sentiment_score": 0, "signal_count": 0, "noise_count": 0, 
            "hot_keywords": [], "signals": [], "noises": [], "sentiment_reason": "查無資料"
        })
    return jsonify(SocialMediaEngine.analyze_posts(all_posts))

@app.route('/api/details/<symbol>')
def get_coin_details(symbol):
    try:
        pair, prices = DataManager.get_kline_safe(symbol + 'USDT')
        btc_pair, btc_prices = DataManager.get_kline_safe('BTCUSDT')
        if not prices: return jsonify({"error": "No data"})
        min_len = min(len(prices), len(btc_prices))
        df = pd.DataFrame({'BTC': btc_prices[-min_len:], symbol: prices[-min_len:]})
        returns = df.pct_change().dropna()
        
        is_stable = symbol in Config.STABLE_COINS
        risk_data = RiskModel.calculate_copula_risk(symbol, df, is_stable, prices[-1])
        sim_data = MonteCarloEngine.simulate_price_paths(prices[-30:])
        
        ai_sfi_text = AIAssistant.generate_sfi_insight(risk_data.get('score', 0))
        ai_copula_text = AIAssistant.generate_copula_insight(risk_data.get('corr', 0), risk_data.get('lambda', 0))
        mean_path_end = sim_data.get('mean_path', [0])[-1] if sim_data and 'mean_path' in sim_data else prices[-1]
        ai_mc_text = AIAssistant.generate_mc_insight(prices[-1], mean_path_end, sim_data.get('volatility', 0) if sim_data else 0)

        return jsonify({
            "btc_returns": [0 if np.isnan(x) else x for x in returns['BTC'].tolist()],
            "coin_returns": [0 if np.isnan(x) else x for x in returns[symbol].tolist()],
            "dates": list(range(len(returns))),
            "simulation": sim_data,
            "risk_data": risk_data, 
            "ai_insights": {
                "sfi": ai_sfi_text,
                "copula": ai_copula_text,
                "mc": ai_mc_text
            }
        })
    except Exception as e: return jsonify({"error": str(e)})

@app.route('/api/coingecko')
@ttl_cache(ttl_seconds=Config.CACHE_TTL)
def live_data():
    crypto_list = DataManager.get_all_tickers()
    if not crypto_list: return jsonify({"timestamp": "--:--", "data": [], "exchange_rate": Config.DEFAULT_EXCHANGE_RATE})
    
    top_symbols = [c['symbol'] for c in crypto_list]
    history_df = DataManager.get_historical_df_parallel(top_symbols)
    current_rate = DataManager.get_realtime_exchange_rate()
    
    for coin in crypto_list:
        symbol = coin['symbol']
        price_usd = coin['price_usd']
        if symbol == 'BTC': 
            coin['risk'] = {"level": "base", "msg": "⚓ 市場基準", "corr": 1.0, "score": 0, "lambda": 0, "beta": 1, "stress": {"s10": price_usd * 0.9}}
        else: 
            coin['risk'] = RiskModel.calculate_copula_risk(symbol, history_df, coin.get('is_stable', False), price_usd)
        coin['price_twd'] = price_usd * current_rate
        
    return jsonify({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "data": crypto_list, 
        "exchange_rate": current_rate,
        "news": NewsEngine.generate_sentiment_news(crypto_list)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
