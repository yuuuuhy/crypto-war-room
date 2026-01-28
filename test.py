import yfinance as yf

print("🚀 系統啟動，連線中...")

try:
    # 這裡我們不加任何破解參數，直接直球對決
    btc = yf.Ticker("BTC-USD")
    
    # 抓取資料
    data = btc.history(period="1d")
    
    if data.empty:
        print("❌ 連線成功但沒抓到資料，請檢查網路。")
    else:
        print("\n✅ 成功！這是剛剛從美國抓回來的比特幣價格：")
        print("-" * 30)
        print(data[['Close', 'Volume']])
        print("-" * 30)
        print("🎉 恭喜！你的 AI 投資平台地基打好了！")

except Exception as e:
    print(f"❌ 發生錯誤: {e}")