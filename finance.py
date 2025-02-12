import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import tkinter as tk
from tkinter import ttk  # ttk modülünü ekleyerek combobox kullanıyoruz
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import matplotlib.pyplot as plt
import threading

# 100 Hisse Sembolü (örnek olarak daha fazla hisse ekledim)
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC', 
    'SPY', 'NFLX', 'BABA', 'AMD', 'V', 'NVDA', 'DIS', 'INTC', 'PYPL', 'AMD', 
    'SNAP', 'BA', 'GE', 'GS', 'VZ', 'UBER', 'F', 'PFE', 'GM', 'WMT', 'T', 
    'UNH', 'NFLX', 'MS', 'SO', 'CRM', 'CVX', 'KHC', 'SBUX', 'KO', 'PEP', 'MO', 
    'IBM', 'MMM', 'DOW', 'CAT', 'LMT', 'GS', 'AMAT', 'INTU', 'ORCL', 'FISV',
    'MCD', 'WBA', 'EXC', 'TGT', 'CMCSA', 'CL', 'MGM', 'RCL', 'BA', 'GS', 'XOM',
    'XEL', 'MDT', 'ABBV', 'MRK', 'CVS', 'UNP', 'NSC', 'LUV', 'LRCX', 'ATVI',
    'CSCO', 'HON', 'BA', 'X', 'OXY', 'PFE', 'MSFT', 'AIG', 'C', 'JNJ', 'MA', 
    'SPG', 'AZO', 'ZTS', 'HRL', 'WFC', 'FDX', 'T', 'DIS', 'UAL', 'ACN', 'SBUX', 
    'HUM', 'EL', 'HSY', 'YUM', 'LULU', 'DPZ', 'TMO', 'BMY', 'STZ', 'WMT'
]

# Hisse Tahmin Fonksiyonu
def predict_stock(ticker):
    data = yf.download(ticker, start='2022-01-01', end='2025-01-20')
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close_scaled'] = scaler.fit_transform(data[['Close']])
    
    # Zaman Serisi
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data['Close_scaled'].iloc[i-sequence_length:i].values)
        y.append(data['Close_scaled'].iloc[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Model
    model = Sequential([ 
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Modeli Eğitme ve Eğitimdeki Doğruluk Bilgisi
    history = model.fit(X, y, batch_size=32, epochs=5, verbose=0)
    loss = history.history['loss'][-1]  # Son epoch'taki loss değeri
    return model, loss

# Anlık Fiyat Fonksiyonu
def fetch_current_price(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.history(period='1d')['Close'].iloc[-1]
    return current_price

# Grafik Çizimi
def draw_graph(ticker):
    model, loss = predict_stock(ticker)
    predictions = predict_future(ticker, model)
    days = range(1, 8)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(days, predictions, marker='o', label=f'{ticker} Tahmini', color='#1E3A8A')  # Mavi renk
    ax.set_title(f"{ticker} Gelecek 7 Gün Tahmini", fontsize=14, color='#FFFFFF')  # Beyaz başlık
    ax.set_xlabel("Gün", fontsize=12, color='#FFFFFF')  # Beyaz yazı
    ax.set_ylabel("Fiyat (USD)", fontsize=12, color='#FFFFFF')  # Beyaz yazı
    ax.legend()

    # Etkileşimli Grafik
    cursor = mplcursors.cursor(ax, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"Gün {int(sel.target[0])}\n{sel.target[1]:.2f} USD"))

    return fig, loss

# Gelecek Tahmin Fonksiyonu
def predict_future(ticker, model):
    data = yf.download(ticker, start='2024-01-20', end='2025-01-20')
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close_scaled'] = scaler.fit_transform(data[['Close']])
    
    sequence_length = 60
    last_sequence = data['Close_scaled'].iloc[-sequence_length:].values
    future_predictions = []
    
    # 7 Günlük Tahmin
    for _ in range(7):
        input_sequence = last_sequence.reshape((1, sequence_length, 1))
        predicted_value = model.predict(input_sequence, verbose=0)
        future_predictions.append(predicted_value[0, 0])
        last_sequence = np.append(last_sequence[1:], predicted_value[0, 0])

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Anlık ve Doğruluk Değeri Güncelleyici
def update_ui(selected_ticker):
    try:
        # Yüklenme Mesajı
        info_label.configure(text="Tahmin işlemi sürüyor... Lütfen bekleyin.")
        info_label.update()

        # Anlık Fiyat ve Tahminleri Al
        current_price = fetch_current_price(selected_ticker)
        model, loss = predict_stock(selected_ticker)
        predictions = predict_future(selected_ticker, model)

        # UI güncelleme işlemini ana thread'e yap
        def update_gui():
            # Tahmin ve Fiyat Bilgisi Güncelle
            info_label.configure(
                text=f"Anlık Fiyat: {current_price:.2f} USD\n\n"
                     + "\n".join([f"Gün {i+1}: {predictions[i][0]:.2f} USD" for i in range(7)]) +
                     f"\n\nModel Doğruluğu (Loss): {loss:.4f}",
                bg="#111827",  # Siyah arka plan
                fg="#FFFFFF",  # Beyaz metin rengi
            )

            # Grafik Güncelle
            fig, _ = draw_graph(selected_ticker)
            for widget in graph_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.get_tk_widget().pack()
            canvas.draw()

        # Ana thread'e UI güncelleme fonksiyonunu ekle
        app.after(0, update_gui)

    except Exception as e:
        info_label.configure(text=f"Hata: {e}")

# Tahmin Başlatıcı
def start_prediction():
    selected_ticker = stock_combobox.get()
    if selected_ticker == "Bir hisse seçin":
        info_label.configure(text="Lütfen bir hisse seçin.")
        return
    
    # Threading ile Arka Planda İşlem
    threading.Thread(target=update_ui, args=(selected_ticker,), daemon=True).start()

# Tkinter Kullanımı
app = tk.Tk()
app.title("Hisse Fiyat Tahmin Uygulaması")
app.geometry("900x700")
app.configure(bg="#111827")  # Siyah genel arka plan

# Hisse Seçimi - ttk.Combobox kullanarak kaydırılabilir dropdown menüsü oluşturuyoruz
stock_combobox = ttk.Combobox(app, values=tickers, width=50, font=("Arial", 12))
stock_combobox.set(tickers[0])  # Varsayılan olarak ilk hisseyi seçili yap
stock_combobox.pack(pady=20)

# Tahmin Butonu
predict_button = tk.Button(app, text="Tahmin Et", command=start_prediction, width=20, height=2, bg="#1E3A8A", fg="white", font=("Arial", 12, "bold"))
predict_button.pack(pady=10)

# Tahmin Bilgisi
info_label = tk.Label(app, text="", font=("Arial", 14), justify="left", bg="#111827", fg="#FFFFFF")
info_label.pack(pady=20)

# Grafik Alanı
graph_frame = tk.Frame(app, width=800, height=400, bg="#111827")  # Siyah grafik alanı
graph_frame.pack(pady=20)

# Çalıştır
app.mainloop()
