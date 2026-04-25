mkdir -p dataset/ETT-small
mkdir -p dataset/electricity
mkdir -p dataset/traffic
mkdir -p dataset/weather
mkdir -p dataset/exchange_rate   # ✅ FIX ADDED

(
  cd dataset

  COMMON_ARGS="-t 5 -nc"
  URL_PREFIX="https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main"
  
  wget $COMMON_ARGS $URL_PREFIX/ETTh1.csv &
  wget $COMMON_ARGS $URL_PREFIX/ETTh2.csv &
  wget $COMMON_ARGS $URL_PREFIX/ETTm1.csv &
  wget $COMMON_ARGS $URL_PREFIX/ETTm2.csv &
  wget $COMMON_ARGS $URL_PREFIX/PEMS03.npz &
  wget $COMMON_ARGS $URL_PREFIX/PEMS04.npz &
  wget $COMMON_ARGS $URL_PREFIX/PEMS07.npz &
  wget $COMMON_ARGS $URL_PREFIX/PEMS08.npz &
  wget $COMMON_ARGS $URL_PREFIX/electricity.csv &
  wget $COMMON_ARGS $URL_PREFIX/exchange_rate.csv &   # already there
  wget $COMMON_ARGS $URL_PREFIX/solar_AL.pkl &
  wget $COMMON_ARGS $URL_PREFIX/solar_AL.txt &
  wget $COMMON_ARGS $URL_PREFIX/traffic.csv &
  wget $COMMON_ARGS $URL_PREFIX/weather.csv &
  wget $COMMON_ARGS $URL_PREFIX/national_illness.csv &
  
  # illness fallback
  if [ ! -f "national_illness.csv" ]; then
      wget $COMMON_ARGS $URL_PREFIX/illness.csv &
  fi
  
  wait

  # rename if needed
  if [ -f "illness.csv" ]; then
      mv illness.csv national_illness.csv
  fi

  # -------------------------
  # 🔗 Create symlinks
  # -------------------------

  # ETT datasets
  cd ETT-small
  ln -sf ../ETT*.csv .
  cd ..

)

# Electricity
(
    cd dataset/electricity
    ln -sf ../electricity.csv .
)

# Traffic
(
    cd dataset/traffic
    ln -sf ../traffic.csv .
)

# Weather
(
    cd dataset/weather
    ln -sf ../weather.csv .
)

# ✅ FIX: Exchange rate
(
    cd dataset/exchange_rate
    ln -sf ../exchange_rate.csv .
)

# -------------------------
# 🧹 Cleanup unused formats
# -------------------------
(
    cd dataset
    rm -f *.npz *.zip *.pkl *.txt illness.csv solar_AL.*
)

wait