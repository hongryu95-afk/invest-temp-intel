"""US 주식 지수(SPY, QQQ) 방향 예측기 (v2).

27개 피처(가격/VIX/섹터/매크로/거래량/모멘텀)를 기반으로 다음 5일 방향과
수익률을 예측한다. GradientBoostingClassifier + Regressor 조합 + TimeSeriesSplit
5-fold CV. 예측 히스토리는 최대 100개 유지한다.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndexPredictor:
    """SPY/QQQ 지수의 5일 방향 예측 모델."""

    FEATURE_NAMES = [
        # SPY (7)
        'spy_return_1w', 'spy_return_1m', 'spy_above_200ma', 'spy_above_50ma',
        'spy_rsi', 'spy_macd_signal', 'spy_bb_position',
        # VIX (3)
        'vix_value', 'vix_change_5d', 'vix_percentile',
        # QQQ (2)
        'qqq_return_1w', 'qqq_rsi',
        # 시장폭 (2)
        'breadth_pct_above_50ma', 'advance_decline_ratio',
        # 섹터 상대강도 (3)
        'xlk_relative_1m', 'xlu_relative_1m', 'xly_relative_1m',
        # 매크로 (3)
        'yield_spread_proxy', 'gold_return_1w', 'dxy_return_1w',
        # 거래량 (3)
        'spy_vol_ratio', 'spy_vol_trend_5d', 'qqq_vol_ratio',
        # 모멘텀 (4)
        'spy_roc_10d', 'spy_price_vs_50ma_pct', 'spy_rsi_slope_5d', 'vix_above_20',
    ]

    INVERSE_FEATURES = {
        'vix_value', 'vix_change_5d', 'vix_percentile', 'vix_above_20',
        'xlu_relative_1m', 'dxy_return_1w',
    }

    def __init__(self, data_dir: str = '.'):
        self.data_dir = Path(data_dir)
        self.output_file = self.data_dir / 'output' / 'index_prediction.json'
        self.model_path_spy = self.data_dir / 'output' / 'predictor_model_spy.joblib'
        self.model_path_qqq = self.data_dir / 'output' / 'predictor_model_qqq.joblib'
        self.history_file = self.data_dir / 'output' / 'prediction_history.json'
        self.config = self._load_regime_config()
        logger.info("IndexPredictor initialized: data_dir=%s, horizon=%d",
                    self.data_dir, self.config.get('prediction_horizon_days', 5))

    # ------------------------------------------------------------------
    # Config (prompt 9)
    # ------------------------------------------------------------------

    def _load_regime_config(self) -> Dict:
        """output/regime_config.json의 'predictor' 키 로드, 없으면 기본값."""
        defaults = {
            'prediction_horizon_days': 5,
            'cv_splits': 5,
            'retrain_interval_days': 7,
            'min_training_samples': 50,
            'confidence_high_threshold': 70,
            'confidence_moderate_threshold': 60,
        }
        config_file = self.data_dir / 'output' / 'regime_config.json'
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                if 'predictor' in config_data:
                    defaults.update(config_data['predictor'])
                    logger.debug("Loaded predictor config from %s", config_file)
            except Exception as e:
                logger.debug("Failed to load regime config: %s", e)
        return defaults

    # ------------------------------------------------------------------
    # Technical indicators (prompt 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's smoothing RSI. gain/loss 모두 0이면 50."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def _calculate_macd_signal(series: pd.Series) -> pd.Series:
        """MACD 시그널: +1 (MACD>signal) / -1 (below) / 0 (equal)."""
        ema_12 = series.ewm(span=12, adjust=False).mean()
        ema_26 = series.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        result = np.where(macd > signal, 1, np.where(macd < signal, -1, 0))
        return pd.Series(result, index=series.index, dtype=int)

    @staticmethod
    def _calculate_bb_position(series: pd.Series, window: int = 20) -> pd.Series:
        """Bollinger Band 내 위치 (0~1). band_width=0 보호, clip + fillna."""
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        band_width = (upper - lower).replace(0, np.nan)
        position = (series - lower) / band_width
        return position.clip(0, 1).fillna(0.5)

    # ------------------------------------------------------------------
    # Feature building (prompt 3)
    # ------------------------------------------------------------------

    def _build_raw_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """27개 피처 생성. 컬럼 없으면 해당 피처 스킵."""
        df = pd.DataFrame(index=data.index)

        # 1. SPY 7개
        if 'SPY' in data.columns:
            spy = data['SPY']
            df['spy_return_1w'] = spy.pct_change(5) * 100
            df['spy_return_1m'] = spy.pct_change(21) * 100
            df['spy_above_200ma'] = (spy > spy.rolling(200).mean()).astype(int)
            df['spy_above_50ma'] = (spy > spy.rolling(50).mean()).astype(int)
            df['spy_rsi'] = self._calculate_rsi(spy)
            df['spy_macd_signal'] = self._calculate_macd_signal(spy)
            df['spy_bb_position'] = self._calculate_bb_position(spy)

        # 2. VIX 3개 + 9. vix_above_20
        if 'VIX' in data.columns:
            vix = data['VIX']
            df['vix_value'] = vix
            df['vix_change_5d'] = vix.pct_change(5) * 100
            df['vix_percentile'] = vix.rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            df['vix_above_20'] = (vix > 20).astype(int)

        # 3. QQQ 2개
        if 'QQQ' in data.columns:
            qqq = data['QQQ']
            df['qqq_return_1w'] = qqq.pct_change(5) * 100
            df['qqq_rsi'] = self._calculate_rsi(qqq)

        # 4. 시장폭 2개
        if 'SPY' in data.columns:
            spy = data['SPY']
            above_50 = (spy > spy.rolling(50).mean()).astype(float)
            df['breadth_pct_above_50ma'] = above_50.rolling(50).mean() * 100
            ret = spy.pct_change()
            adv = (ret > 0).rolling(20).sum()
            dec = (ret < 0).rolling(20).sum()
            df['advance_decline_ratio'] = (adv + 1) / (dec + 1)  # Laplace smoothing

        # 5. 섹터 상대강도 3개
        if 'SPY' in data.columns:
            spy_1m = data['SPY'].pct_change(21)
            for ticker, name in [('XLK', 'xlk_relative_1m'),
                                 ('XLU', 'xlu_relative_1m'),
                                 ('XLY', 'xly_relative_1m')]:
                if ticker in data.columns:
                    df[name] = (data[ticker].pct_change(21) - spy_1m) * 100

        # 6. 매크로 3개
        if 'TNX' in data.columns:
            if 'FVX' in data.columns:
                df['yield_spread_proxy'] = data['TNX'] - data['FVX']
            else:
                df['yield_spread_proxy'] = data['TNX'].pct_change(5) * 100
        if 'GOLD' in data.columns:
            df['gold_return_1w'] = data['GOLD'].pct_change(5) * 100
        if 'DXY' in data.columns:
            df['dxy_return_1w'] = data['DXY'].pct_change(5) * 100

        # 7. 거래량 3개
        if 'SPY_VOL' in data.columns:
            spy_vol = data['SPY_VOL']
            df['spy_vol_ratio'] = spy_vol / spy_vol.rolling(20).mean()
            df['spy_vol_trend_5d'] = (spy_vol.rolling(5).mean()
                                      / spy_vol.rolling(20).mean() - 1)
        if 'QQQ_VOL' in data.columns:
            qqq_vol = data['QQQ_VOL']
            df['qqq_vol_ratio'] = qqq_vol / qqq_vol.rolling(20).mean()

        # 8. 모멘텀 3개 (vix_above_20 is in section 2)
        if 'SPY' in data.columns:
            spy = data['SPY']
            sma_50 = spy.rolling(50).mean()
            df['spy_roc_10d'] = spy.pct_change(10) * 100
            df['spy_price_vs_50ma_pct'] = ((spy - sma_50) / sma_50) * 100
            if 'spy_rsi' in df.columns:
                df['spy_rsi_slope_5d'] = df['spy_rsi'] - df['spy_rsi'].shift(5)

        return df

    # ------------------------------------------------------------------
    # Data fetching (prompt 4)
    # ------------------------------------------------------------------

    def _fetch_price_data(self, start_date: str = '2023-01-01') -> pd.DataFrame:
        """yfinance로 가격 데이터 수집 (300일 lookback)."""
        try:
            import yfinance as yf
            from curl_cffi import requests as curl_requests
        except ImportError as e:
            logger.error("Missing dependency for data fetch: %s", e)
            return pd.DataFrame()

        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            logger.error("Invalid start_date format: %s", start_date)
            return pd.DataFrame()

        lookback_start = (start_dt - timedelta(days=300)).strftime('%Y-%m-%d')

        tickers = ['SPY', 'QQQ', '^VIX', 'XLK', 'XLU', 'XLY',
                   'GC=F', 'DX-Y.NYB', '^TNX', '^FVX']
        rename_map = {
            '^VIX': 'VIX', 'GC=F': 'GOLD', 'DX-Y.NYB': 'DXY',
            '^TNX': 'TNX', '^FVX': 'FVX',
        }

        session = curl_requests.Session(impersonate="chrome")
        result = pd.DataFrame()

        try:
            for ticker in tickers:
                clean_name = rename_map.get(ticker, ticker)
                try:
                    t = yf.Ticker(ticker, session=session)
                    hist = t.history(start=lookback_start, auto_adjust=False)
                    if hist.empty:
                        logger.warning("Empty data for %s", ticker)
                        continue
                    # Normalize index: strip timezone + reduce to date-only
                    # (yfinance returns different timezones per ticker:
                    #  SPY→NYC, VIX→Chicago, causing reindex misalignment)
                    if hist.index.tz is not None:
                        hist.index = hist.index.tz_localize(None)
                    hist.index = hist.index.normalize()
                    if result.empty:
                        result = pd.DataFrame(index=hist.index)
                    result[clean_name] = hist['Close'].reindex(result.index)
                    if ticker in ('SPY', 'QQQ'):
                        result[f'{ticker}_VOL'] = hist['Volume'].reindex(result.index)
                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", ticker, e)

            logger.info("Fetched price data: %d rows, %d columns",
                        len(result), len(result.columns))
            return result

        except Exception as e:
            logger.error("Failed to fetch price data: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Training dataset (prompt 5)
    # ------------------------------------------------------------------

    def reconstruct_signals_from_prices(
        self, start_date: str = '2023-01-01'
    ) -> pd.DataFrame:
        """가격 데이터로부터 27 피처 + SPY/QQQ 타겟을 재구성."""
        data = self._fetch_price_data(start_date)
        if data.empty:
            logger.warning("No price data, returning empty DataFrame")
            return pd.DataFrame()

        features = self._build_raw_features(data)

        horizon = self.config.get('prediction_horizon_days', 5)

        if 'SPY' in data.columns:
            features['spy_target_return'] = (
                data['SPY'].pct_change(horizon).shift(-horizon) * 100
            )
            features['spy_target_direction'] = (
                (features['spy_target_return'] > 0).astype(int)
            )
        if 'QQQ' in data.columns:
            features['qqq_target_return'] = (
                data['QQQ'].pct_change(horizon).shift(-horizon) * 100
            )
            features['qqq_target_direction'] = (
                (features['qqq_target_return'] > 0).astype(int)
            )

        # Filter by start_date
        start_ts = pd.Timestamp(start_date)
        features = features[features.index >= start_ts]

        feat_cols = [c for c in features.columns
                     if not c.startswith(('spy_target', 'qqq_target'))]
        logger.info("Reconstructed signals: %d rows, %d features",
                    len(features), len(feat_cols))
        return features

    # ------------------------------------------------------------------
    # Training (prompt 6)
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, target_ticker: str = 'SPY') -> Dict:
        """GradientBoosting + TimeSeriesSplit 5-fold CV 학습."""
        try:
            from sklearn.ensemble import (
                GradientBoostingClassifier, GradientBoostingRegressor,
            )
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score, brier_score_loss
            import joblib
        except ImportError as e:
            msg = (f"scikit-learn/joblib not installed: {e}. "
                   "Run: pip install scikit-learn joblib")
            logger.error(msg)
            return {'error': msg}

        ticker_lower = target_ticker.lower()
        direction_col = f'{ticker_lower}_target_direction'
        return_col = f'{ticker_lower}_target_return'

        if direction_col not in df.columns or return_col not in df.columns:
            msg = f"Missing target columns: {direction_col}, {return_col}"
            logger.error(msg)
            return {'error': msg}

        # Drop rows with NaN targets (last N rows due to shift)
        clean_df = df.dropna(subset=[direction_col, return_col]).copy()

        # Select available features (some may have been skipped by _build_raw_features)
        available_features = [f for f in self.FEATURE_NAMES if f in clean_df.columns]
        clean_df = clean_df.dropna(subset=available_features)

        min_samples = self.config.get('min_training_samples', 50)
        if len(clean_df) < min_samples:
            msg = f"Insufficient training samples: {len(clean_df)} < {min_samples}"
            logger.error(msg)
            return {'error': msg}

        X = clean_df[available_features].values
        y_dir = clean_df[direction_col].values.astype(int)
        y_ret = clean_df[return_col].values.astype(float)

        # Class imbalance correction
        n_bullish = int((y_dir == 1).sum())
        n_bearish = int((y_dir == 0).sum())
        total = len(y_dir)
        weight_bullish = total / (2 * n_bullish) if n_bullish > 0 else 1.0
        weight_bearish = total / (2 * n_bearish) if n_bearish > 0 else 1.0
        sample_weights = np.where(y_dir == 1, weight_bullish, weight_bearish)

        # TimeSeriesSplit CV
        n_splits = self.config.get('cv_splits', 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        accuracies: List[float] = []
        brier_scores: List[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_dir[train_idx], y_dir[test_idx]
            w_train = sample_weights[train_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                logger.warning("Fold %d skipped: single class", fold_idx)
                continue

            scaler_cv = StandardScaler()
            X_train_s = scaler_cv.fit_transform(X_train)
            X_test_s = scaler_cv.transform(X_test)

            clf = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42,
            )
            clf.fit(X_train_s, y_train, sample_weight=w_train)
            y_pred = clf.predict(X_test_s)
            y_proba = clf.predict_proba(X_test_s)[:, 1]

            accuracies.append(accuracy_score(y_test, y_pred))
            brier_scores.append(brier_score_loss(y_test, y_proba))

        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        avg_brier = float(np.mean(brier_scores)) if brier_scores else 0.0

        # Final training on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        classifier = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42,
        )
        classifier.fit(X_scaled, y_dir, sample_weight=sample_weights)

        regressor = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42,
        )
        regressor.fit(X_scaled, y_ret, sample_weight=sample_weights)

        # Feature importance from classifier
        feature_importance = dict(
            zip(available_features, classifier.feature_importances_.tolist())
        )
        top_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Save model
        model_path = (self.model_path_spy if target_ticker.upper() == 'SPY'
                      else self.model_path_qqq)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_dict = {
            'classifier': classifier,
            'regressor': regressor,
            'scaler': scaler,
            'features': available_features,
            'feature_importance': feature_importance,
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(clean_df),
            'cv_accuracy': avg_accuracy,
            'target_std': float(np.std(y_ret)),
        }
        joblib.dump(model_dict, model_path)

        logger.info("Trained %s: accuracy=%.3f, brier=%.3f, samples=%d → %s",
                    target_ticker, avg_accuracy, avg_brier,
                    len(clean_df), model_path)

        return {
            'target_ticker': target_ticker,
            'accuracy': avg_accuracy,
            'brier_score': avg_brier,
            'training_samples': len(clean_df),
            'features_used': len(available_features),
            'top_features': top_features,
        }

    # ------------------------------------------------------------------
    # Latest features (prompt 7)
    # ------------------------------------------------------------------

    def build_latest_features(self) -> Optional[pd.Series]:
        """예측 시점의 최신 피처 벡터. NaN 과반이면 None."""
        data = self._fetch_price_data(start_date='2024-01-01')
        if data.empty:
            logger.warning("No data to build latest features")
            return None

        features = self._build_raw_features(data)
        available = [f for f in self.FEATURE_NAMES if f in features.columns]
        if not available:
            logger.warning("No usable features in raw features")
            return None

        latest = features[available].iloc[-1]
        nan_count = int(latest.isna().sum())
        if nan_count > len(latest) / 2:
            logger.warning("Latest features have too many NaN: %d/%d",
                           nan_count, len(latest))
            return None

        return latest.fillna(0)

    # ------------------------------------------------------------------
    # Prediction (prompt 8)
    # ------------------------------------------------------------------

    def predict_next_week(self) -> Dict:
        """SPY/QQQ 다음 5일 방향 예측 + output/index_prediction.json 저장."""
        try:
            import joblib
        except ImportError as e:
            msg = f"joblib not installed: {e}"
            logger.error(msg)
            return {'error': msg}

        latest_features = self.build_latest_features()
        if latest_features is None:
            return {'error': 'Failed to build latest features'}

        retrain_interval = self.config.get('retrain_interval_days', 7)
        high_threshold = self.config.get('confidence_high_threshold', 70)
        mod_threshold = self.config.get('confidence_moderate_threshold', 60)

        results: Dict = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'predictions': {},
        }

        training_df = None  # lazy load

        for ticker in ['SPY', 'QQQ']:
            ticker_lower = ticker.lower()
            model_path = (self.model_path_spy if ticker == 'SPY'
                          else self.model_path_qqq)

            needs_training = not model_path.exists()
            model_dict = None

            if not needs_training:
                try:
                    model_dict = joblib.load(model_path)
                    trained_at_str = model_dict.get('trained_at', '2000-01-01')
                    trained_at = datetime.fromisoformat(trained_at_str)
                    age_days = (datetime.now() - trained_at).days
                    if age_days > retrain_interval:
                        logger.info("%s model age=%d days, retraining",
                                    ticker, age_days)
                        needs_training = True
                except Exception as e:
                    logger.warning("Failed to load %s model: %s", ticker, e)
                    needs_training = True

            if needs_training:
                if training_df is None:
                    training_df = self.reconstruct_signals_from_prices()
                if training_df.empty:
                    logger.error("%s training skipped: no data", ticker)
                    continue
                train_result = self.train(training_df, target_ticker=ticker)
                if 'error' in train_result:
                    logger.error("%s training failed: %s",
                                 ticker, train_result['error'])
                    continue
                model_dict = joblib.load(model_path)

            if model_dict is None:
                continue

            classifier = model_dict['classifier']
            regressor = model_dict['regressor']
            scaler = model_dict['scaler']
            features_list = model_dict['features']
            feature_importance = model_dict.get('feature_importance', {})

            # Align latest features with model features
            feat_values = []
            for feat in features_list:
                if feat in latest_features.index:
                    feat_values.append(float(latest_features[feat]))
                else:
                    logger.warning("Missing feature %s for %s, using 0",
                                   feat, ticker)
                    feat_values.append(0.0)

            X_arr = np.array(feat_values).reshape(1, -1)
            X_scaled = scaler.transform(X_arr)

            proba_up = float(classifier.predict_proba(X_scaled)[0, 1])
            pred_return = float(regressor.predict(X_scaled)[0])

            confidence_pct = max(proba_up, 1 - proba_up) * 100
            if confidence_pct >= high_threshold:
                confidence = 'high'
            elif confidence_pct >= mod_threshold:
                confidence = 'moderate'
            else:
                confidence = 'low'

            # Key drivers: top 5 by importance + direction
            sorted_imp = sorted(feature_importance.items(),
                                key=lambda x: x[1], reverse=True)
            key_drivers = []
            for feat_name, importance in sorted_imp[:5]:
                if feat_name in latest_features.index:
                    value = float(latest_features[feat_name])
                    direction = self._get_driver_direction(feat_name, value)
                    key_drivers.append({
                        'feature': feat_name,
                        'importance': round(float(importance), 4),
                        'value': round(value, 3),
                        'direction': direction,
                    })

            direction = 'bullish' if proba_up > 0.5 else 'bearish'
            results['predictions'][ticker_lower] = {
                'direction': direction,
                'probability_up': round(proba_up, 3),
                'predicted_return': round(pred_return, 3),
                'confidence': confidence,
                'confidence_pct': round(confidence_pct, 1),
                'key_drivers': key_drivers,
                'model_trained_at': model_dict.get('trained_at'),
                'model_accuracy': round(float(model_dict.get('cv_accuracy', 0.0)), 4),
            }

        # Save to output file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved prediction to %s", self.output_file)
        try:
            from db import data_store as _ds
            _conn = _ds.get_db()
            _ds.upsert_index_prediction(_conn, results)
            _conn.close()
        except Exception as _e:
            logger.warning("SQLite index_prediction 쓰기 실패: %s", _e)

        # Save to history
        self._save_prediction_history(results)

        return results

    def _get_driver_direction(self, feature_name: str, value: float) -> str:
        """피처 현재값 기반 방향 해석 (bullish/bearish/neutral)."""
        # Inverse features: 양수면 bearish
        if feature_name in self.INVERSE_FEATURES:
            return 'bearish' if value > 0 else 'bullish'

        # RSI: >70 bearish, <30 bullish
        if 'rsi' in feature_name and 'slope' not in feature_name:
            if value > 70:
                return 'bearish'
            if value < 30:
                return 'bullish'
            return 'neutral'

        # BB position: >0.8 bearish, <0.2 bullish
        if 'bb_position' in feature_name:
            if value > 0.8:
                return 'bearish'
            if value < 0.2:
                return 'bullish'
            return 'neutral'

        # Default: positive = bullish
        if value > 0:
            return 'bullish'
        if value < 0:
            return 'bearish'
        return 'neutral'

    # ------------------------------------------------------------------
    # Prediction history (prompt 10)
    # ------------------------------------------------------------------

    def _save_prediction_history(self, prediction: Dict) -> None:
        """prediction_history.json에 append (최대 100개 유지)."""
        try:
            history: List[Dict] = []
            if self.history_file.exists():
                with open(self.history_file) as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = []

            spy_pred = prediction.get('predictions', {}).get('spy', {})
            qqq_pred = prediction.get('predictions', {}).get('qqq', {})

            entry = {
                'date': prediction.get('date', datetime.now().strftime('%Y-%m-%d')),
                'spy': {
                    'direction': spy_pred.get('direction'),
                    'probability': spy_pred.get('probability_up'),
                    'predicted_return': spy_pred.get('predicted_return'),
                },
                'qqq': {
                    'direction': qqq_pred.get('direction'),
                    'probability': qqq_pred.get('probability_up'),
                    'predicted_return': qqq_pred.get('predicted_return'),
                },
                'model_accuracy': round(
                    float(spy_pred.get('model_accuracy', 0.0)) * 100, 1
                ),
            }
            history.append(entry)

            if len(history) > 100:
                history = history[-100:]

            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            logger.info("Saved prediction history (%d entries)", len(history))
            try:
                from db import data_store as _ds
                _conn = _ds.get_db()
                _ds.upsert_prediction_history_entry(_conn, entry)
                _conn.close()
            except Exception as _e:
                logger.warning("SQLite prediction_history 쓰기 실패: %s", _e)
        except Exception as e:
            logger.warning("Failed to save prediction history: %s", e)


# ------------------------------------------------------------------
# CLI entry point (prompt 12)
# ------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    predictor = IndexPredictor(data_dir='.')
    result = predictor.predict_next_week()

    if 'error' in result:
        logger.error("Prediction failed: %s", result['error'])
    else:
        predictions = result.get('predictions', {})
        logger.info("=" * 60)
        for ticker in ['spy', 'qqq']:
            pred = predictions.get(ticker, {})
            if not pred:
                continue
            logger.info("%s Prediction:", ticker.upper())
            logger.info("  Direction: %s", pred.get('direction'))
            logger.info("  Probability (up): %.1f%%",
                        pred.get('probability_up', 0) * 100)
            logger.info("  Predicted Return: %.2f%%",
                        pred.get('predicted_return', 0))
            logger.info("  Confidence: %s (%.1f%%)",
                        pred.get('confidence'), pred.get('confidence_pct', 0))
            drivers = pred.get('key_drivers', [])[:3]
            if drivers:
                logger.info("  Top 3 Key Drivers:")
                for i, d in enumerate(drivers, 1):
                    logger.info("    %d. %s = %.3f (%s, imp=%.4f)",
                                i, d['feature'], d['value'],
                                d['direction'], d['importance'])
            logger.info("-" * 60)
        logger.info("Saved to: %s", predictor.output_file)
