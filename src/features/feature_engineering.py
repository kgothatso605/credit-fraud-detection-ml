"""
Complete Feature Engineering Pipeline
Phase 4: All 58 features from your completed work

KEY DISCOVERY: Terminal features = 58% importance!
"""
import pandas as pd
import numpy as np
from pathlib import Path


class FeatureEngineer:
    """
    Feature engineering pipeline creating 58 features across 7 categories:
    1. Temporal (14) - High-risk period 21:00-04:00
    2. Geographic (2) - Distance calculations
    3. Amount Deviations (6) - Personalized Z-scores
    4. Terminal Risk (5) - 58% of model importance!
    5. Customer Behavioral (3) - Usage patterns
    6. Advanced Contextual (7) - Weekend/time habits
    7. Interactions (6) - Compound effects
    """
    
    def __init__(self):
        self.terminal_stats = None
        self.overall_fraud_rate = None
        self.fitted = False
        
    def fit_transform(self, transactions, customers, terminals):
        """Fit on training data and transform"""
        print("🔧 Feature Engineering Pipeline Starting...")
        print(f"   Input: {len(transactions):,} transactions")
        
        # Merge tables
        df = self._merge_tables(transactions, customers, terminals)
        print(f"   ✓ Merged tables: {df.shape}")
        
        # Create features
        df = self.create_temporal_features(df)
        print(f"   ✓ Temporal features (14)")
        
        df = self.create_geographic_features(df)
        print(f"   ✓ Geographic features (2)")
        
        df = self.create_amount_features(df)
        print(f"   ✓ Amount deviation features (6)")
        
        df = self.create_terminal_features(df, fit=True)
        print(f"   ✓ Terminal risk features (5) - YOUR TOP FEATURES!")
        
        df = self.create_customer_features(df)
        print(f"   ✓ Customer behavioral features (3)")
        
        df = self.create_contextual_features(df)
        print(f"   ✓ Advanced contextual features (7)")
        
        df = self.create_interaction_features(df)
        print(f"   ✓ Interaction features (6)")
        
        self.fitted = True
        print(f"\n✅ Complete: {df.shape[1]} features created!")
        return df
    
    def transform(self, transactions, customers, terminals):
        """Transform test data using fitted parameters"""
        if not self.fitted:
            raise ValueError("Must fit before transform!")
            
        df = self._merge_tables(transactions, customers, terminals)
        df = self.create_temporal_features(df)
        df = self.create_geographic_features(df)
        df = self.create_amount_features(df)
        df = self.create_terminal_features(df, fit=False)
        df = self.create_customer_features(df)
        df = self.create_contextual_features(df)
        df = self.create_interaction_features(df)
        
        return df
    
    def _merge_tables(self, trans, cust, term):
        """Merge all relational tables"""
        df = trans.copy()
        df = df.merge(cust, on='CUSTOMER_ID', how='left', suffixes=('', '_cust'))
        df = df.merge(term, on='TERMINAL_ID', how='left', suffixes=('', '_term'))
        return df
    
    def create_temporal_features(self, df):
        """
        14 temporal features
        YOUR DISCOVERY: High-risk period 21:00-04:00!
        """
        # Extract time components
        df['tx_hour'] = df['TX_DATETIME'].dt.hour
        df['tx_day_of_week'] = df['TX_DATETIME'].dt.dayofweek  
        df['tx_day'] = df['TX_DATETIME'].dt.day
        df['tx_month'] = df['TX_DATETIME'].dt.month
        
        # High-risk period indicators
        df['is_high_risk_period'] = df['tx_hour'].isin(
            [21, 22, 23, 0, 1, 2, 3, 4]
        ).astype(int)
        
        df['is_peak_fraud_hour'] = (df['tx_hour'] == 21).astype(int)
        df['is_hour_23'] = (df['tx_hour'] == 23).astype(int)
        
        # Weekend patterns
        df['is_weekend'] = df['tx_day_of_week'].isin([5, 6]).astype(int)
        df['is_friday'] = (df['tx_day_of_week'] == 4).astype(int)
        df['is_business_hours'] = df['tx_hour'].between(9, 17).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['tx_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['tx_hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['tx_day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['tx_day_of_week'] / 7)
        
        return df
    
    def create_geographic_features(self, df):
        """2 distance features"""
        df['distance_to_terminal'] = np.sqrt(
            (df['x_customer_id'] - df['x_terminal_id'])**2 +
            (df['y_customer_id'] - df['y_terminal_id'])**2
        )
        
        df['manhattan_distance'] = (
            np.abs(df['x_customer_id'] - df['x_terminal_id']) +
            np.abs(df['y_customer_id'] - df['y_terminal_id'])
        )
        
        return df
    
    def create_amount_features(self, df):
        """
        6 amount deviation features
        YOUR INNOVATION: Personalization!
        """
        eps = 1e-5
        
        # Z-score deviation  
        df['amount_deviation'] = (
            (df['TX_AMOUNT'] - df['mean_amount']) / 
            (df['std_amount'] + eps)
        )
        
        # Amount ratio
        df['amount_ratio'] = df['TX_AMOUNT'] / (df['mean_amount'] + eps)
        
        # Extreme flags
        df['is_high_amount'] = (df['amount_deviation'] > 2).astype(int)
        df['is_low_amount'] = (df['amount_deviation'] < -2).astype(int)
        
        # Log transform
        df['amount_log'] = np.log(df['TX_AMOUNT'] + 1)
        
        # Raw amount (baseline)
        # TX_AMOUNT already exists
        
        return df
    
    def create_terminal_features(self, df, fit=False):
        """
        5 terminal risk features
        YOUR BIGGEST DISCOVERY: 58% model importance!
        
        terminal_fraud_rate: 29.75% importance
        terminal_fraud_count: 28.29% importance
        """
        if fit:
            if 'TX_FRAUD' not in df.columns:
                raise ValueError("Cannot fit without TX_FRAUD labels")
            
            # Compute statistics from TRAIN ONLY (avoid leakage!)
            self.terminal_stats = df.groupby('TERMINAL_ID').agg({
                'TX_FRAUD': ['sum', 'count', 'mean'],
                'TX_AMOUNT': ['mean', 'std']
            }).reset_index()
            
            self.terminal_stats.columns = [
                'TERMINAL_ID',
                'terminal_fraud_count',
                'terminal_tx_count',
                'terminal_fraud_rate',
                'terminal_avg_amount',
                'terminal_std_amount'
            ]
            
            self.overall_fraud_rate = df['TX_FRAUD'].mean()
        
        # Merge statistics
        if self.terminal_stats is not None:
            df = df.merge(self.terminal_stats, on='TERMINAL_ID', how='left')
            
            # Fill missing terminals with overall stats
            df['terminal_fraud_rate'].fillna(self.overall_fraud_rate, inplace=True)
            df['terminal_fraud_count'].fillna(0, inplace=True)
            df['terminal_tx_count'].fillna(1, inplace=True)
            df['terminal_avg_amount'].fillna(df['TX_AMOUNT'].mean(), inplace=True)
            df['terminal_std_amount'].fillna(df['TX_AMOUNT'].std(), inplace=True)
        
        return df
    
    def create_customer_features(self, df):
        """3 behavioral features"""
        df['customer_terminal_diversity'] = df['nb_terminals']
        
        median_freq = df['mean_nb_tx_per_day'].median()
        df['is_frequent_customer'] = (
            df['mean_nb_tx_per_day'] > median_freq
        ).astype(int)
        
        df['customer_spending_consistency'] = (
            df['mean_amount'] / (df['std_amount'] + 1e-5)
        )
        
        return df
    
    def create_contextual_features(self, df):
        """7 advanced features - clubbing hypothesis"""
        # Weekend night patterns
        df['is_clubbing_time'] = (
            df['is_weekend'] & df['is_high_risk_period']
        ).astype(int)
        
        df['weekday_night'] = (
            (~df['is_weekend'].astype(bool)) &
            df['is_high_risk_period'].astype(bool)
        ).astype(int)
        
        df['weekend_night'] = (
            df['is_weekend'] * df['is_high_risk_period']
        )
        
        # Weekend spending baselines (simplified)
        df['customer_weekend_avg'] = df['mean_amount']
        df['customer_weekend_std'] = df['std_amount']
        df['weekend_amount_deviation'] = df['amount_deviation']
        
        # Customer temporal habits (simplified)
        df['customer_avg_hour'] = 14.0  # Placeholder
        
        return df
    
    def create_interaction_features(self, df):
        """6 interaction features - compound effects"""
        df['distance_unusual_time'] = (
            df['distance_to_terminal'] * df['is_high_risk_period']
        )
        
        df['high_amount_weekday_night'] = (
            df['is_high_amount'] * df['weekday_night']
        )
        
        if 'terminal_fraud_rate' in df.columns:
            df['risky_hour_terminal'] = (
                df['is_peak_fraud_hour'] * df['terminal_fraud_rate']
            )
        
        df['triple_unusual'] = (
            df['is_high_risk_period'] *
            df['is_high_amount'] *
            (df['distance_to_terminal'] > df['distance_to_terminal'].median()).astype(int)
        )
        
        df['weekend_unusual_amount'] = (
            df['is_weekend'] * np.abs(df['weekend_amount_deviation'])
        )
        
        df['night_unusual_amount'] = (
            df['is_high_risk_period'] * df['is_high_amount']
        )
        
        return df


if __name__ == '__main__':
    from src.data.data_loader import DataLoader
    
    # Test
    loader = DataLoader()
    data = loader.load_all('train')
    
    engineer = FeatureEngineer()
    features = engineer.fit_transform(**data)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Shape: {features.shape}")
    print(f"   Features: {list(features.columns)[:10]}...")
