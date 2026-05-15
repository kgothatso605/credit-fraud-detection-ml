"""
Data Loading Module
Phases 1-2: Load and merge relational data
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


class DataLoader:
    """Load fraud detection data from CSV files"""
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = Path(data_dir)
        
    def load_transactions(self, split: str = 'train') -> pd.DataFrame:
        """
        Load transaction data
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            DataFrame with transactions
        """
        filename = f'{split}_transactions.csv'
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Parse datetime
        if 'TX_DATETIME' in df.columns:
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        
        print(f"✓ Loaded {len(df):,} {split} transactions")
        return df
    
    def load_customers(self) -> pd.DataFrame:
        """Load customer profiles"""
        filepath = self.data_dir / 'customers.csv'
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df):,} customers")
        return df
    
    def load_terminals(self) -> pd.DataFrame:
        """Load terminal data"""
        filepath = self.data_dir / 'terminals.csv'
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df):,} terminals")
        return df
    
    def load_all(self, split: str = 'train') -> Dict[str, pd.DataFrame]:
        """
        Load all datasets
        
        Returns:
            Dictionary with 'transactions', 'customers', 'terminals'
        """
        return {
            'transactions': self.load_transactions(split),
            'customers': self.load_customers(),
            'terminals': self.load_terminals()
        }
    
    def merge_data(self, transactions: pd.DataFrame, 
                   customers: pd.DataFrame,
                   terminals: pd.DataFrame) -> pd.DataFrame:
        """
        Merge relational tables
        From your Phase 2 work
        """
        df = transactions.copy()
        
        # Merge customers
        df = df.merge(customers, on='CUSTOMER_ID', how='left')
        print(f"✓ Merged customers: {len(df.columns)} columns")
        
        # Merge terminals  
        df = df.merge(terminals, on='TERMINAL_ID', how='left')
        print(f"✓ Merged terminals: {len(df.columns)} columns")
        
        return df


if __name__ == '__main__':
    # Test loading
    loader = DataLoader()
    data = loader.load_all('train')
    merged = loader.merge_data(**data)
    print(f"\n✓ Final shape: {merged.shape}")
