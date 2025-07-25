"""
YouTube Data Cleaning Pipeline

This module contains utility functions for cleaning YouTube comment data for research purposes.
Can be imported into notebooks or used as a standalone script.

Usage Example:
    from cleaning_utils import DataCleaner

    # Initialize cleaner
    cleaner = DataCleaner(target_language='pt')

    # Load and clean data
    raw_df = pd.read_parquet('../data/raw/youtube_comments.parquet')
    clean_df = cleaner.clean_pipeline(raw_df)

    # Export cleaned data
    cleaner.export_cleaned_data(clean_df, '../data/intermediate/cleaned_comments.parquet')
"""

import pandas as pd
import numpy as np
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from tqdm.auto import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Comprehensive data cleaning pipeline for YouTube comment research.
    """
    
    def __init__(self, target_language: str = 'pt', min_percentile: float = 0.1, 
                 max_percentile: float = 0.999, batch_size: int = 784):
        """
        Initialize the data cleaner.
        
        Args:
            target_language: Target language code (default: 'pt')
            min_percentile: Minimum text length percentile (default: 0.1)
            max_percentile: Maximum text length percentile (default: 0.999)
            batch_size: Batch size for language detection (default: 784)
        """
        self.target_language = target_language
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.batch_size = batch_size
        self.language_detector = None
        
    def setup_language_detection(self):
        """Set up language detection pipeline."""
        if self.language_detector is None:
            try:
                from transformers import pipeline
                self.language_detector = pipeline(
                    "text-classification", 
                    model="papluca/xlm-roberta-base-language-detection",
                    device=-1
                )
                logger.info("Language detection model loaded successfully")
            except ImportError:
                raise ImportError("transformers library required for language detection")
        return self.language_detector
    
    def filter_by_text_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter comments by text length using percentile thresholds."""
        df_with_length = df.copy()
        df_with_length["len_text"] = df_with_length["textDisplay"].str.len()
        
        min_length = df_with_length["len_text"].quantile(self.min_percentile)
        max_length = df_with_length["len_text"].quantile(self.max_percentile)
        
        df_filtered = df_with_length[
            (df_with_length["len_text"] > min_length) & 
            (df_with_length["len_text"] < max_length)
        ].copy()
        
        df_filtered = df_filtered.drop(columns=["len_text"])
        return df_filtered.reset_index(drop=True)
    
    def remove_emoji_only_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove comments that consist only of emojis."""
        def is_emoji_only(text: str) -> bool:
            if pd.isna(text) or text.strip() == "":
                return False
                
            text_clean = text.strip()
            if not text_clean:
                return False
            
            emoji_ranges = [
                (0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6FF),
                (0x1F1E0, 0x1F1FF), (0x2600, 0x26FF), (0x2700, 0x27BF),
                (0xFE00, 0xFE0F), (0x1F900, 0x1F9FF),
            ]
            
            for char in text_clean:
                char_code = ord(char)
                is_emoji = any(start <= char_code <= end for start, end in emoji_ranges)
                if not (is_emoji or char.isspace()):
                    return False
            return True
        
        emoji_mask = df["textDisplay"].apply(is_emoji_only)
        return df[~emoji_mask].reset_index(drop=True)
    
    def normalize_data_structures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize nested data structures."""
        df_normalized = df.copy()
        
        if 'authorChannelId' in df_normalized.columns:
            sample_value = df_normalized['authorChannelId'].dropna().iloc[0] if len(df_normalized['authorChannelId'].dropna()) > 0 else None
            if isinstance(sample_value, dict):
                df_normalized['authorChannelId'] = df_normalized['authorChannelId'].apply(
                    lambda x: x.get('value') if isinstance(x, dict) and 'value' in x else x
                )
        
        return df_normalized
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate comments using content-based hashing."""
        def create_comment_hash(row):
            try:
                video_id = str(row.get('video_id', ''))
                text = str(row.get('textDisplay', ''))
                author_url = str(row.get('authorChannelUrl', ''))
                composite = video_id + text + author_url
                return hashlib.md5(composite.encode('utf-8')).hexdigest()
            except Exception:
                return str(hash(str(row)))
        
        df_dedup = df.copy()
        df_dedup['comment_uuid'] = df_dedup.apply(create_comment_hash, axis=1)
        df_dedup = df_dedup.drop_duplicates(subset=['comment_uuid'])
        df_dedup = df_dedup.drop(columns=['comment_uuid'])
        return df_dedup.reset_index(drop=True)
    
    def detect_and_filter_languages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect languages and filter for target language."""
        pipe = self.setup_language_detection()
        
        # Detect comment languages
        language_results = []
        for batch_start in tqdm(range(0, len(df), self.batch_size), desc="Detecting languages"):
            batch_end = min(batch_start + self.batch_size, len(df))
            batch_texts = df.iloc[batch_start:batch_end]["textDisplay"].tolist()
            batch_texts = [str(text) if text is not None else "" for text in batch_texts]
            
            batch_results = pipe(batch_texts, top_k=1, truncation=True, batch_size=min(self.batch_size, len(batch_texts)))
            batch_languages = [result[0]["label"] for result in batch_results]
            language_results.extend(batch_languages)
        
        df['language'] = language_results
        
        # Filter for target language
        df_filtered = df[df['language'] == self.target_language].copy()
        
        # Validate video languages
        videos = df_filtered[["video_id", "video_title"]].drop_duplicates()
        video_titles = [str(title) if title is not None else "" for title in videos["video_title"].tolist()]
        video_language_results = pipe(video_titles, top_k=1, truncation=True, batch_size=512)
        videos['language'] = [result[0]["label"] for result in video_language_results]
        
        target_video_ids = videos[videos['language'] == self.target_language]['video_id'].tolist()
        df_final = df_filtered[df_filtered['video_id'].isin(target_video_ids)].copy()
        
        return df_final.reset_index(drop=True)
    
    def clean_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete cleaning pipeline.
        
        Args:
            df: Raw DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline")
        
        # Step 1: Text length filtering
        logger.info("Step 1: Filtering by text length")
        df = self.filter_by_text_length(df)
        
        # Step 2: Remove emoji-only comments
        logger.info("Step 2: Removing emoji-only comments")
        df = self.remove_emoji_only_comments(df)
        
        # Step 3: Normalize data structures
        logger.info("Step 3: Normalizing data structures")
        df = self.normalize_data_structures(df)
        
        # Step 4: Remove duplicates
        logger.info("Step 4: Removing duplicates")
        df = self.remove_duplicates(df)
        
        # Step 5: Language detection and filtering
        logger.info("Step 5: Language detection and filtering")
        df = self.detect_and_filter_languages(df)
        
        logger.info(f"Cleaning pipeline completed. Final dataset: {len(df):,} records")
        return df
    
    def export_cleaned_data(self, df: pd.DataFrame, output_path: Path, 
                          include_metadata: bool = True) -> None:
        """Export cleaned dataset with optional metadata."""
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export main dataset
        df.to_parquet(output_path, index=False)
        
        # Export CSV backup
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        if include_metadata:
            # Export metadata
            metadata = {
                'export_info': {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'total_records': len(df),
                    'unique_videos': df['video_id'].nunique(),
                    'target_language': self.target_language
                },
                'cleaning_parameters': {
                    'min_text_percentile': self.min_percentile,
                    'max_text_percentile': self.max_percentile,
                    'batch_size': self.batch_size
                }
            }
            
            import json
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data exported to: {output_path}")

if __name__ == "__main__":
    # Example usage
    print("YouTube Data Cleaning Utilities")
    print("Import this module to use the DataCleaner class")
