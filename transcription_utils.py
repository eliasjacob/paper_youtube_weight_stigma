
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from tqdm import tqdm
import time
import joblib
import os
import logging
from pathlib import Path

class VideoTranscriptDownloader:
    """
    A class to download video transcripts using the youtube_transcript_api.

    Handles checkpointing to resume downloads and processes the transcripts
    into a consolidated format.
    """
    def __init__(self, video_ids, checkpoint_file, sleep_interval=2):
        """
        Initializes the VideoTranscriptDownloader.

        Args:
            video_ids (list): A list of YouTube video IDs to process.
            checkpoint_file (str or Path): Path to the joblib file for saving progress.
            sleep_interval (int): Seconds to wait between requests to avoid rate limiting.
        """
        self.video_ids = video_ids
        self.checkpoint_file = Path(checkpoint_file)
        self.sleep_interval = sleep_interval
        self.ytt_api = YouTubeTranscriptApi()
        self.logger = logging.getLogger(__name__)
        
        # Ensure the directory for the checkpoint file exists
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self):
        """Loads progress from the checkpoint file if it exists."""
        if self.checkpoint_file.exists():
            self.logger.info(f"Resuming from checkpoint: {self.checkpoint_file}")
            return joblib.load(self.checkpoint_file)
        self.logger.info("No checkpoint found. Starting a new download process.")
        return {"success": [], "fail": [], "dataframes": []}

    def _save_checkpoint(self, data):
        """Saves progress to the checkpoint file."""
        joblib.dump(data, self.checkpoint_file)
        self.logger.info(f"Checkpoint saved with {len(data['success'])} successes and {len(data['fail'])} failures.")

    def _get_single_transcript(self, video_id, translate_to_pt=False):
        """
        Fetches and processes a transcript for a single video.
        
        Tries to get the original 'pt' transcript. If not available and
        `translate_to_pt` is True, it attempts to translate the first available one.

        Args:
            video_id (str): The YouTube video ID.
            translate_to_pt (bool): Whether to translate if 'pt' is not original.

        Returns:
            pd.DataFrame or None: A DataFrame with the transcript or None if unavailable.
        """
        try:
            transcript_list = self.ytt_api.list_transcripts(video_id)
            
            # Try to find an original Portuguese transcript
            try:
                transcript = transcript_list.find_manually_created_transcript(['pt', 'pt-BR'])
                self.logger.info(f"Found manually created 'pt' transcript for {video_id}.")
                return pd.DataFrame(transcript.fetch())
            except NoTranscriptFound:
                pass # Continue to check for generated transcripts

            try:
                transcript = transcript_list.find_generated_transcript(['pt', 'pt-BR'])
                self.logger.info(f"Found auto-generated 'pt' transcript for {video_id}.")
                return pd.DataFrame(transcript.fetch())
            except NoTranscriptFound:
                pass # Continue to translation logic if enabled

            # If no 'pt' transcript, translate if requested
            if translate_to_pt:
                self.logger.warning(f"'pt' transcript not found for {video_id}. Attempting translation.")
                for t in transcript_list:
                    return pd.DataFrame(t.translate("pt").fetch())
            
            return None # No 'pt' transcript and translation not requested

        except (NoTranscriptFound, TranscriptsDisabled) as e:
            self.logger.warning(f"Could not retrieve transcript for {video_id}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred for video {video_id}: {e}", exc_info=True)
            return None

    def download_transcripts(self, translate_to_pt=False):
        """
        Downloads transcripts for all video_ids, handling checkpoints.

        Args:
            translate_to_pt (bool): Whether to translate non-Portuguese transcripts.

        Returns:
            pd.DataFrame: A concatenated DataFrame of all successfully fetched transcripts.
        """
        checkpoint = self._load_checkpoint()
        
        processed_videos = set(checkpoint["success"] + checkpoint["fail"])
        videos_to_process = [v for v in self.video_ids if v not in processed_videos]

        if not videos_to_process:
            self.logger.info("All videos have already been processed.")
        else:
            self.logger.info(f"Starting download for {len(videos_to_process)} new videos.")
            for video_id in tqdm(videos_to_process, desc="Fetching Transcripts"):
                try:
                    df = self._get_single_transcript(video_id, translate_to_pt=translate_to_pt)
                    if df is not None and not df.empty:
                        df["video_id"] = video_id
                        checkpoint["dataframes"].append(df)
                        checkpoint["success"].append(video_id)
                    else:
                        checkpoint["fail"].append(video_id)
                except Exception as e:
                    self.logger.error(f"Processing failed for video {video_id}: {e}", exc_info=True)
                    checkpoint["fail"].append(video_id)
                
                # Save checkpoint periodically
                if (len(checkpoint["success"]) + len(checkpoint["fail"])) % 10 == 0:
                    self._save_checkpoint(checkpoint)
                
                time.sleep(self.sleep_interval)

            # Final save after the loop
            self._save_checkpoint(checkpoint)

        self.logger.info(f"Processing complete. Total successes: {len(checkpoint['success'])}, Total failures: {len(checkpoint['fail'])}.")
        
        if not checkpoint["dataframes"]:
            return pd.DataFrame()
            
        return pd.concat(checkpoint["dataframes"], ignore_index=True)

def process_transcripts_to_final_format(df_raw_transcripts, df_video_info):
    """
    Processes raw transcript data into a final, clean DataFrame.

    Args:
        df_raw_transcripts (pd.DataFrame): DataFrame containing raw transcript segments.
        df_video_info (pd.DataFrame): DataFrame with video_id and video_title.

    Returns:
        pd.DataFrame: A DataFrame with one row per video, containing the full
                      concatenated transcript, duration, and title.
    """
    if df_raw_transcripts.empty:
        return pd.DataFrame(columns=["video_id", "transcription", "duration", "video_title"])

    # Ensure correct columns
    df_raw_transcripts.columns = ["text", "start", "duration", "video_id"]

    # Aggregate text and calculate total duration
    agg_df = df_raw_transcripts.sort_values("start").groupby("video_id").agg(
        transcription=("text", " ".join),
        duration=("duration", "sum")
    ).reset_index()

    # Merge with video titles
    final_df = pd.merge(agg_df, df_video_info, on="video_id", how="left")
    
    # Ensure no duplicates
    final_df.drop_duplicates(subset=["video_id"], inplace=True)
    
    return final_df
