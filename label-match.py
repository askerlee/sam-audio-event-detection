from ast import If
import torch
import torchaudio
import torchaudio.functional as F
from core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
from core.audio_visual_encoder.pe import PEAudioFrameOutput
import os
import re
import math
import argparse
from tqdm import tqdm
from typing import Iterable, List, Tuple, Dict, Union, Optional, TextIO
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
import json
from pathlib import Path

Interval = Tuple[float, float, Tuple[float, float], float]   # start, end, (prob_mean, prob_max), dbfs
# start, end, "prob_max/prob_mean/dbfs"
# or 
# start, end, dbfs
Interval_str = Tuple[str, str, str]
# label -> "No events detected" or list of Interval_strs
Events = Dict[str, Union[str, List[Interval_str]]]
# start, end: string "MM:SS" or "%Y-%m-%dT%H:%M:%S" (clef-format output). 
# dbfs, prob_mean, prob_max: float.
SpanStat = namedtuple("SpanStat", ["start", "end", "dbfs", "prob_mean", "prob_max"])
# Reolink camera recordings.
PATTERN1 = re.compile(r"(?P<date>\d{8})[/\\](?P<prefix>.+)-(?P<start>\d{6})-(?P<end>\d{6})\.(mp4|wav)$", re.IGNORECASE)
# Manually recorded videos.
PATTERN2 = re.compile(r"(?P<prefix>.+)_(?P<date>\d{8})_(?P<start>\d{6})\.(mp4|wav)$", re.IGNORECASE)
# ＜08 August 2025＞-＜12-06-24＞-＜kitchen noises＞.mp4
# desc is always either "kitchen noises" or "kitchen noises (sound level meter)".
# The latter are actually PATTERN2 recordings renamed to this format for CDRT submission.
PATTERN3 = re.compile(r"＜(?P<date>\d{2} \w+ \d{4})＞-＜(?P<start>\d{2}-\d{2}-\d{2})＞-＜(?P<desc>.+)＞\.(mp4|wav)$", re.IGNORECASE)

# For phones with stereo recording, convert to mono using mid-side technique.
def stereo_to_mono(wav: torch.Tensor) -> torch.Tensor:
    L, R = wav[0], wav[1]

    # L+R (mid) keeps content common to both channels (often voice).
    mid  = 0.5 * (L + R)
    # L−R (side) keeps content that differs between channels (sometimes wind/ambient/spatial).
    side = 0.5 * (L - R)

    # pick the one with higher RMS energy
    def rms(x): return x.pow(2).mean().sqrt()

    mono = mid if rms(mid) >= rms(side) else side
    mono = mono.unsqueeze(0)  # [1, T]
    return mono

def mmss_to_sec(t: str) -> int:
    # t: "MM:SS" or "MMM:MM:SS"
    m, s = t.split(":")
    return int(m) * 60 + int(s)

def sec_to_mmss(x: int) -> str:
    return f"{x//60:02d}:{x%60:02d}"

def merge_intervals(intervals: Iterable[Interval], time_tolerance: float = 0.0, db_max_gap: float = 0.0) -> List[Interval]:
    xs = sorted((s, e, p, db) for s, e, p, db in intervals)
    merged: List[Interval] = []
    for s, e, p, db in xs:
        if not merged:
            # The first interval.
            merged.append((s, e, p, db))
            continue

        s0, e0, p0, db0 = merged[-1]
        if s <= e0 + time_tolerance and abs(db - db0) <= db_max_gap:
            # merge if the gap is small enough
            # Take the max prob_mean and prob_max of the two intervals, as the new prob.
            merged[-1] = (s0, max(e0, e), max(p0, p), max(db0, db))
        else:
            # the gap is too large, start a new interval.
            merged.append((s, e, p, db))
    return merged


# Extract event segments from event_mask and compute dBFS for each
def extract_event_segments_from_mask(audio: torch.Tensor, mask: torch.Tensor, sample_rate: int,
                                     db_offset: float) -> List[Tuple[float, float]]:
    """
    Extract contiguous True segments from boolean mask and compute dBFS for each.
    Returns list of (dbfs, duration_seconds) tuples.
    """
    segments = []
    if mask.sum() == 0:
        return segments
        
    # Find boundaries where mask changes from False to True or True to False
    padded_mask = torch.cat([torch.tensor([False], device=mask.device), mask, torch.tensor([False], device=mask.device)])
    changes = torch.diff(padded_mask.int())
    
    starts = torch.where(changes == 1)[0]  # False -> True transitions
    ends = torch.where(changes == -1)[0]   # True -> False transitions
    
    for start_idx, end_idx in zip(starts, ends):
        start_idx = start_idx.item()
        end_idx = end_idx.item()
        
        # Extract audio segment
        segment_audio = audio[:, start_idx:end_idx]
        if segment_audio.shape[1] > 0:
            # Compute dBFS for this segment
            segment_dbfs = calc_dbfs(segment_audio, db_offset, search_peak_window_len=-1)
            duration_seconds = (end_idx - start_idx) / sample_rate
            segments.append((segment_dbfs, duration_seconds))
    
    return segments

def merge_events_along_timeline(all_events: Events, start_date_obj: Optional[datetime] = None, 
                                time_tolerance: int = 0, db_max_gap: float = 0.0,
                                output_cdrt_transcripts: bool = False,
                                dvd_label: Optional[str] = None, file_trunk: Optional[str] = None,
                                cdrt_transcript_fh: Optional[TextIO] = None,
                                debug=False) -> None:
    """
    Prints segments in time order.
    If events overlap in time, their labels appear together in that segment.
    time_tolerance expands each interval by +/- time_tolerance seconds.
    db_max_gap: float, the max gap in dBFS for merging the same event.
    merge_events_along_timeline uses a time_tolerance <= the time_tolerance used in merging events above.
    Otherwise, the overlap timeline merging algorithm may not work as expected.
    """
    # Whether a second is associated with multiple labels
    seconds_painted_with_labels = defaultdict(dict)  # second -> dict(label: prob)
    # seconds_painted_with_labels2: seconds_painted_with_labels extended with time tolerance at both ends.
    seconds_painted_with_labels2 = defaultdict(dict)  # second -> dict(label: prob)
    seconds_are_labels_in_extended_regions = defaultdict(dict) # second -> dict(label: bool)
    file_trunk = file_trunk.replace("＜", "<").replace("＞", ">") if file_trunk else None

    # 1) paint timeline per second (inclusive endpoints)
    for label, spans in all_events.items():
        if not spans:
            continue
        for span_stat in spans:
            a, b = span_stat.start, span_stat.end
            s, e = mmss_to_sec(a), mmss_to_sec(b)
            assert e >= s, f"Invalid interval: {a}-{b}"
            for t in range(s, e + 1):
                seconds_painted_with_labels[t][label] = span_stat

            # Only add tolerance to the start time, not the end time, 
            # to avoid adding extra seconds to the end of each event.
            # Since time_tolerance is intentionally set to be <= the time_tolerance used in merging events above,
            # the gaps between adjacent events of the same label are always >= time_tolerance.
            # Thus, we are sure there is no event of this label in the interval [s - time_tolerance, s - 1].
            # Setting seconds_painted_with_labels2[t][label] backwards at [s - time_tolerance, ... s - 1]
            # won't accidentally overwrite a previous event of the same type.
            s2 = max(0, s - time_tolerance)
            for t in range(s2, e + 1):
                seconds_painted_with_labels2[t][label] = span_stat
            for t in range(s2, s):
                seconds_are_labels_in_extended_regions[t][label] = True

    if not seconds_painted_with_labels:
        return

    # 2) compress consecutive seconds with same label-set
    t0 = min(seconds_painted_with_labels.keys())
    t1 = max(seconds_painted_with_labels2.keys())

    seg_start = t0
    curr_labels = seconds_painted_with_labels.get(t0, dict())

    def print_seg_labels(seg_start: int, seg_end: int, label2span_stat: dict, 
                         start_date_obj: Optional[datetime] = None):
        if label2span_stat:
            if debug:
                labels_str = ", ".join(f"{label} ({span_stat.prob_max:.2f}/{span_stat.prob_mean:.2f}/{span_stat.dbfs:.1f}db)" for label, span_stat in sorted(label2span_stat.items()))
            else:
                labels_str = ", ".join(f"{label} ({span_stat.dbfs:.1f}db)" for label, span_stat in sorted(label2span_stat.items()))
            if start_date_obj:
                seg_start_dt = start_date_obj + timedelta(seconds=seg_start)
                seg_end_dt   = start_date_obj + timedelta(seconds=seg_end)
                abs_time = f"{seg_start_dt.strftime('%H:%M:%S')}-{seg_end_dt.strftime('%H:%M:%S')}"
                print(f"{abs_time}/{sec_to_mmss(seg_start)}-{sec_to_mmss(seg_end)}: {labels_str}")
            else:
                print(f"{sec_to_mmss(seg_start)}-{sec_to_mmss(seg_end)}: {labels_str}")

            if output_cdrt_transcripts and cdrt_transcript_fh:
                # We always output relative time, regardless of whether start_date_obj is given.
                # Since csv uses "," as separator, we use "; " to separate multiple events.
                # TODO: Not sure if we should output dbfs in the transcript. 
                # The issue is, if we don't output dbfs, then multiple events of the same type
                # that are split due to disparate dbfs values will appear as consecutive events of the same type
                # without apparent distinction, which may look confusing to examiners.
                # Example:
                # 00:01:52 to 00:01:55,kitchen clatter and clank
                # 00:01:56 to 00:01:57,kitchen clatter and clank
                # If we output dbfs, then the two events will be:
                # 00:01:52 to 00:01:55,kitchen clatter and clank (72.5db)
                # 00:01:56 to 00:01:57,kitchen clatter and clank (60.0db)
                # The examiner can then see the difference and may better understand why the events are split.
                event_transcript = "; ".join(f"{label} ({span_stat.dbfs:.1f}db)" for label, span_stat in sorted(label2span_stat.items()))
                cdrt_transcript_fh.write(f"{dvd_label},{file_trunk},00:{sec_to_mmss(seg_start)} to 00:{sec_to_mmss(seg_end)},{event_transcript}\n")

    for t in range(t0 + 1, t1 + 2):  # +2 to flush the last segment
        # Get labels from seconds_painted_with_labels2 (with time tolerance).
        new_labels = seconds_painted_with_labels2.get(t, dict())
        # If no event is detected at t, new_labels will be empty set.
        # The condition below will hold and the current segment will be printed.
        if new_labels.keys() & curr_labels.keys() == set():
            # If new_labels and curr_labels contain no common elements, 
            # print and start a new segment
            print_seg_labels(seg_start, t - 1, curr_labels, start_date_obj=start_date_obj)
            seg_start = t
            curr_labels = seconds_painted_with_labels.get(t, dict())
        else:
            merged_labels = {}
            mergeable = True
            # If new_labels and curr_labels share some elements, 
            # try to merge the label dicts and continue.
            are_labels_in_extended_region = seconds_are_labels_in_extended_regions.get(t, dict())

            for label, span_stat in new_labels.items():
                if label in curr_labels:
                    # For the shared label, update to the max prob_mean, prob_max, dbfs.
                    existing_stat = curr_labels[label]
                    is_label_in_extended_region = are_labels_in_extended_region.get(label, False)
                    if (abs(existing_stat.dbfs - span_stat.dbfs) > db_max_gap) and (not is_label_in_extended_region):
                        # dBFS difference too large, and both spans are not in extended regions, 
                        # so we cannot merge this label. If a label is not mergeable, 
                        # then these two spans are not mergeable.
                        mergeable = False
                        #breakpoint()
                        break
                    # start and end don't matter, as we only use dbfs, prob_mean, prob_max for printing.
                    # So start and end are simply taken from existing_stat.
                    merged_stat = SpanStat(
                        start=existing_stat.start,
                        end=existing_stat.end,
                        dbfs=max(existing_stat.dbfs, span_stat.dbfs),
                        prob_mean=max(existing_stat.prob_mean, span_stat.prob_mean),
                        prob_max=max(existing_stat.prob_max, span_stat.prob_max),
                    )
                    merged_labels[label] = merged_stat
                else:
                    merged_labels[label] = span_stat

            if not mergeable:
                # Cannot merge due to dBFS difference too large
                print_seg_labels(seg_start, t - 1, curr_labels, start_date_obj=start_date_obj)
                seg_start = t
                curr_labels = seconds_painted_with_labels.get(t, dict())
            else:
                curr_labels = merged_labels
                    
def split_waveform(wav: torch.Tensor, seg_samples: int):
    """
    wav: (channels, samples)
    yields: (segment_tensor, valid_len)
      - segment_tensor always has seg_samples (last chunk is zero-padded)
      - valid_len is the non-padded length in samples (<= seg_samples)
    """
    c, total = wav.shape
    for start in range(0, total, seg_samples):
        end = min(start + seg_samples, total)
        chunk = wav[:, start:end]
        valid_len = end - start
        if valid_len < seg_samples:
            pad = torch.zeros((c, seg_samples - valid_len), dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=1)
        yield chunk, valid_len

def PEAudioFrame_forward(
    self,
    input_ids: torch.Tensor,  # tokenized text
    input_values: Optional[torch.Tensor] = None,  # audio waveform (may be None if input_features is provided)
    input_features: Optional[torch.Tensor] = None,  # codec_features (if already computed)
    attention_mask: Optional[torch.Tensor] = None,  # text attention mask
    padding_mask: Optional[torch.Tensor] = None,  # audio padding mask
    threshold: float = 0.3,
    return_spans: bool = True,
    return_span_probs: bool = False,
) -> PEAudioFrameOutput:
    audio_output = self.audio_model(input_values, padding_mask, input_features=input_features)
    text_model_output = self._get_text_output(input_ids, attention_mask)

    text_embeds = self.text_head(text_model_output.pooler_output)
    audio_embeds = self.audio_head(audio_output.last_hidden_state)

    spans = None
    if return_spans:
        bsz = input_ids.size(0)
        unscaled_logits = audio_embeds @ text_embeds.unsqueeze(1).transpose(-1, -2)
        logits = unscaled_logits.squeeze(-1) * self.logit_scale + self.logit_bias
        probs = logits.sigmoid()

        preds = probs > threshold
        # Find where predictions changed from False->True and True->False
        changes = torch.diff(torch.nn.functional.pad(preds, (1, 1), value=False), dim=1).nonzero()
        span_tensor = torch.cat([changes[::2], changes[1::2, [1]]], dim=1)
        # Convert audio frame index to time
        dac_config = self.config.audio_model.dac_vae_encoder
        sec_per_frame = dac_config.hop_length / dac_config.sampling_rate

        spans = []
        # Each instance in the batch corresponds to one sublist in spans.
        for i in range(bsz):
            # (nspans, 2) start,end in frames
            # [..., 1:] to drop batch index, since the batch index is implicitly i.
            spans_i = span_tensor[span_tensor[:, 0] == i, 1:]  

            if spans_i.numel() == 0:
                # If no spans detected, return empty list for this instance.
                spans.append([])
                continue

            if return_span_probs:
                out_i = []
                for s_f, e_f in spans_i.tolist():
                    s_f = int(s_f)
                    e_f = int(e_f)
                    span_p = (probs[i, s_f:e_f].mean().item() if e_f > s_f else probs[i, s_f].item(),
                              probs[i, s_f:e_f].max().item()  if e_f > s_f else probs[i, s_f].item())
                    out_i.append([s_f * sec_per_frame, e_f * sec_per_frame, span_p])
                spans.append(out_i)
            else:
                spans.append((spans_i.float() * sec_per_frame).tolist())

    return PEAudioFrameOutput(
        text_embeds=text_embeds,
        audio_embeds=audio_embeds,
        spans=spans,
        text_output=text_model_output,
        audio_output=audio_output,
    )

# Assume wav has been demean'ed.
def calc_dbfs(wav: torch.Tensor, db_offset: float, search_peak_window_len: float=-1) -> float:
    rms = torch.sqrt(torch.mean(wav**2))
    dbfs = 20.0 * torch.log10(torch.clamp(rms, min=1e-12)) + db_offset

    if search_peak_window_len > 0:
        # Search for peak dBFS in sliding windows of length search_peak_window_len (in samples)
        wav_len = wav.shape[-1]
        window_len = int(search_peak_window_len)
        hop_len = window_len // 2
        dbfs_values = []
        for start in range(0, wav_len, hop_len):
            end = start + window_len
            if end > wav_len:
                start = wav_len - window_len
            w = wav[:, start:end]
            rms = torch.sqrt(torch.mean(w**2))
            dbfs = 20.0 * torch.log10(torch.clamp(rms, min=1e-12)) + db_offset
            dbfs_values.append(dbfs)

        # Take the average of the top 20% highest dBFS values as the peak dBFS.
        top_20_percent_count = max(1, len(dbfs_values) // 5)
        top_20_percent_values = sorted(dbfs_values, reverse=True)[:top_20_percent_count]
        dbfs_peak = sum(top_20_percent_values) / len(top_20_percent_values)
        return dbfs.item(), dbfs_peak.item()
    else:
        return dbfs.item()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# -------------------- args --------------------
parser = argparse.ArgumentParser(description="Audio Source Separation using SAM-Audio (chunked + tqdm)")
parser.add_argument("--model_size", type=str, default="large", choices=["small", "medium", "large"], help="Model size to use")
parser.add_argument("--input_folders", type=str, nargs='*',
                    default=None, help="Paths to the input folders containing audio files")
parser.add_argument("--input_files", type=str, nargs='*', default=None, help="Paths to the input audio file (.wav/.mp4/...)")
parser.add_argument("--segment_seconds", type=float, default=120.0, help="Chunk length in seconds (e.g. 5, 10, 30)")
parser.add_argument("--sample_rate", type=int, default=48000, help="Target sample rate for event detection")
parser.add_argument("--weak_thres_discount", type=float, default=0.8, 
                    help="Discount factor for weak event detection threshold")
# db_offset has been calibrated by comparing dBFS values measured by the DB meter and those calculated here.
parser.add_argument("--default_db_offset", type=float, default=85, 
                    help="Default dBFS offset to add to all detected events' dBFS, if unable to be inferred from the audio type")
parser.add_argument("--loud_db_offset", type=float, default=8.0, help="dB offset above average event dBFS to consider an event 'loud'")
parser.add_argument("--search_peak_window_sec", type=float, default=0.2, 
                    help="Window length in seconds to search for peak dBFS within an event span (set to -1 to disable)")
parser.add_argument("--db_max_gap", type=float, default=5.0,
                    help="Max dBFS gap to merge same event types along timeline")
parser.add_argument("--abs_time", type=str2bool, nargs='?', const=True, default=False, 
                    help="Whether to print absolute time for timestamps")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output results")
parser.add_argument("--output_clef", type=str2bool, nargs='?', const=True, default=False, 
                    help="Whether to output CLEF-format results for Seq visualization")
parser.add_argument("--output_cdrt_transcripts", type=str2bool, nargs='?', const=True, default=False, 
                    help="Whether to output CDRT-format transcripts for submission")
parser.add_argument("--dvd_label_mapping_file", type=str, default=None,
                    help="Path to DVD label mapping .txt file (if outputting CDRT-format transcripts)")
parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True, help="Whether to print debug info")

def analyze_audio_labels(model: PEAudioFrame, transform: PEAudioFrameTransform, 
                         input_file: str, segment_seconds: float, sample_rate: int, 
                         weak_thres_discount: float, default_db_offset: float, loud_db_offset: float, 
                         desc2det_thres: Dict[str, float], desc2db_thres: Dict[str, float],
                         search_peak_window_sec: float,
                         device: str, print_abs_time: bool, output_cdrt_transcripts: bool, 
                         dvd_label_mapping: Optional[Dict[str, str]], 
                         cdrt_transcript_fh: Optional[TextIO],
                         debug: bool):
    """
    Analyze audio file for specific event labels and print detected events with timestamps.
    output_cdrt_transcripts: bool, whether to output CDRT-format transcripts for submission.
    If output_cdrt_transcripts is True, the output will be saved to a .csv file, in the format of:
    CD or DVD label, File name of recording, 
    Time location within recording in <HH:MM:SS> to <HH:MM:SS> format, Event Transcript
    https://www.judiciary.gov.sg/civil/prepare-evidence-neighbour-dispute-claim
    """

    # input_file is the full path (relative or absolute) to the audio file.
    # Define audio file and event descriptions
    dvd_label, file_trunk = None, None
    is_meter_video = False
    # Try to extract start time from filename
    m1 = PATTERN1.search(input_file) # Reolink camera recordings.
    m2 = PATTERN2.search(input_file) # Manually recorded videos by phones.
    m3 = PATTERN3.search(input_file) # CDRT submissions.
    if m1:
        start_date = m1.group("date")
        start_time = m1.group("start")
        # 10262025
        month, day, year = int(start_date[0:2]), int(start_date[2:4]), int(start_date[4:8])
        hr, minute, second = int(start_time[0:2]), int(start_time[2:4]), int(start_time[4:6])
        start_date_obj = datetime(year, month, day, hr, minute, second)
        # dB offset for Reolink cameras, calibrated against DB meter readings.
        db_offset = 85
    elif m2:
        start_date = m2.group("date")
        start_time = m2.group("start")
        # 20251026
        year, month, day = int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8])
        hr, minute, second = int(start_time[0:2]), int(start_time[2:4]), int(start_time[4:6])
        start_date_obj = datetime(year, month, day, hr, minute, second)
        # db offset for phone recordings, calibrated against DB meter readings.
        # Phones have less sensitive microphones, so we use a higher dB offset.
        db_offset = 95
        is_meter_video = True
    elif m3:
        start_date = m3.group("date")
        start_time = m3.group("start")
        # date: "08 August 2025"
        date_obj = datetime.strptime(start_date, "%d %B %Y")
        year, month, day = date_obj.year, date_obj.month, date_obj.day
        hr, minute, second = map(int, start_time.split("-"))
        start_date_obj = datetime(year, month, day, hr, minute, second)
        if m3.group("desc").lower() == "kitchen noises":
            # db offset for phone recordings without sound level meter app, calibrated against DB meter readings.
            db_offset = 85
        elif m3.group("desc").lower() == "kitchen noises (sound level meter)":
            # db offset for phone recordings with sound level meter app, calibrated against DB meter readings.
            db_offset = 95
            is_meter_video = True
        else:
            db_offset = default_db_offset
            breakpoint() # SHouldn't reach here.

        if output_cdrt_transcripts:
            parent_dir = Path(input_file).parent
            file_trunk = Path(input_file).stem
            dvd_label = dvd_label_mapping[parent_dir.name]
    else:
        print(f"Warning: Could not extract start time from filename {os.path.basename(input_file)}. Output relative time instead.")
        print_abs_time = False
        start_date_obj = None
        db_offset = default_db_offset

    # -------------------- load + resample once --------------------
    # NOTE: here we directly load the mp4 file using torchaudio.load(), and get the audio track.
    # If your torchaudio has an FFmpeg backend available, torchaudio.load("video.mp4") 
    # can load the audio track from an MP4.
    # If you only have the SoX backend, it generally won’t load MP4 
    # (SoX doesn’t handle MP4 containers by default).
    wav, sr = torchaudio.load(input_file)  # wav: (channels, samples)
    # Video/Audio recorded by phones.
    if wav.shape[0] == 2:
        wav = stereo_to_mono(wav)
    seg_samples = max(1, int(round(segment_seconds * sr)))
    # total_samples: total points in wav.
    total_samples = wav.shape[1]
    num_segments = max(1, math.ceil(total_samples / seg_samples))

    print(f"Loaded: {input_file} | sr={sr} | chunk={segment_seconds}s ({seg_samples} samples) | segments={num_segments}")
  
    # -------------------- process each description over chunks --------------------
    pbar = tqdm(
        split_waveform(wav, seg_samples),
        total=num_segments,
        desc="Matching by Segments",
        unit="seg",
        dynamic_ncols=True,
    )

    wav_demeaned = wav - wav.mean()
    chunks_demeaned, valid_lens_demeaned = zip(*list(split_waveform(wav_demeaned, seg_samples)))

    t0 = 0
    desc2intervals = {desc: [] for desc in desc2det_thres.keys()}
    event_db_mask = torch.zeros(total_samples, dtype=float).to(device)

    for chunk, valid_len in pbar:
        chunk_rs = F.resample(chunk, orig_freq=sr, new_freq=sample_rate)
        chunk_demeaned = chunks_demeaned[pbar.n]
        
        # Process inputs
        inputs = transform(audio=[chunk_rs], text=list(desc2det_thres.keys()), sampling_rate=sample_rate).to(device)
        # inputs.input_values have been normalized to [-1.02, 1.02] in the transform.
        # Run inference
        with torch.inference_mode():
            # Discount the minimal threshold a bit to allow "weak (less confident) events" to be detected 
            # and later merged with adjacent "strong (confident) events".
            outputs = model(**inputs, return_spans=True, threshold=min(desc2det_thres.values()) * weak_thres_discount, 
                            return_span_probs=True)

        # outputs: PEAudioFrameOutput
        # audio_embeds, text_embeds, spans, audio_output, text_output
        # Print detected time spans for each event
        for description, spans in zip(desc2det_thres.keys(), outputs.spans):
            if spans:
                # Discount the threshold to allow weak events to pass.
                # Filter out events below the discounted threshold here.
                # p: (prob_mean, prob_max). p[0] is prob_mean.
                for s, e, p in spans:
                    if p[1] < desc2det_thres[description] * weak_thres_discount:
                        continue
                    # NOTE: chunk_rs has been resampled to args.sample_rate.
                    # But we compute start/end, dbfs in the original chunk.
                    start_s = float(s)
                    end_s   = float(e)
                    start = int(start_s * sr)
                    end   = min(int(end_s   * sr), valid_len)
                    global_start_s = start_s + t0
                    global_start   = int(global_start_s * sr)
                    global_end_s   = end_s   + t0
                    global_end     = int(global_end_s * sr)
                    # Take the span from the original chunk (not resampled).
                    span_sound = chunk_demeaned[:, start:end]
                    dbfs, dbfs_peak = calc_dbfs(span_sound, db_offset, search_peak_window_len=search_peak_window_sec * sr)
                    # NOTE: The loudest part usually determines how disturbing the event is. 
                    # Also the loudest part will linger in attention for a short while 
                    # (usually long enough to last through the whole span, and only fades out till much later).
                    # Therefore, we use peak dBFS as the event's dBFS, 
                    # to better reflect the actual psychological effect of the noise.
                    dbfs = dbfs_peak
                    # Ignore events below their dBFS thresholds.
                    # If in debug mode, keep all events for analysis.
                    if (not debug) and (dbfs < desc2db_thres[description]):
                        continue
                    # event_db_mask corresponds to the original wav's timeline.
                    event_db_mask[global_start: global_end] = torch.maximum(event_db_mask[global_start: global_end], torch.tensor(dbfs, device=device))
                    desc2intervals[description].append((global_start_s, global_end_s, p, dbfs))

        t0 += segment_seconds

    all_events = {}

    for description in desc2det_thres.keys():
        # Merge the same type of events, if their time gap <= 2 seconds, and the dBFS difference <= 3.0 dB.
        num_merges = 0
        while True:
            spans = merge_intervals(desc2intervals[description], time_tolerance=2, db_max_gap=args.db_max_gap)
            num_merges += 1
            desc2intervals[description] = spans
            if len(spans) == len(desc2intervals[description]):
                # No more merges
                break

        if spans:
            span_stats = []
            for start, end, (prob_mean, prob_max), dbfs in spans:
                # Filter weak events below threshold.
                if prob_max < desc2det_thres[description]:
                    continue
                start = round(start)
                end   = round(end)
                start_min = start // 60
                start_sec = start %  60
                end_min   = end   // 60
                end_sec   = end   %  60

                span_stat = SpanStat(
                    start=f"{start_min:02d}:{start_sec:02d}",
                    end=f"{end_min:02d}:{end_sec:02d}",
                    dbfs=dbfs,
                    prob_mean=prob_mean,
                    prob_max=prob_max,
                )
                span_stats.append(span_stat)

            all_events[description] = span_stats

    if len(all_events) == 0:
        print("No events detected in the audio.")
        return {}, None, 0, False
    
    # merge_events_along_timeline uses a time_tolerance <= the time_tolerance used in merging events above.
    # Otherwise, the overlap timeline merging algorithm may discard some overlapped events.
    merge_events_along_timeline(all_events, start_date_obj=start_date_obj if print_abs_time else None,
                                time_tolerance=2, db_max_gap=args.db_max_gap, 
                                output_cdrt_transcripts=output_cdrt_transcripts,
                                dvd_label=dvd_label, file_trunk=file_trunk, cdrt_transcript_fh=cdrt_transcript_fh,
                                debug=debug)
    wav_demeaned = wav_demeaned.to(device)
    avg_dbfs = calc_dbfs(wav_demeaned, db_offset, search_peak_window_len=-1)
    print(f"Average Audio dBFS: {avg_dbfs:.1f}db")
    if event_db_mask.sum() == 0:
        print("No events detected, skipping event dBFS analysis.")
        return {}, None, 0, False
    
    nonevent_mask = (event_db_mask == 0)
    all_nonevent_dbfs = extract_event_segments_from_mask(wav_demeaned, nonevent_mask, sr, db_offset)
    avg_nonevent_dbfs = sum(dbfs * dur for dbfs, dur in all_nonevent_dbfs) / sum(dur for _, dur in all_nonevent_dbfs)
    print(f"Average Non-Event dBFS: {avg_nonevent_dbfs:.1f}db")

    '''
    event_db_mask_s = torch.nn.functional.interpolate(
        event_db_mask.unsqueeze(0).unsqueeze(0), 
        scale_factor=1.0/sr,
        mode='linear',
    ).squeeze()
    # breakpoint()
    '''

    avg_event_dbfs = event_db_mask[event_db_mask > 0].mean().item()
    print(f"Average Event dBFS: {avg_event_dbfs:.1f}db")
    print(f"Duration of All Events:                  {(event_db_mask > 0).sum().item() / sr:.1f} seconds")
    if (event_db_mask > avg_nonevent_dbfs + loud_db_offset).sum() == 0:
        print(f"No loud events (> avg + {loud_db_offset}dB) detected.")
    else:
        print(f"Duration of Loud Events (> avg + {loud_db_offset}dB): {(event_db_mask > avg_nonevent_dbfs + loud_db_offset).sum().item() / sr:.1f} seconds")
        avg_loud_dbfs = event_db_mask[event_db_mask > avg_nonevent_dbfs + loud_db_offset].mean().item()
        print(f"Avg dBFS of Loud Events: {avg_loud_dbfs:.1f}db = Average + {avg_loud_dbfs - avg_nonevent_dbfs:.1f}db")

    if start_date_obj:
        # Adjust all_events to absolute time
        # all_events: label -> list of SpanStat
        for label, spans in all_events.items():
            if not spans:
                continue
            for i in range(len(spans)):
                span_stat = spans[i]
                s_sec = mmss_to_sec(span_stat.start)
                e_sec = mmss_to_sec(span_stat.end)
                s_dt = start_date_obj + timedelta(seconds=s_sec)
                # The detected end time is inclusive. Seq visualization calculates span length as (end - start).
                # So we add 1 second to make the calculated span length correct.
                e_dt = start_date_obj + timedelta(seconds=e_sec + 1)
                # Extend start and end to absolute time format.
                spans[i] = SpanStat(
                    start=s_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    end=e_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    dbfs=span_stat.dbfs,
                    prob_mean=span_stat.prob_mean,
                    prob_max=span_stat.prob_max,
                )
    return all_events, start_date_obj, avg_nonevent_dbfs, is_meter_video
    
if __name__ == "__main__":
    args = parser.parse_args()
    device = "cuda"

    input_files = []
    if args.input_folders:
        # Process all mp4 files in the input folders
        for folder in args.input_folders:
            root = Path(folder)
            # Don't include .wav files, as they are converted from .mp4 files.
            # We don't count the .wav files to avoid duplicate processing.
            files = sorted(root.rglob("*.mp4"))
            for input_file in files:
                input_files.append(str(input_file))

    elif args.input_files:
        input_files.extend(args.input_files)
    else:
        print("Error: Please provide either --input_folder or --input_file.")
        exit(1)

    if args.output_cdrt_transcripts:
        if args.dvd_label_mapping_file is None:
            print("Error: Please provide --dvd_label_mapping_file when --output_cdrt_transcripts is set to True.")
            exit(1)
        # Load DVD label mapping
        dvd_label_mapping = {}
        with open(args.dvd_label_mapping_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    print(f"Warning: Invalid line in DVD label mapping file: {line}")
                    continue
                folder_name, dvd_label = parts
                dvd_label_mapping[folder_name] = dvd_label
        cdrt_transcript_header = "CD or DVD label,File name of recording,Time location within recording,Transcript"
        CDRT_TRANSCRIPT = open(os.path.join(args.output_dir, "cdrt_transcripts.csv"), "w")
    else:
        dvd_label_mapping = None
        CDRT_TRANSCRIPT   = None

    # Load the model and transform.
    # The 'large' model takes 30GB GPU memory for inference with batch size 1.
    # It takes 40 minutes to process all 167 reolink recordings on a single NVIDIA RTX 6000 Ada GPU.
    # The 'large' model is recommended for best accuracy.
    model = PEAudioFrame.from_config(f"pe-a-frame-{args.model_size}", pretrained=True).to(device)
    transform = PEAudioFrameTransform.from_config(f"pe-a-frame-{args.model_size}")

    # Patch the forward method to return span probabilities.
    model.forward = PEAudioFrame_forward.__get__(model, PEAudioFrame)

    # Threshold for detection confidence per description
    desc2det_thres = {
        "open and close closet": 0.55,
        "door creak or squeak": 0.55,
        "door slam": 0.65,
        "chopstick clatter": 0.5,
        "ceramic dish clatter": 0.5,
        "kitchen clatter and clank": 0.5,
        "thud or thump": 0.6,
        "clack and clunk": 0.6,
        "chopping or cutting": 0.5,
        "human talking": 0.65,
        "oil sizzle": 0.6,
        "water splattering": 0.5,
        "plastic bag rustle": 0.5,
    }

    # dBFS threshold per description
    desc2db_thres = {
        "open and close closet": 55,
        "door creak or squeak": 55,
        "door slam": 55,
        "chopstick clatter": 55,
        "ceramic dish clatter": 55,
        "kitchen clatter and clank": 55,
        "thud or thump": 55,
        "clack and clunk": 55,
        "chopping or cutting": 55,
        # Noises below are generally softer, but still disturbing.
        # So we set a lower dBFS threshold to catch more of them.
        "human talking": 50,
        "oil sizzle": 50,
        "water splattering": 50,
        "plastic bag rustle": 50,
    }

    for input_file in tqdm(input_files):
        print(f"\nAnalyzing audio file: {input_file}")
         # Analyze audio labels
        all_events, start_date_obj, avg_nonevent_dbfs, is_meter_video = \
            analyze_audio_labels(
                model=model,
                transform=transform,
                input_file=input_file,
                segment_seconds=args.segment_seconds,
                sample_rate=args.sample_rate,
                weak_thres_discount=args.weak_thres_discount,
                default_db_offset=args.default_db_offset,
                loud_db_offset=args.loud_db_offset,
                desc2det_thres=desc2det_thres,
                desc2db_thres=desc2db_thres,
                search_peak_window_sec=args.search_peak_window_sec,
                device=device,
                print_abs_time=args.abs_time,
                output_cdrt_transcripts=args.output_cdrt_transcripts,
                dvd_label_mapping=dvd_label_mapping,
                cdrt_transcript_fh=CDRT_TRANSCRIPT,
                debug=args.debug,
            )

        if args.output_clef and all_events:
            # Output CLEF-format results for Seq visualization
            output_file = start_date_obj.strftime("%Y%m%d_%H%M%S_events.clef") if start_date_obj else \
                          os.path.basename(input_file).rsplit(".", 1)[0] + "_events.clef"
            output_file = os.path.join(args.output_dir, output_file)

            with open(output_file, "w") as f:
                for label, spans in all_events.items():
                    for span_stat in spans:
                        # {"@t":"1970-01-01T00:00:01+08:00","@st":"1970-01-01T00:00:00+08:00",
                        # "@tr":"00000000000000000000000000000001","@sp":"0000000000000001",
                        # "@l":"Information","@mt":"Detected {Event} ({Prob1}/{Prob2}/{Db} dB)",
                        # "Event":"chopstick clatter","Prob1":0.55,"Prob2":0.45,"Db":58.6,
                        # "SegmentStart":"00:00","SegmentEnd":"00:00"}
                        f.write(json.dumps({
                            "@t": span_stat.end + "+08:00",
                            "@st": span_stat.start + "+08:00",
                            "@tr": "0" * 31 + "1",  # using an arbitrary 32-bit trace ID
                            "@sp": "0" * 15 + "1",  # using an arbitrary 16-bit span ID
                            "@l": "Information",
                            "@mt": "Detected {Event} ({Prob1}/{Prob2}/{Db}/{DbDelta} dB)",    # template for Seq to understand
                            "Event": label,
                            "Prob1": round(span_stat.prob_max, 3),
                            "Prob2": round(span_stat.prob_mean, 3),
                            "Db": round(span_stat.dbfs, 1),
                            "DbDelta": round(span_stat.dbfs - avg_nonevent_dbfs, 1),
                            "SegmentStart": span_stat.start[-5:],
                            "SegmentEnd": span_stat.end[-5:],
                            "IsMeterVideo": is_meter_video,
                        }) + "\n")

            print(f"CLEF-format results written to {output_file}")

