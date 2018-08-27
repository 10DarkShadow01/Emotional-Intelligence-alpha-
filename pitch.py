import librosa
import numpy as np
import math

def ratio_to_cents(r): 
  return 1200.0 * np.log2(r)

def ratio_to_cents_protected(f1, f2): 

  out = np.zeros_like(f1)
  key = (f1!=0.0) * (f2!=0.0)
  out[key] = 1200.0 * np.log2(f1[key]/f2[key])
  out[f1==0.0] = -np.inf
  out[f2==0.0] = np.inf
  out[(f1==0.0) * (f2==0.0)] = 0.0
  return out

def cents_to_ratio(c): 
  return np.power(2, c/1200.0)

def freq_to_midi(f): 
  return 69.0 + 12.0 * np.log2(f/440.0)

def midi_to_freq(m): 
  return np.power(2, (m-69.0)/12.0) * 440.0 if m!= 0.0 else 0.0

def bin_to_freq(b, sr, n_fft):
  return b * float(sr) / float(n_fft)

def freq_to_bin(f, sr, n_fft): 
  return np.round(f/(float(sr)/float(n_fft))).astype('int')

def ppitch(y, sr=44100, n_fft=8820, win_length=1024, hop_length=2048, 
  num_peaks=20, num_pitches=3, min_peak=2, max_peak=743, min_fund=55.0, 
  max_fund=1000.0, harm_offset=0, harm_rolloff=0.75, ml_width=25, 
  bounce_width=0, bounce_hist=0, bounce_ratios=None, max_harm=32767, 
  npartial=7):

  # other params will go here with update (sept 2018)
  # go to time-freq

  if_gram, D = librosa.core.ifgram(y, sr=sr, n_fft=n_fft, 
    win_length=win_length,hop_length=hop_length)
	
  peak_thresh = 1e-3

  min_bin = 2
  max_bin = freq_to_bin(float(max_peak), sr, n_fft)

  if npartial is not None:
    harm_rolloff = math.log(2, float(npartial))

  num_bins, num_frames = if_gram.shape

  pitches = np.zeros([num_frames, num_pitches])       
  peaks = np.zeros([num_frames, num_peaks])           
  fundamentals = np.zeros([num_frames, num_pitches])  
  confidences = np.zeros([num_frames, num_pitches])   

  for i in range(num_frames):

    frqs = if_gram[:,i]
    mags = np.abs(D[:,i])
    total_power = mags.sum()
    max_amp = np.max(mags)

    lower  = mags[(min_bin-1):(max_bin)]
    middle = mags[(min_bin)  :(max_bin+1)]
    upper  = mags[(min_bin+1):(max_bin+2)]

    peaks_mask_all = (middle > lower) & (middle > upper)

    zeros_left = np.zeros(min_bin)
    zeros_right = np.zeros(num_bins - min_bin - max_bin + 1)
    peaks_mask_all = np.concatenate((zeros_left, peaks_mask_all, zeros_right)) 

    peaks_mags_all = peaks_mask_all * mags
    top20 = np.argsort(peaks_mags_all)[::-1][0:num_peaks]

    peaks_frqs = frqs[top20]
    peaks_mags = mags[top20]

    num_peaks_found = top20.shape[0]

    min_freq = float(min_fund)
    max_freq = float(max_fund)

    def b2f(index): return min_freq * np.power(np.power(2, 1.0/48.0), index)

    max_histo_bin = int(math.log(max_freq / min_freq, math.pow(2, 1.0/48.0))) 

    histo = np.fromfunction(lambda x,y: b2f(y), (num_peaks_found, max_histo_bin))
  
    frqs_tile = np.tile(peaks_frqs, (max_histo_bin,1)).transpose()
    mags_tile = np.tile(peaks_mags, (max_histo_bin,1)).transpose()

    def ml_a(amp): 
      return np.sqrt(np.sqrt(amp/max_amp))

    def ml_t(r1, r2):
      max_dist = ml_width 
      cents = np.abs(ratio_to_cents_protected(r1,r2))
      dist = np.clip(1.0 - (cents / max_dist), 0, 1)
      return dist 

    def ml_i(nearest_multiple): 
      out = np.zeros_like(nearest_multiple)
      out[nearest_multiple.nonzero()] = 1/np.power(nearest_multiple[nearest_multiple.nonzero()], harm_rolloff) 
      return out

    ml = (ml_a(mags_tile) * \
      ml_t((frqs_tile/histo), (frqs_tile/histo).round()) * \
      ml_i((frqs_tile/histo).round())).sum(axis=0)

    num_found = 0
    maybe = 0
    found_so_far = []
    bounce_list = list(np.ravel(pitches[i-bounce_hist:i]))
    bounce_ratios = [1.0, 2.0, 3.0/2.0, 3.0, 4.0] if bounce_ratios is None else bounce_ratios
    prev_frame = list(pitches[i-1]) if i>0 else []
    indices = ml.argsort()[::-1]
    while num_found < num_pitches and maybe <= ml.shape[0]:
      this_one = b2f(indices[maybe])
      bounce1 = any([any([abs(ratio_to_cents(
        this_one/(other_one*harm))) < bounce_width 
        for harm in bounce_ratios]) 
        for other_one in bounce_list])
      bounce2 = any([any([abs(ratio_to_cents
        (other_one/(this_one*harm))) < bounce_width
        for harm in bounce_ratios]) 
        for other_one in bounce_list])
      bounce = bounce1 or bounce2 
      if not bounce:
        found_so_far += [this_one]
        bounce_list += [this_one]
        num_found += 1
      maybe += 1

    indices = ml.argsort()[::-1][0:num_pitches]
    pitches[i] = np.array(found_so_far)
    confidences[i] = ml[indices]
    peaks[i] = peaks_frqs

    ml_peaks = pitches[i] 
    width = 25 

    frame_fundamentals = []
    for bin_frq in ml_peaks:

      nearest_harmonic = (peaks_frqs/bin_frq).round()

      mask = np.abs(ratio_to_cents_protected( 
        (peaks_frqs/bin_frq), (peaks_frqs/bin_frq).round())) <= width

      weights = ml_i( (peaks_frqs/bin_frq).round() )

      A = np.matrix(nearest_harmonic).T
      b = np.matrix(peaks_frqs).T
      W = np.matrix(np.diag(mask * weights))

      fund = np.linalg.lstsq(W*A, W*b)[0][0].item()

      frame_fundamentals += [fund]

    fundamentals[i] = np.array(frame_fundamentals)

  tracks = np.copy(pitches)

  return fundamentals, pitches, D, peaks, confidences, tracks

  # error at this point

def voice_tracks(pitches, confidences):

  out = np.zeros_like(pitches)
  out[0] = pitches[0]

  out_confidences = np.zeros_like(confidences)
  out_confidences[0] = confidences[0]

  num_frames, num_voices = pitches.shape

  for i in range(1,num_frames):

    prev_frame = out[i-1]
    next_frame = pitches[i]

    delta = np.abs(ratio_to_cents(np.atleast_2d(prev_frame).T/np.repeat(
      np.atleast_2d(next_frame),num_voices,axis=0)))

    delta_sorted = np.unravel_index(np.argsort(delta.ravel()), delta.shape)

    num_found = 0; found_prevs = []; found_nexts = []; index = 0

    while num_found < num_voices:
      prev_voice,next_voice = delta_sorted[0][index], delta_sorted[1][index]
      if prev_voice not in found_prevs and next_voice not in found_nexts:
        out[i][prev_voice] = pitches[i][next_voice]
        out_confidences[i][prev_voice] = confidences[i][next_voice]
        found_prevs += [prev_voice]
        found_nexts += [next_voice]
        num_found += 1
      index += 1

  return out, out_confidences


def pitchsets(pitches, win=3):

  pitches_change = change_filter(pitches)

  pitches_sustain = sustain_filter(pitches_change, win=win)

  pitchsets = np.sort(np.array(list(set(np.argwhere(np.nan_to_num(pitches_sustain))[:,0]))))
  
  return pitchsets, np.where(np.nan_to_num(pitches_sustain))



def change_filter(pitches):

  wpitches = np.copy(pitches)

  out = np.zeros_like(wpitches)
  out[out==0] = np.nan


  wpitches = freq_to_midi(wpitches).round()

  indices = np.diff(wpitches, axis=0).nonzero()

  out[0] = pitches[0]
  out[1:][indices] = pitches[1:][indices]

  return out 


def sustain_filter(pitches, win=3):

  out = np.zeros_like(pitches)

  for i in range(pitches.shape[0]-win):
    for j in range(pitches.shape[1]):

        out[i,j] = pitches[i,j] * np.isnan(pitches[i+1:i+win,j]).all()

  out[out==0] = np.nan

  return out 



def pitch_onsets(funds, win=2, depth=25):
  
  onsets = np.zeros_like(funds)
  for i in range(funds.shape[0]):
    for j in range(funds.shape[1]):
      onsets[i,j] = (np.abs(pyfid.ratio_to_cents(funds[i-win:i,:] / funds[i,j])) <= thresh).any(axis=1).all()

  return onsets