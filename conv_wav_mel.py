import os
import shutil
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB
from tqdm import tqdm
import random
from shutil import copytree, rmtree
from pydub import AudioSegment

def split_wav_files(src_folder, dest_folder, time_length, hop_length):
    # Copy the original folder structure to the destination folder
    if os.path.exists(dest_folder):
        rmtree(dest_folder)
    copytree(src_folder, dest_folder)

    # Iterate through the subfolders and .wav files
    for subdir, dirs, files in os.walk(src_folder):
        for file in tqdm(files):
            if file.endswith('.wav'):
                src_file_path = os.path.join(subdir, file)
                dest_subfolder = os.path.join(dest_folder, os.path.relpath(subdir, src_folder))

                # Split the .wav file
                audio = AudioSegment.from_wav(src_file_path) #pydub
                audio_length_ms = len(audio)
                time_length_ms = time_length * 1000
                hop_length_ms = hop_length * 1000

                # Save the segments into the destination folder
                for i in range(0, audio_length_ms - time_length_ms + 1, hop_length_ms):
                    segment = audio[i:i + time_length_ms]
                    segment_file_name = f"{os.path.splitext(file)[0]}_part{i // hop_length_ms}.wav"
                    segment.export(os.path.join(dest_subfolder, segment_file_name), format="wav")

                # Remove the original .wav file from the destination folder
                os.remove(os.path.join(dest_subfolder, file))

def create_test_set(test_folder, src_folder):

    # Create the test folder if it doesn't exist
    os.makedirs(test_folder, exist_ok=True)

    # Iterate through the subfolders in the dataset folder
    for class_folder in os.listdir(src_folder):
        class_path = os.path.join(src_folder, class_folder)
        
        # Check if the current item is a directory (class folder)
        if os.path.isdir(class_path):
            # Create a new subfolder for the current class in the test folder
            test_class_folder = os.path.join(test_folder, class_folder)
            os.makedirs(test_class_folder, exist_ok=True)
            
            # Get a list of all samples in the class folder
            samples = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            # Randomly select 20 samples from the list
            samples_to_move = random.sample(samples, 20)
            
            # Move the selected samples to the corresponding test folder
            for sample in samples_to_move:
                src_sample_path = os.path.join(class_path, sample)
                dest_sample_path = os.path.join(test_class_folder, sample)
                shutil.move(src_sample_path, dest_sample_path)
    
def preprocess_audio(waveform, sample_rate):
    # Load audio and resample to 16 kHz
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Convert to mel-spectrogram
    n_fft = int(0.064 * 16000)  # 64 ms window
    hop_length = int(0.01 * 16000)  # 10 ms step size
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=n_fft, hop_length=hop_length, n_mels=64, f_min=60, f_max=7800)(waveform)

    # Convert to log-scaled mel-spectrogram
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB(stype='power')(mel_spectrogram)

    return log_mel_spectrogram

class convert_wav_mel():

    def __init__(self, target_sample_rate, target_sec):
        self.target_sample_rate = target_sample_rate
        self.target_sec = target_sec * target_sample_rate

    def convert_folder(self, root_dir, save_dir):

        for filename in tqdm(os.listdir(root_dir)):
            file = os.path.join(root_dir, filename)
            save_filename = filename[:-3] + 'pt'
            save_file_path = os.path.join(save_dir, save_filename)
            waveform, sample_rate = torchaudio.load(file)

            if waveform.shape[1] == 0:
                print(f"Skipping empty waveform in file: {filename}")
                continue

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            log_mel_spec = preprocess_audio(waveform=waveform, sample_rate=sample_rate)
            torch.save(log_mel_spec, save_file_path)

    def convert(self, root_dir, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for dirname in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dirname)
            if os.path.isdir(dir_path):
                new_folder_path = os.path.join(save_dir, dirname)
                if os.path.exists(new_folder_path):
                    shutil.rmtree(new_folder_path)
                os.mkdir(new_folder_path)
                self.convert_folder(root_dir=dir_path, save_dir=new_folder_path)
            else:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.mkdir(save_dir)
                self.convert_folder(root_dir=root_dir, save_dir=save_dir)
                return


