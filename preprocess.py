def DataPreProcess(raw):
      # 1. Read CNT
      raw = mne.io.read_raw_cnt(cnt_file, preload=True)

      #2. Remove noise and false differences (other denoising methods can be added as needed, such as ICA, etc.)
      raw.notch_filter([50, 60], fir_design='firwin')

      # 3. Create Xdawn object
      n_components = 20  # Choose the number of components to keep
      xdawn = Xdawn(n_components=n_components)

      # Fit Xdawn on the data
      xdawn.fit(raw)

      # Apply Xdawn components to the data
      raw = xdawn.apply(raw)

      # 4. Select the channel of interest
      raw.pick_channels(channels)

      # 5. Filtering
      low_freq = 4  # Delta Waves Theta Waves）
      high_freq = 12  #（Alpha Waves、Beta Waves Gamma Waves）
      raw.filter(low_freq, high_freq)

      # 6. Re-reference to average reference
      raw.set_eeg_reference('average', projection=True)

      # # 7.（Baseline Correction）
      raw.apply_baseline(baseline=(None, 0))

      # 8. Downsampling
      # raw.resample(resample_rate)

def DivideData(raw, window_size, sampling_rate):
      # Get filtered signal data
      data, times = raw[:]

      # Perform Continuous Wavelet Transform (CWT)
      wavelet_type = 'morl'  # Choose a wavelet for CWT
      scales = np.arange(1, 11)  # Define the scales for CWT
      cwt_coeffs, frequencies = pywt.cwt(data, scales, wavelet_type)

      # Compute wavelet coefficient energy
      energy = [np.sum(np.abs(cwt_coeff)**2, axis=0) for cwt_coeff in cwt_coeffs]

      max_length = max(len(energy_level) for energy_level in energy)
      energy_padded = [np.pad(energy_level, (0, max_length - len(energy_level)), 'constant') for energy_level in energy]

      # normalized energy value
      scaler = MinMaxScaler()
      energy_normalized = scaler.fit_transform(np.vstack(energy_padded).T).T

      # apply logarithmic scaling to energy values
      log_energy = np.log10(energy_normalized + 1)

      # Divide the energy value according to the size of the time window
      window_size_samples = int(window_size * sampling_rate)
      num_windows = (log_energy.shape[1] - window_size_samples) // window_size_samples + 1

      # Initialize as an empty list
      wavelet_time_freq_plots = []

      for i in range(num_windows):
          start_sample = i * window_size_samples
          end_sample = start_sample + window_size_samples

          # Extract the energy value of the current time window
          window_energy = log_energy[:, start_sample:end_sample]

          wavelet_time_freq_plots.append(window_energy)

      # Convert the list to numpy array
      wavelet_time_freq_plots = np.array(wavelet_time_freq_plots)
      return wavelet_time_freq_plots


def DrawAndSaveImages(wavelet_time_freq_plots,total_wavelet_plots,window_size,sampling_rate):
      print("Shape of wavelet_time_freq_plots in DrawAndSaveImages:", wavelet_time_freq_plots.shape)
      total_file_wavelet_plots = 0
      # Get time axis and frequency axis
      time_axis = np.arange(wavelet_time_freq_plots.shape[2]) * window_size
      freq_axis = np.arange(wavelet_time_freq_plots.shape[2]) * (sampling_rate / (2 * wavelet_time_freq_plots.shape[2]))

      # Traverse the wavelet time-frequency graph of each time window and draw
      for i in range(wavelet_time_freq_plots.shape[0]):
          plt.figure(figsize=(10, 6))

          current_window_data = wavelet_time_freq_plots[i]

          time_axis_window = np.linspace(time_axis[0], time_axis[-1], current_window_data.shape[0])
          freq_axis_window = np.linspace(freq_axis[0], freq_axis[-1], current_window_data.shape[1])
          plt.imshow(current_window_data, aspect='auto', cmap='jet', extent=[time_axis_window[0], time_axis_window[-1], freq_axis_window[0], freq_axis_window[-1]])

          # Image extraction feature values for CNN do not need to display xy axis coordinates and colorbar
          plt.xlabel('Time (s)')
          plt.ylabel('Frequency')
          plt.title(f'Wavelet Time-Frequency Plot - Window {i+1} - {cnt_file}')
          plt.colorbar()
          # plt.axis('off')

          # Set the save path and image file
          save_path = ''
          prefix = os.path.basename(cnt_file)[:1]
          print(prefix)
          if prefix == 'c':
              save_path = New_CT_path
          if prefix == 'p':
              save_path = New_PB_path
          file_name = f'{cnt_file.split(".")[0]} - Window{i+1}.png'
          plt.savefig(f'{save_path}/{file_name}')

          plt.show()

          plt.close()

          # Determine whether the edf_file belongs to healthy(0) or schizophrenia(1), and then add the corresponding label
          if cnt_file.startswith('c'):
              y_labels.append(0)
          else:
              y_labels.append(1)
          total_file_wavelet_plots+=1
          total_wavelet_plots+=total_file_wavelet_plots

  # cnt_file = '/content/cnt_file'

  New_CT_path = '/content/drive/MyDrive/PaperData/NewCT20'
  New_PB_path = '/content/drive/MyDrive/PaperData/NewPB20'

  channels = ["Pz", "Fz", "Cz", "Pz", "C3", "T3", "C4", "T4", "Fp1", "Fp2", "F3", "F4", "F7", "F8",
  "P3", "P4", "T5", "T6", "O1", "O2"]


  # Define time window size and overlap size in seconds
  # Split the data into time windows of fixed length 25 seconds
  window_size = 25

  # No data overlap between adjacent time windows
  overlap_size = 0

  # Reduce the sample rate to 250 Hz
  # resample_rate = 200

  # Store feature values and labels for all images
  X_features = []
  y_labels = []

  total_wavelet_plots = 0
  import mne

  # for cnt_file in os.listdir('/content/'):
  #     full_file_path = os.path.join('/content/', cnt_file)
  #     if cnt_file.endswith('.cnt'):
  #         raw = mne.io.read_raw_cnt(full_file_path, preload=True)
  #         # Process the raw data here
  #         sampling_rate = raw.info['sfreq']
  #         DataPreProcess(raw)
  #         wavelet_time_freq_plots = DivideData(raw, window_size, sampling_rate)
  #         DrawAndSaveImages(wavelet_time_freq_plots, total_wavelet_plots, window_size, sampling_rate)
  #         except Exception as e:
  #             error_message = str(e)
  #             if "tuple index out of range" in error_message:
  #                 print(f"Error processing {cnt_file}: {error_message}")
  #                 continue  # Skip to the next file
  #             else:
  #                 raise  # Re-raise the exception if it's not the expected one


  # print(f"Total Wavelet Time-Frequency Plot count：{total_wavelet_plots}")

  for cnt_file in os.listdir('/content/'):
    full_file_path = os.path.join('/content/', cnt_file)
    if cnt_file.endswith('.cnt'):
        try:
            raw = mne.io.read_raw_cnt(full_file_path, preload=True)
            raw.set_meas_date(None)

            # Set the sampling rate to 200 Hz
            sampling_rate = 200

            # Process the raw data here
            wavelet_time_freq_plots = DivideData(raw, window_size, sampling_rate)
            DrawAndSaveImages(wavelet_time_freq_plots, total_wavelet_plots, window_size, sampling_rate)
        except Exception as e:
            error_message = str(e)
            if "tuple index out of range" in error_message:
                print(f"Error processing {cnt_file}: {error_message}")
                continue  # Skip to the next file
            else:
                raise  # Re-raise the exception if it's not the expected one


  print(f"Total Wavelet Time-Frequency Plot count: {total_wavelet_plots}")