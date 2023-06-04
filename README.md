# ML4SA
My curated list of machine learning-based techniques for sound art in creative media.

[TOC]



## Audio Recognition

- Audio recognition involves the identification and classification of audio signals or specific features within audio data.

- Examples in Sound Art: Audio recognition can be used in sound art to analyze and classify sounds based on their content or characteristics. For instance, using machine learning techniques, sound artists can create installations that classify and respond to specific sound events, such as distinguishing between footsteps and bird chirping.

- State-of-the-Art Methods: Popular machine learning libraries for audio recognition include:

  - Librosa: A Python library for music and audio analysis, providing features for audio preprocessing, feature extraction, and classification.

  - Tensorflow Audio: A library that extends TensorFlow with audio-specific functionalities, including audio preprocessing, feature extraction, and model building.

  - YAMNet: A deep learning model developed by Google that can recognize a wide range of audio events from 521 audio classes.

 

## Audio Synthesis

- Audio synthesis involves the generation or creation of new audio signals using various algorithms and techniques.
- Examples in Sound Art: Sound artists can use audio synthesis to create unique and expressive sounds for their installations or compositions. They can generate abstract textures, atmospheric tones, or even imitate real-world sounds using synthesis techniques. For example, a sound artist may use granular synthesis to manipulate and transform recorded environmental sounds into a mesmerizing sonic landscape.
- State-of-the-Art Methods: Notable machine learning libraries and models for audio synthesis include:
  - Magenta: An open-source project by Google that explores the intersection of machine learning and music generation. It provides models like MusicVAE and NSynth for generating new musical compositions and timbres.
  - WaveGAN: A Generative Adversarial Network (GAN)-based model that can generate high-quality audio samples. It can learn and mimic the distribution of real audio data to create realistic and diverse sounds.

## Audio Transformation

- Audio transformation involves modifying or manipulating audio signals to achieve desired effects or transformations.
- Examples in Sound Art: Sound artists can use audio transformation techniques to alter and shape sound elements in their compositions or installations. They can apply effects like time stretching, pitch shifting, or spectral manipulation to create unique sonic experiences. For instance, a sound artist might use time stretching to slow down or stretch out a sound sample, creating an ethereal and atmospheric effect.
- State-of-the-Art Methods: Some machine learning libraries and models for audio transformation include:
  - Essentia: An open-source library for audio analysis and transformation, providing a wide range of audio processing algorithms and tools.
  - Spleeter: A library developed by Deezer that uses deep learning to perform source separation, allowing users to separate vocals, drums, and other sound sources from mixed audio.



## Audio Related Packages

- Total number of packages: 66

#### Read-Write

* [audiolazy](https://github.com/danilobellini/audiolazy) [:octocat:](https://github.com/danilobellini/audiolazy) [:package:](https://pypi.python.org/pypi/audiolazy/) - Expressive Digital Signal Processing (DSP) package for Python.
* [audioread](https://github.com/beetbox/audioread) [:octocat:](https://github.com/beetbox/audioread) [:package:](https://pypi.python.org/pypi/audioread/) - Cross-library (GStreamer + Core Audio + MAD + FFmpeg) audio decoding.
* [mutagen](https://mutagen.readthedocs.io/) [:octocat:](https://github.com/quodlibet/mutagen) [:package:](https://pypi.python.org/pypi/mutagen) - Reads and writes all kind of audio metadata for various formats.
* [pyAV](http://docs.mikeboers.com/pyav/) [:octocat:](https://github.com/mikeboers/PyAV) - PyAV is a Pythonic binding for FFmpeg or Libav.
* [(Py)Soundfile](http://pysoundfile.readthedocs.io/) [:octocat:](https://github.com/bastibe/PySoundFile) [:package:](https://pypi.python.org/pypi/SoundFile) - Library based on libsndfile, CFFI, and NumPy.
* [pySox](https://github.com/rabitt/pysox) [:octocat:](https://github.com/rabitt/pysox) [:package:](https://pypi.python.org/pypi/pysox/) - Wrapper for sox.
* [stempeg](https://github.com/faroit/stempeg) [:octocat:](https://github.com/faroit/stempeg) [:package:](https://pypi.python.org/pypi/stempeg/) - read/write of STEMS multistream audio.
* [tinytag](https://github.com/devsnd/tinytag) [:octocat:](https://github.com/devsnd/tinytag) [:package:](https://pypi.python.org/pypi/tinytag/) - reading music meta data of MP3, OGG, FLAC and Wave files.

#### Transformations - General DSP

* [acoustics](http://python-acoustics.github.io/python-acoustics/) [:octocat:](https://github.com/python-acoustics/python-acoustics/) [:package:](https://pypi.python.org/pypi/acoustics) - useful tools for acousticians.
* [AudioTK](https://github.com/mbrucher/AudioTK) [:octocat:](https://github.com/mbrucher/AudioTK) - DSP filter toolbox (lots of filters).
* [AudioTSM](https://audiotsm.readthedocs.io/) [:octocat:](https://github.com/Muges/audiotsm) [:package:](https://pypi.python.org/pypi/audiotsm/) - real-time audio time-scale modification procedures.
* [Gammatone](https://github.com/detly/gammatone) [:octocat:](https://github.com/detly/gammatone) - Gammatone filterbank implementation.
* [pyFFTW](http://pyfftw.github.io/pyFFTW/) [:octocat:](https://github.com/pyFFTW/pyFFTW) [:package:](https://pypi.python.org/pypi/pyFFTW/) - Wrapper for FFTW(3).
* [NSGT](https://grrrr.org/research/software/nsgt/) [:octocat:](https://github.com/grrrr/nsgt) [:package:](https://pypi.python.org/pypi/nsgt) - Non-stationary gabor transform, constant-q.
* [matchering](https://github.com/sergree/matchering) [:octocat:](https://github.com/sergree/matchering) [:package:](https://pypi.org/project/matchering/) - Automated reference audio mastering.
* [MDCT](https://github.com/nils-werner/mdct) [:octocat:](https://github.com/nils-werner/mdct) [:package:](https://pypi.python.org/pypi/mdct) - MDCT transform.
* [pydub](http://pydub.com) [:octocat:](https://github.com/jiaaro/pydub) [:package:](https://pypi.python.org/pypi/mdct) - Manipulate audio with a simple and easy high level interface.
* [pytftb](http://tftb.nongnu.org) [:octocat:](https://github.com/scikit-signal/pytftb) - Implementation of the MATLAB Time-Frequency Toolbox.
* [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) [:octocat:](https://github.com/LCAV/pyroomacoustics) [:package:](https://pypi.python.org/pypi/pyroomacoustics) - Room Acoustics Simulation (RIR generator)
* [PyRubberband](https://github.com/bmcfee/pyrubberband) [:octocat:](https://github.com/bmcfee/pyrubberband) [:package:](https://pypi.python.org/pypi/pyrubberband/) - Wrapper for [rubberband](http://breakfastquay.com/rubberband/) to do pitch-shifting and time-stretching.
* [PyWavelets](http://pywavelets.readthedocs.io) [:octocat:](https://github.com/PyWavelets/pywt) [:package:](https://pypi.python.org/pypi/PyWavelets) - Discrete Wavelet Transform in Python.
* [Resampy](http://resampy.readthedocs.io) [:octocat:](https://github.com/bmcfee/resampy) [:package:](https://pypi.python.org/pypi/resampy) - Sample rate conversion.
* [SFS-Python](http://www.sfstoolbox.org) [:octocat:](https://github.com/sfstoolbox/sfs-python) [:package:](https://pypi.python.org/pypi/sfs/) - Sound Field Synthesis Toolbox.
* [sound_field_analysis](https://appliedacousticschalmers.github.io/sound_field_analysis-py/) [:octocat:](https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py) [:package:](https://pypi.org/project/sound-field-analysis/) - Analyze, visualize and process sound field data recorded by spherical microphone arrays.
* [STFT](http://stft.readthedocs.io) [:octocat:](https://github.com/nils-werner/stft) [:package:](https://pypi.python.org/pypi/stft) - Standalone package for Short-Time Fourier Transform.

#### Feature extraction

* [aubio](http://aubio.org/) [:octocat:](https://github.com/aubio/aubio) [:package:](https://pypi.python.org/pypi/aubio) - Feature extractor, written in C, Python interface.
* [audioFlux](https://github.com/libAudioFlux/audioFlux) [:octocat:](https://github.com/libAudioFlux/audioFlux) [:package:](https://pypi.python.org/pypi/audioflux) - A library for audio and music analysis, feature extraction.
* [audiolazy](https://github.com/danilobellini/audiolazy) [:octocat:](https://github.com/danilobellini/audiolazy) [:package:](https://pypi.python.org/pypi/audiolazy/) - Realtime Audio Processing lib, general purpose.
* [essentia](http://essentia.upf.edu) [:octocat:](https://github.com/MTG/essentia) - Music related low level and high level feature extractor, C++ based, includes Python bindings.
* [python_speech_features](https://github.com/jameslyons/python_speech_features) [:octocat:](https://github.com/jameslyons/python_speech_features) [:package:](https://pypi.python.org/pypi/python_speech_features) - Common speech features for ASR.
* [pyYAAFE](https://github.com/Yaafe/Yaafe) [:octocat:](https://github.com/Yaafe/Yaafe) - Python bindings for YAAFE feature extractor.
* [speechpy](https://github.com/astorfi/speechpy) [:octocat:](https://github.com/astorfi/speechpy) [:package:](https://pypi.python.org/pypi/speechpy) - Library for Speech Processing and Recognition, mostly feature extraction for now.
* [spafe](https://github.com/SuperKogito/spafe) [:octocat:](https://github.com/SuperKogito/spafe) [:package:](https://pypi.org/project/spafe/) - Python library for features extraction from audio files.

#### Data augmentation

* [audiomentations](https://github.com/iver56/audiomentations) [:octocat:](https://github.com/iver56/audiomentations) [:package:](https://pypi.org/project/audiomentations/) -  Audio Data Augmentation.
* [muda](https://muda.readthedocs.io/en/latest/) [:octocat:](https://github.com/bmcfee/muda) [:package:](https://pypi.python.org/pypi/muda) -  Musical Data Augmentation.
* [pydiogment](https://github.com/SuperKogito/pydiogment) [:octocat:](https://github.com/SuperKogito/pydiogment) [:package:](https://pypi.org/project/pydiogment/) -  Audio Data Augmentation.

#### Speech Processing

* [aeneas](https://www.readbeyond.it/aeneas/) [:octocat:](https://github.com/readbeyond/aeneas/) [:package:](https://pypi.python.org/pypi/aeneas/) - Forced aligner, based on MFCC+DTW, 35+ languages.
* [deepspeech](https://github.com/mozilla/DeepSpeech) [:octocat:](https://github.com/mozilla/DeepSpeech) [:package:](https://pypi.org/project/deepspeech/) - Pretrained automatic speech recognition.
* [gentle](https://github.com/lowerquality/gentle) [:octocat:](https://github.com/lowerquality/gentle) - Forced-aligner built on Kaldi.
* [Parselmouth](https://github.com/YannickJadoul/Parselmouth) [:octocat:](https://github.com/YannickJadoul/Parselmouth) [:package:](https://pypi.org/project/praat-parselmouth/) - Python interface to the [Praat](http://www.praat.org) phonetics and speech analysis, synthesis, and manipulation software.
* [persephone](https://persephone.readthedocs.io/en/latest/) [:octocat:](https://github.com/persephone-tools/persephone) [:package:](https://pypi.org/project/persephone/) - Automatic phoneme transcription tool.
* [pyannote.audio](https://github.com/pyannote/pyannote-audio) [:octocat:](https://github.com/pyannote/pyannote-audio) [:package:](https://pypi.org/project/pyannote-audio/) - Neural building blocks for speaker diarization.
* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)² [:octocat:](https://github.com/tyiannak/pyAudioAnalysis) [:package:](https://pypi.python.org/pypi/pyAudioAnalysis/) - Feature Extraction, Classification, Diarization.
* [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) [:octocat:](https://github.com/wiseman/py-webrtcvad) [:package:](https://pypi.python.org/pypi/webrtcvad/) -  Interface to the WebRTC Voice Activity Detector.
* [pypesq](https://github.com/vBaiCai/python-pesq) [:octocat:](https://github.com/vBaiCai/python-pesq) - Wrapper for the PESQ score calculation.
* [pystoi](https://github.com/mpariente/pystoi) [:octocat:](https://github.com/mpariente/pystoi) [:package:](https://pypi.org/project/pystoi) - Short Term Objective Intelligibility measure (STOI).
* [PyWorldVocoder](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) [:octocat:](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) - Wrapper for Morise's World Vocoder.
* [Montreal Forced Aligner](https://montrealcorpustools.github.io/Montreal-Forced-Aligner/) [:octocat:](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) - Forced aligner, based on Kaldi (HMM), English (others can be trained).
* [SIDEKIT](http://lium.univ-lemans.fr/sidekit/) [:package:](https://pypi.python.org/pypi/SIDEKIT/) - Speaker and Language recognition.
* [SpeechRecognition](https://github.com/Uberi/speech_recognition) [:octocat:](https://github.com/Uberi/speech_recognition) [:package:](https://pypi.python.org/pypi/SpeechRecognition/) -  Wrapper for several ASR engines and APIs, online and offline.

#### Environmental Sounds

* [sed_eval](http://tut-arg.github.io/sed_eval) [:octocat:](https://github.com/TUT-ARG/sed_eval) [:package:](https://pypi.org/project/sed_eval/) - Evaluation toolbox for Sound Event Detection

#### Perceptial Models - Auditory Models

* [cochlea](https://github.com/mrkrd/cochlea) [:octocat:](https://github.com/mrkrd/cochlea) [:package:](https://pypi.python.org/pypi/cochlea/) - Inner ear models.
* [Brian2](http://briansimulator.org/) [:octocat:](https://github.com/brian-team/brian2) [:package:](https://pypi.python.org/pypi/Brian2) - Spiking neural networks simulator, includes cochlea model.
* [Loudness](https://github.com/deeuu/loudness) [:octocat:](https://github.com/deeuu/loudness) - Perceived loudness, includes Zwicker, Moore/Glasberg model.
* [pyloudnorm](https://www.christiansteinmetz.com/projects-blog/pyloudnorm) [:octocat:](https://github.com/csteinmetz1/pyloudnorm) - Audio loudness meter and normalization, implements ITU-R BS.1770-4.
* [Sound Field Synthesis Toolbox](http://www.sfstoolbox.org) [:octocat:](https://github.com/sfstoolbox/sfs-python) [:package:](https://pypi.python.org/pypi/sfs/) - Sound Field Synthesis Toolbox.

#### Source Separation

* [commonfate](https://github.com/aliutkus/commonfate) [:octocat:](https://github.com/aliutkus/commonfate) [:package:](https://pypi.python.org/pypi/commonfate) - Common Fate Model and Transform.
* [NTFLib](https://github.com/stitchfix/NTFLib) [:octocat:](https://github.com/stitchfix/NTFLib) - Sparse Beta-Divergence Tensor Factorization.
* [NUSSL](https://interactiveaudiolab.github.io/project/nussl.html) [:octocat:](https://github.com/interactiveaudiolab/nussl) [:package:](https://pypi.python.org/pypi/nussl) - Holistic source separation framework including DSP methods and deep learning methods.
* [NIMFA](http://nimfa.biolab.si) [:octocat:](https://github.com/marinkaz/nimfa) [:package:](https://pypi.python.org/pypi/nimfa) - Several flavors of non-negative-matrix factorization.

#### Music Information Retrieval

* [Catchy](https://github.com/jvbalen/catchy) [:octocat:](https://github.com/jvbalen/catchy) - Corpus Analysis Tools for Computational Hook Discovery.
* [Madmom](https://madmom.readthedocs.io/en/latest/) [:octocat:](https://github.com/CPJKU/madmom) [:package:](https://pypi.python.org/pypi/madmom) - MIR packages with strong focus on beat detection, onset detection and chord recognition.
* [mir_eval](http://craffel.github.io/mir_eval/) [:octocat:](https://github.com/craffel/mir_eval) [:package:](https://pypi.python.org/pypi/mir_eval) - Common scores for various MIR tasks. Also includes bss_eval implementation.
* [msaf](http://pythonhosted.org/msaf/) [:octocat:](https://github.com/urinieto/msaf) [:package:](https://pypi.python.org/pypi/msaf) - Music Structure Analysis Framework.
* [librosa](http://librosa.github.io/librosa/) [:octocat:](https://github.com/librosa/librosa) [:package:](https://pypi.python.org/pypi/librosa) - General audio and music analysis.

#### Deep Learning

* [Kapre](https://github.com/keunwoochoi/kapre) [:octocat:](https://github.com/keunwoochoi/kapre) [:package:](https://pypi.python.org/pypi/kapre) - Keras Audio Preprocessors
* [TorchAudio](https://github.com/pytorch/audio) [:octocat:](https://github.com/pytorch/audio) - PyTorch Audio Loaders
* [nnAudio](https://github.com/KinWaiCheuk/nnAudio) [:octocat:](https://github.com/KinWaiCheuk/nnAudio) [:package:](https://pypi.org/project/nnAudio/) - Accelerated audio processing using 1D convolution networks in PyTorch.

#### Symbolic Music - MIDI - Musicology

* [Music21](http://web.mit.edu/music21/) [:octocat:](https://github.com/cuthbertLab/music21) [:package:](https://pypi.python.org/pypi/music21) - Toolkit for Computer-Aided Musicology.
* [Mido](https://mido.readthedocs.io/en/latest/) [:octocat:](https://github.com/olemb/mido) [:package:](https://pypi.python.org/pypi/mido) - Realtime MIDI wrapper.
* [mingus](https://github.com/bspaans/python-mingus) [:octocat:](https://github.com/bspaans/python-mingus) [:package:](https://pypi.org/project/mingus) - Advanced music theory and notation package with MIDI file and playback support.
* [Pretty-MIDI](http://craffel.github.io/pretty-midi/) [:octocat:](https://github.com/craffel/pretty-midi) [:package:](https://pypi.python.org/pypi/pretty-midi) - Utility functions for handling MIDI data in a nice/intuitive way.

#### Realtime applications

* [Jupylet](https://github.com/nir/jupylet) [:octocat:](https://github.com/nir/jupylet) - Subtractive, additive, FM, and sample-based sound synthesis.
* [PYO](http://ajaxsoundstudio.com/software/pyo/) [:octocat:](https://github.com/belangeo/pyo) - Realtime audio dsp engine.
* [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) [:octocat:](http://python-sounddevice.readthedocs.io) [:package:](https://pypi.python.org/pypi/sounddevice) - PortAudio wrapper providing realtime audio I/O with NumPy.
* [ReTiSAR](https://github.com/AppliedAcousticsChalmers/ReTiSAR) [:octocat:](https://github.com/AppliedAcousticsChalmers/ReTiSAR) - Binarual rendering of streamed or IR-based high-order spherical microphone array signals.

#### Web Audio

* [TimeSide (Beta)](https://github.com/Parisson/TimeSide/tree/dev) [:octocat:](https://github.com/Parisson/TimeSide/tree/dev) - high level audio analysis, imaging, transcoding, streaming and labelling.

#### Audio Dataset and Dataloaders

* [beets](http://beets.io/) [:octocat:](https://github.com/beetbox/beets) [:package:](https://pypi.python.org/pypi/beets) - Music library manager and [MusicBrainz](https://musicbrainz.org/) tagger.
* [musdb](http://dsdtools.readthedocs.io) [:octocat:](https://github.com/sigsep/sigsep-mus-db) [:package:](https://pypi.python.org/pypi/musdb) - Parse and process the MUSDB18 dataset.
* [medleydb](http://medleydb.readthedocs.io) [:octocat:](https://github.com/marl/medleydb) - Parse [medleydb](http://medleydb.weebly.com/) audio + annotations.
* [Soundcloud API](https://github.com/soundcloud/soundcloud-python) [:octocat:](https://github.com/soundcloud/soundcloud-python) [:package:](https://pypi.python.org/pypi/soundcloud) - Wrapper for [Soundcloud API](https://developers.soundcloud.com/).
* [Youtube-Downloader](http://rg3.github.io/youtube-dl/) [:octocat:](https://github.com/rg3/youtube-dl) [:package:](https://pypi.python.org/pypi/youtube_dl) - Download youtube videos (and the audio).
* [audiomate](https://github.com/ynop/audiomate) [:octocat:](https://github.com/ynop/audiomate) [:package:](https://pypi.python.org/pypi/audiomate/) - Loading different types of audio datasets.
* [mirdata](https://mirdata.readthedocs.io/en/latest/) [:octocat:](https://github.com/mir-dataset-loaders/mirdata) [:package:](https://pypi.python.org/pypi/mirdata) - Common loaders for Music Information Retrieval (MIR) datasets.
#### Wrappers for Audio Plugins

* [VamPy Host](https://code.soundsoftware.ac.uk/projects/vampy-host) [:package:](https://pypi.python.org/pypi/vamp) - Interface compiled vamp plugins.



## References

- https://github.com/faroit/awesome-python-scientific-audio
- https://github.com/topics/audio-processing
