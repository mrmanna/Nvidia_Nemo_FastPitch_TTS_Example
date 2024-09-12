### A comprehensive guide is available in my Medium article:
[How to Build a High-Quality Text-to-Speech (TTS) System Locally with Nvidia NeMo FastPitch](https://mrmanna.medium.com/how-to-build-a-high-quality-text-to-speech-tts-system-locally-with-nvidia-nemo-fastpitch-98fc7b626819)

After running command:

``poetry install``

We have to update gcc to 12 to install cython and youtokentome which are required.

``conda install -c conda-forge gcc_linux-64=12``

``pip install cython youtokentome``

and then we can run the app like

``poetry run start <yourpdfile.pdf> <youroutputfile.wav>``

### Caution: 
This code includes minimal error handling. I encourage you to enhance it by addressing any issues you encounter during use.

Further if you want change pitch or speed, you can do that manipulating the spectrogram, happy coding. 