//audio_processor.cpp
#include "AudioProcessor.h"
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <arm_math.h>
#include "mel_filters.h"

//constants
const int sampleRate = 22050;
const int durationSec = 3;
const int totalSamples = sampleRate * durationSec;
const int fftSize = 1024;
const int hopSize = 512;
const int padding = fftSize / 2;
const int paddedSamples = totalSamples + 2 * padding;
const int maxFrames = (paddedSamples - fftSize) / hopSize + 1;
#define FIR_LEN 64

//buffers (PSRAM)
EXTMEM float audioBuffer[paddedSamples];
EXTMEM float melMatrix[maxFrames][MEL_BANDS];
EXTMEM float dbMatrix[maxFrames][MEL_BANDS];
EXTMEM float firCoeffs_ram[FIR_LEN];
EXTMEM float melFilters_ram[MEL_BANDS][FFT_BINS];

//FIR coefficients in PROGMEM (flash)
const float firCoeffs_prog[FIR_LEN] PROGMEM = {
    0.000632f, -0.000432f, -0.000893f, 0.000261f, 0.001339f, 0.000093f, -0.001955f, -0.000834f,
    0.002608f, 0.002142f, -0.003035f, -0.004119f, 0.002866f, 0.006725f, -0.001658f, -0.009740f,
    -0.001047f, 0.012738f, 0.005683f, -0.015076f, -0.012640f, 0.015876f, 0.022328f, -0.013922f,
    -0.035455f, 0.007242f, 0.053958f, 0.008633f, -0.085114f, -0.051212f, 0.178146f, 0.415860f,
    0.415860f, 0.178146f, -0.051212f, -0.085114f, 0.008633f, 0.053958f, 0.007242f, -0.035455f,
    -0.013922f, 0.022328f, 0.015876f, -0.012640f, -0.015076f, 0.005683f, 0.012738f, -0.001047f,
    -0.009740f, -0.001658f, 0.006725f, 0.002866f, -0.004119f, -0.003035f, 0.002142f, 0.002608f,
    -0.000834f, -0.001955f, 0.000093f, 0.001339f, 0.000261f, -0.000893f, -0.000432f, 0.000632f
};

//FIR state (PSRAM for larger buffers)
EXTMEM float firBuffer[FIR_LEN];
int firIndex = 0;

//audio setup
AudioInputAnalog adc1(A9);
AudioRecordQueue queue1;
AudioConnection patchCord1(adc1, queue1);

//state
bool downsampleToggle = false;
int frameCount = 0;

float applyFIR(int16_t sample) {
    firBuffer[firIndex] = (float)sample;
    float result = 0.0f;
    int idx = firIndex;
    for (int i = 0; i < FIR_LEN; i++) {
        result += firCoeffs_ram[i] * firBuffer[idx];
        idx = (idx == 0) ? FIR_LEN - 1 : idx - 1;
    }
    firIndex = (firIndex + 1) % FIR_LEN;
    return result;
}

void init_audio_processor() {
    analogReadResolution(16);
    AudioMemory(40);  // Reduced from 60 to free RAM1

    //copy large arrays from PROGMEM to PSRAM
    memcpy(firCoeffs_ram, firCoeffs_prog, sizeof(firCoeffs_ram));
    memcpy(melFilters_ram, melFilters_prog, sizeof(melFilters_ram));

    queue1.begin();
}

bool process_audio(float* features_out) {
    // Reset buffers and state
    memset(audioBuffer, 0, sizeof(audioBuffer));
    memset(firBuffer, 0, sizeof(firBuffer));
    firIndex = 0;
    downsampleToggle = false;

    int writeIndex = padding;
    int samplesWritten = 0;

    //record audio
    while (samplesWritten < totalSamples) {
        if (queue1.available()) {
            int16_t* block = queue1.readBuffer();
            if (!block) continue;

            for (int i = 0; i < AUDIO_BLOCK_SAMPLES; i++) {
                float filtered = applyFIR(block[i]);
                downsampleToggle = !downsampleToggle;

                if (downsampleToggle && samplesWritten < totalSamples) {
                    audioBuffer[writeIndex++] = filtered;
                    samplesWritten++;
                }
            }

            queue1.freeBuffer();
        }
    }

    //compute Mel spectrogram
    frameCount = 0;
    static EXTMEM arm_rfft_fast_instance_f32 fft_inst;  // Static to PSRAM
    static EXTMEM float windowed[fftSize];
    static EXTMEM float fft_out[fftSize];
    for (int i = 0; i <= paddedSamples - fftSize; i += hopSize) {
        if (frameCount >= maxFrames) break;

        for (int k = 0; k < fftSize; k++) {
            float win = 0.5f * (1.0f - cosf(2.0f * M_PI * k / (fftSize - 1.0f)));
            windowed[k] = audioBuffer[i + k] * win;
        }

        arm_rfft_fast_init_f32(&fft_inst, fftSize);
        arm_rfft_fast_f32(&fft_inst, windowed, fft_out, 0);

        float fftMag[FFT_BINS];
        for (int j = 0; j < fftSize / 2; j++) {
            float re = fft_out[2 * j];
            float im = fft_out[2 * j + 1];
            fftMag[j] = sqrtf(re * re + im * im);
        }
        fftMag[FFT_BINS - 1] = 0.0f;

        for (int m = 0; m < MEL_BANDS; m++) {
            float mel = 0.0f;
            for (int k = 0; k < FFT_BINS; k++) {
                mel += melFilters_ram[m][k] * fftMag[k];
            }
            melMatrix[frameCount][m] = mel;
        }

        frameCount++;
    }

    //compute dB scale with global max normalization
    float maxVal = 0.0f;
    for (int i = 0; i < frameCount; i++) {
        for (int j = 0; j < MEL_BANDS; j++) {
            if (melMatrix[i][j] > maxVal) maxVal = melMatrix[i][j];
        }
    }

    for (int i = 0; i < frameCount; i++) {
        for (int j = 0; j < MEL_BANDS; j++) {
            float db = 10.0f * log10f(melMatrix[i][j] / (maxVal + 1e-10f));
            if (db < -80.0f) db = -80.0f;
            dbMatrix[i][j] = db;
        }
    }

    //find min/max in dB matrix
    float dataMin = 1e9f, dataMax = -1e9f;
    for (int i = 0; i < frameCount; i++) {
        for (int j = 0; j < MEL_BANDS; j++) {
            if (dbMatrix[i][j] < dataMin) dataMin = dbMatrix[i][j];
            if (dbMatrix[i][j] > dataMax) dataMax = dbMatrix[i][j];
        }
    }

    //normalize, map to grayscale, pack to hex RGB, and flatten
    int idx = 0;
    for (int j = 0; j < MEL_BANDS; j++) {
        for (int i = 0; i < frameCount; i++) {
            float norm = (dbMatrix[i][j] - dataMin) / (dataMax - dataMin);
            if (norm < 0.0f) norm = 0.0f;
            if (norm > 1.0f) norm = 1.0f;

            uint8_t gray = (uint8_t)(norm * 255.0f + 0.5f);
            uint32_t hexVal = ((uint32_t)gray << 16) | ((uint32_t)gray << 8) | (uint32_t)gray; // 0xGGGGGG
            features_out[idx++] = (float)hexVal;
        }
    }

    return true;
}