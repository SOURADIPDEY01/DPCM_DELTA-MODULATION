import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. GENERATE INPUT SIGNAL
# ============================================================

def generate_signal(frequency, fs, duration):

    t = np.arange(0, duration, 1 / fs)

    x = np.sin(2 * np.pi * frequency * t)

    return t, x


# ============================================================
# 2. FIRST-ORDER DPCM
# ============================================================

def dpcm(x, delta):

    N = len(x)

    predicted = np.zeros(N)
    prediction_error = np.zeros(N)
    quantized_error = np.zeros(N)
    reconstructed = np.zeros(N)

    for n in range(N):

        # ---------------- PREDICTION ----------------
        if n == 0:
            predicted[n] = 0
        else:
            predicted[n] = reconstructed[n - 1]

        # ---------------- PREDICTION ERROR ----------------
        prediction_error[n] = x[n] - predicted[n]

        # ---------------- QUANTIZATION ----------------
        quantized_error[n] = (
            delta * np.round(prediction_error[n] / delta)
        )

        # ---------------- RECONSTRUCTION ----------------
        reconstructed[n] = (
            predicted[n] + quantized_error[n]
        )

    return (
        predicted,
        prediction_error,
        quantized_error,
        reconstructed
    )


# ============================================================
# 3. PCM QUANTIZATION
# ============================================================

def pcm(x, bits=4):

    L = 2 ** bits

    delta = 2 / L

    index = np.floor((x + 1) / delta)

    index = np.clip(index, 0, L - 1)

    reconstructed = (
        -1 + (index + 0.5) * delta
    )

    error = x - reconstructed

    return reconstructed, error


# ============================================================
# 4. DELTA MODULATION
# ============================================================

def delta_modulation(x, delta):

    N = len(x)

    staircase = np.zeros(N)

    dm_output = np.zeros(N)

    for n in range(1, N):

        # Comparator
        if x[n] >= staircase[n - 1]:

            dm_output[n] = 1

        else:

            dm_output[n] = -1

        # Reconstruction
        staircase[n] = (
            staircase[n - 1]
            + dm_output[n] * delta
        )

    return dm_output, staircase


# ============================================================
# 5. CHECK DELTA MODULATION STEPS
# ============================================================

def check_delta_steps(staircase, delta):

    steps = np.diff(staircase)

    valid = np.logical_or(
        np.isclose(steps, delta),
        np.isclose(steps, -delta)
    )

    return np.all(valid)


# ============================================================
# 6. CALCULATE MSE
# ============================================================

def calculate_mse(original, reconstructed):

    return np.mean(
        (original - reconstructed) ** 2
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

print("\n====================================================")
print("       DPCM AND DELTA MODULATION SIMULATION")
print("====================================================")


# ============================================================
# USER INPUT
# ============================================================

fs = float(
    input("Enter sampling frequency (Hz): ")
)

frequency = float(
    input("Enter input signal frequency (Hz): ")
)

duration = float(
    input("Enter signal duration (seconds): ")
)

dpcm_delta = float(
    input("Enter DPCM quantization step size Δ: ")
)


# ============================================================
# GENERATE USER INPUT SIGNAL
# ============================================================

t, x = generate_signal(
    frequency,
    fs,
    duration
)


# ============================================================
# PART A: FIRST-ORDER DPCM
# ============================================================

(
    predicted,
    prediction_error,
    quantized_error,
    dpcm_reconstructed
) = dpcm(
    x,
    dpcm_delta
)

dpcm_mse = calculate_mse(
    x,
    dpcm_reconstructed
)


# ============================================================
# PART B: PCM VS DPCM
# ============================================================

pcm_reconstructed, pcm_error = pcm(
    x,
    bits=4
)

pcm_mse = np.mean(
    pcm_error ** 2
)


print("\n====================================================")
print("                  DPCM RESULTS")
print("====================================================")

print("Input frequency              :", frequency, "Hz")
print("DPCM quantization step Δ     :", dpcm_delta)

print(
    "Prediction Error MSE         :",
    np.mean(prediction_error ** 2)
)

print(
    "DPCM Reconstruction MSE      :",
    dpcm_mse
)

print("\n--------------- PCM VS DPCM ----------------")

print("PCM bits                     : 4")

print(
    "PCM Quantization MSE         :",
    pcm_mse
)

print(
    "DPCM Reconstruction MSE      :",
    dpcm_mse
)


# ============================================================
# PLOT 1: ORIGINAL VS PREDICTED SAMPLES
# ============================================================

Nplot = min(50, len(x))

plt.figure(figsize=(12, 6))

plt.plot(
    range(Nplot),
    x[:Nplot],
    'o-',
    label="Original Samples"
)

plt.plot(
    range(Nplot),
    predicted[:Nplot],
    's--',
    label="Predicted Samples"
)

plt.xlabel("Sample Number")
plt.ylabel("Amplitude")

plt.title(
    "DPCM: Original Samples vs Predicted Samples"
)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# PLOT 2: PREDICTION ERROR
# ============================================================

plt.figure(figsize=(12, 5))

plt.stem(
    range(Nplot),
    prediction_error[:Nplot]
)

plt.xlabel("Sample Number")
plt.ylabel("Prediction Error")

plt.title(
    "DPCM Prediction Error"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# PART C: DELTA MODULATION
# ============================================================

# Automatically selected step sizes

small_delta = 0.02
moderate_delta = 0.10
large_delta = 0.30

delta_values = [
    small_delta,
    moderate_delta,
    large_delta
]

delta_names = [
    "Small Step Size",
    "Moderate Step Size",
    "Large Step Size"
]


# ============================================================
# SLOW AND RAPID INPUTS
# ============================================================

slow_frequency = frequency

# Automatically create a rapidly varying input
rapid_frequency = min(
    fs / 4,
    max(5 * frequency, frequency + 1)
)

t_slow, x_slow = generate_signal(
    slow_frequency,
    fs,
    duration
)

t_rapid, x_rapid = generate_signal(
    rapid_frequency,
    fs,
    duration
)


print("\n====================================================")
print("              DELTA MODULATION")
print("====================================================")

print("Slow input frequency  :", slow_frequency, "Hz")
print("Rapid input frequency :", rapid_frequency, "Hz")


# ============================================================
# STORE MSE VALUES
# ============================================================

slow_mse_values = []
rapid_mse_values = []


# ============================================================
# TEST SMALL, MODERATE AND LARGE Δ
# ============================================================

for delta, name in zip(
    delta_values,
    delta_names
):

    print("\n----------------------------------------------------")
    print(name)
    print("----------------------------------------------------")

    print("Step size Δ =", delta)


    # ========================================================
    # SLOW INPUT
    # ========================================================

    dm_slow, staircase_slow = delta_modulation(
        x_slow,
        delta
    )

    slow_mse = calculate_mse(
        x_slow,
        staircase_slow
    )

    slow_mse_values.append(
        slow_mse
    )

    slow_steps_ok = check_delta_steps(
        staircase_slow,
        delta
    )

    print("\nSlow input:")
    print("MSE =", slow_mse)

    if slow_steps_ok:
        print("Step inspection: PASS")
        print("Every step is +Δ or -Δ")
    else:
        print("Step inspection: FAIL")


    # ========================================================
    # RAPID INPUT
    # ========================================================

    dm_rapid, staircase_rapid = delta_modulation(
        x_rapid,
        delta
    )

    rapid_mse = calculate_mse(
        x_rapid,
        staircase_rapid
    )

    rapid_mse_values.append(
        rapid_mse
    )

    rapid_steps_ok = check_delta_steps(
        staircase_rapid,
        delta
    )

    print("\nRapid input:")
    print("MSE =", rapid_mse)

    if rapid_steps_ok:
        print("Step inspection: PASS")
        print("Every step is +Δ or -Δ")
    else:
        print("Step inspection: FAIL")


    # ========================================================
    # PLOT SLOW INPUT + STAIRCASE
    # ========================================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        t_slow[:Nplot],
        x_slow[:Nplot],
        label="Original Signal"
    )

    plt.step(
        t_slow[:Nplot],
        staircase_slow[:Nplot],
        where="post",
        label="Delta Staircase"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.title(
        f"Slow Input - {name} (Δ = {delta})"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ========================================================
    # PLOT RAPID INPUT + STAIRCASE
    # ========================================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        t_rapid[:Nplot],
        x_rapid[:Nplot],
        label="Original Signal"
    )

    plt.step(
        t_rapid[:Nplot],
        staircase_rapid[:Nplot],
        where="post",
        label="Delta Staircase"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.title(
        f"Rapid Input - {name} (Δ = {delta})"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# PART D: MSE VS STEP SIZE
# ============================================================

step_sizes = np.linspace(
    0.01,
    0.50,
    30
)

mse_slow = []
mse_rapid = []


for delta in step_sizes:

    _, staircase_slow = delta_modulation(
        x_slow,
        delta
    )

    _, staircase_rapid = delta_modulation(
        x_rapid,
        delta
    )

    mse_slow.append(
        calculate_mse(
            x_slow,
            staircase_slow
        )
    )

    mse_rapid.append(
        calculate_mse(
            x_rapid,
            staircase_rapid
        )
    )


# ============================================================
# PLOT MSE VS STEP SIZE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    step_sizes,
    mse_slow,
    'o-',
    label="Slow Input"
)

plt.plot(
    step_sizes,
    mse_rapid,
    's-',
    label="Rapid Input"
)

plt.xlabel("Step Size Δ")
plt.ylabel("MSE")

plt.title(
    "Delta Modulation: MSE vs Step Size"
)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


