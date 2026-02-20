#!/usr/bin/env python3
"""Sim2Real Analysis Script - Generates plots from logged data."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create output directory
os.makedirs(os.path.join(SCRIPT_DIR, 'analysis_output'), exist_ok=True)

# Plotting settings
plt.rcParams['figure.figsize'] = [14, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Joint names in real robot order (29 DOF)
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"
]

JOINT_GROUPS = {
    "Left Leg": list(range(0, 6)),
    "Right Leg": list(range(6, 12)),
    "Waist": list(range(12, 15)),
    "Left Arm": list(range(15, 22)),
    "Right Arm": list(range(22, 29)),
}

# Data column indices
IDX_TIME = 0
IDX_JOINT_POS = slice(1, 30)
IDX_JOINT_VEL = slice(30, 59)
IDX_QUAT = slice(59, 63)
IDX_GYRO = slice(63, 66)
IDX_TORQUE = slice(66, 95)
IDX_TARGET = slice(95, 124)

CONTROL_FREQ = 50

print("=" * 60)
print("SIM2REAL ANALYSIS")
print("=" * 60)

# Load data
DATA_PATH = os.path.join(SCRIPT_DIR, "../eval_data/unitree_g1_data_log.csv")
raw_data = pd.read_csv(DATA_PATH, header=0)
data = raw_data.values

# Filter out zero rows
valid_mask = data[:, IDX_TIME] > 0
data = data[valid_mask]

print(f"\nLoaded {len(data)} timesteps ({len(data)/CONTROL_FREQ:.1f} seconds)")
print(f"Time range: {data[0, IDX_TIME]:.2f}s - {data[-1, IDX_TIME]:.2f}s")

# Extract data columns
time = data[:, IDX_TIME]
joint_pos = data[:, IDX_JOINT_POS]
joint_vel = data[:, IDX_JOINT_VEL]
quat = data[:, IDX_QUAT]
gyro = data[:, IDX_GYRO]
torque = data[:, IDX_TORQUE]
target = data[:, IDX_TARGET]
tracking_error = target - joint_pos

# ============================================================
# Plot 00: Control loop time analysis
# ============================================================

print("Generating control loop time plot...")

dt = np.diff(time)
time_dt = time[1:]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(time_dt, dt * 1000, linewidth=0.8)
ax.axhline(y=1000.0 / CONTROL_FREQ, color='r', linestyle='--', alpha=0.5,
           label=f'Expected ({1000.0/CONTROL_FREQ:.0f} ms)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Loop Duration (ms)')
ax.set_title('Control Loop Timing (timestep[i] - timestep[i-1])')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'analysis_output/00_control_loop_time.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 1: Tracking Error by Joint Group
# ============================================================
print("\nGenerating tracking error plots...")

fig, axes = plt.subplots(len(JOINT_GROUPS), 1, figsize=(14, 12), sharex=True)
fig.suptitle('Target vs Actual Joint Angles by Group', fontsize=14)

for ax, (group_name, indices) in zip(axes, JOINT_GROUPS.items()):
    colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))
    for i, idx in enumerate(indices):
        ax.plot(time, target[:, idx], '--', color=colors[i], alpha=0.8,
                label=f'{JOINT_NAMES[idx]} (target)')
        ax.plot(time, joint_pos[:, idx], '-', color=colors[i], alpha=0.8,
                label=f'{JOINT_NAMES[idx]} (actual)')
    ax.set_ylabel('Angle (rad)')
    ax.set_title(group_name)
    ax.legend(loc='upper right', fontsize=7, ncol=2)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'analysis_output/01_target_vs_actual_by_group.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 01b: RMS Tracking Error Summary
# ============================================================
rms_error = np.sqrt(np.mean(tracking_error**2, axis=0))
max_abs_error = np.max(np.abs(tracking_error), axis=0)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(JOINT_NAMES))
width = 0.35

bars1 = ax.bar(x - width/2, rms_error, width, label='RMS Error', color='steelblue')
bars2 = ax.bar(x + width/2, max_abs_error, width, label='Max Abs Error', color='coral')

ax.set_ylabel('Error (rad)')
ax.set_title('Tracking Error Summary per Joint')
ax.set_xticks(x)
ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'analysis_output/01b_tracking_error_summary.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 2: Joint Velocity by Group
# ============================================================
print("Generating joint velocity plots...")

fig, axes = plt.subplots(len(JOINT_GROUPS), 1, figsize=(14, 12), sharex=True)
fig.suptitle('Joint Velocity by Group', fontsize=14)

for ax, (group_name, indices) in zip(axes, JOINT_GROUPS.items()):
    for idx in indices:
        ax.plot(time, joint_vel[:, idx], label=JOINT_NAMES[idx], alpha=0.8)
    ax.set_ylabel('Velocity (rad/s)')
    ax.set_title(group_name)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'analysis_output/02_joint_velocity_by_group.png'), dpi=150, bbox_inches='tight')
plt.close()



# ============================================================
# Plot 3: Motor Torque by Joint Group
# ============================================================
print("Generating torque analysis plots...")

fig, axes = plt.subplots(len(JOINT_GROUPS), 1, figsize=(14, 12), sharex=True)
fig.suptitle('Motor Torque by Joint Group', fontsize=14)

for ax, (group_name, indices) in zip(axes, JOINT_GROUPS.items()):
    for idx in indices:
        ax.plot(time, torque[:, idx], label=JOINT_NAMES[idx], alpha=0.8)
    ax.set_ylabel('Torque (Nm)')
    ax.set_title(group_name)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'analysis_output/03_torque_by_group.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 4: IMU / Orientation Analysis
# ============================================================
print("Generating IMU analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Quaternion components
ax = axes[0, 0]
ax.plot(time, quat[:, 0], label='w')
ax.plot(time, quat[:, 1], label='x')
ax.plot(time, quat[:, 2], label='y')
ax.plot(time, quat[:, 3], label='z')
ax.set_title('Quaternion Components (wxyz)')
ax.set_xlabel('Time (s)')
ax.legend()

# Quaternion norm
quat_norm = np.linalg.norm(quat, axis=1)
ax = axes[0, 1]
ax.plot(time, quat_norm)
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
ax.set_title('Quaternion Norm (should be 1.0)')
ax.set_xlabel('Time (s)')
ax.set_ylim([0.95, 1.05])

# Gyroscope
ax = axes[1, 0]
ax.plot(time, gyro[:, 0], label='X (roll rate)')
ax.plot(time, gyro[:, 1], label='Y (pitch rate)')
ax.plot(time, gyro[:, 2], label='Z (yaw rate)')
ax.set_title('Gyroscope (rad/s)')
ax.set_xlabel('Time (s)')
ax.legend()

# Euler angles
try:
    quat_xyzw = quat[:, [1, 2, 3, 0]]
    euler = Rotation.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
    ax = axes[1, 1]
    ax.plot(time, euler[:, 0], label='Roll')
    ax.plot(time, euler[:, 1], label='Pitch')
    ax.plot(time, euler[:, 2], label='Yaw')
    ax.set_title('Euler Angles (degrees)')
    ax.set_xlabel('Time (s)')
    ax.legend()
except Exception as e:
    axes[1, 1].text(0.5, 0.5, f'Euler conversion failed: {e}',
                    ha='center', va='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'analysis_output/04_imu_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# Plot 5: Detailed Joint Analysis (Left Knee)
# ============================================================
print("Generating detailed joint analysis...")

def compute_fft(signal_data, fs=CONTROL_FREQ):
    n = len(signal_data)
    fft_result = np.fft.rfft(signal_data - np.mean(signal_data))
    freqs = np.fft.rfftfreq(n, 1/fs)
    magnitude = np.abs(fft_result) / n
    return freqs, magnitude

def detailed_joint_analysis(joint_idx, filename):
    joint_name = JOINT_NAMES[joint_idx]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, target[:, joint_idx], 'b-', label='Target', alpha=0.7)
    ax1.plot(time, joint_pos[:, joint_idx], 'r-', label='Actual', alpha=0.7)
    ax1.set_title(f'{joint_name} - Position Tracking')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (rad)')
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, tracking_error[:, joint_idx], 'g-')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_title(f'{joint_name} - Tracking Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (rad)')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time, joint_vel[:, joint_idx], 'purple')
    ax3.set_title(f'{joint_name} - Velocity')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (rad/s)')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time, torque[:, joint_idx], 'orange')
    ax4.set_title(f'{joint_name} - Motor Torque')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Torque (Nm)')

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(joint_pos[:, joint_idx], joint_vel[:, joint_idx],
               c=time, cmap='viridis', s=2, alpha=0.5)
    ax5.set_title(f'{joint_name} - Phase Space')
    ax5.set_xlabel('Position (rad)')
    ax5.set_ylabel('Velocity (rad/s)')

    ax6 = fig.add_subplot(gs[2, 1])
    freqs, mag = compute_fft(tracking_error[:, joint_idx])
    ax6.semilogy(freqs, mag)
    ax6.set_title(f'{joint_name} - Error Spectrum')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Magnitude')
    ax6.set_xlim([0, CONTROL_FREQ/2])

    plt.suptitle(f'Detailed Analysis: {joint_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

detailed_joint_analysis(3, os.path.join(SCRIPT_DIR, 'analysis_output/05_detailed_left_knee.png'))



