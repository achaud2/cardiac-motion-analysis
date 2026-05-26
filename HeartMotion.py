import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Normal heart frames
normal_frames = []

# Create 50 frames
for t in range(51):

    # Black image
    img = np.zeros((200, 200), dtype=np.uint8)

    # Radius changes over time
    radius = int(40 + 10 * np.sin(2 * np.pi * t / 30))

    # Draw white circle
    cv2.circle(img, (100, 100), radius, 255, -1)

    normal_frames.append(img)

# Save as video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('normal_heart.mp4', fourcc, 10.0, (200, 200), isColor=False)

for frame in normal_frames:
    out.write(frame)

out.release()
print("Video saved as normal_heart.mp4")

'''
# Show one frame
plt.imshow(normal_frames[10], cmap='gray')
plt.title("Fake Heart Frame")
plt.axis('off')
plt.show()
'''

# Abnormal heart frames
abnormal_frames = []

# Create 50 frames
for t in range(51):

    # Black image
    img = np.zeros((200, 200), dtype=np.uint8)

    # Radius changes over time
    radius = int(40 + 4 * np.sin(2 * np.pi * t / 30))

    # Draw white ellipse
    cv2.ellipse(img, (100,100), (radius, radius-10), 0, 0, 360, 255, -1)

    abnormal_frames.append(img)

# Save as video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('abnormal_heart.mp4', fourcc, 10.0, (200, 200), isColor=False)

for frame in abnormal_frames:
    out.write(frame)

out.release()
print("Video saved as abnormal_heart.mp4")

'''
# Show one frame
plt.imshow(abnormal_frames[10], cmap='gray')
plt.title("Fake Heart Frame")
plt.axis('off')
plt.show()
'''
# Motion analysis using optical flow

# Compute optical flow between consecutive frames
def compute_optical_flow(frames):
    flows = []
    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(frames[i-1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
    return flows

# Compute optical flow for both normal and abnormal heart videos

normal_flows = compute_optical_flow(normal_frames)

abnormal_flows = compute_optical_flow(abnormal_frames)

results = []

# Normal heart
for i, flow in enumerate(normal_flows):

    dx = flow[:, :, 0]
    dy = flow[:, :, 1]

    motion_magnitude = np.sqrt(dx**2 + dy**2)

    mean_motion = np.mean(motion_magnitude)
    std_motion = np.std(motion_magnitude)
    max_motion = np.max(motion_magnitude)

    results.append({
        "type": "normal",
        "frame": i,
        "mean_motion": mean_motion,
        "std_motion": std_motion,
        "max_motion": max_motion
    })

# Abnormal heart
for i, flow in enumerate(abnormal_flows):

    dx = flow[:, :, 0]
    dy = flow[:, :, 1]

    motion_magnitude = np.sqrt(dx**2 + dy**2)

    mean_motion = np.mean(motion_magnitude)
    std_motion = np.std(motion_magnitude)
    max_motion = np.max(motion_magnitude)

    results.append({
        "type": "abnormal",
        "frame": i,
        "mean_motion": mean_motion,
        "std_motion": std_motion,
        "max_motion": max_motion
    })

print("Motion Analysis Results:")
for res in results:
    print(f"Type: {res['type']}, Frame: {res['frame']}, Mean Motion: {res['mean_motion']:.2f}, Std Motion: {res['std_motion']:.2f}, Max Motion: {res['max_motion']:.2f}")       


df = pd.DataFrame(results)

df.to_csv("motion_stats.csv", index=False)

print(df)


'''
# Visualizes motion magnitude
plt.imshow(motion_magnitude1, cmap='hot')
plt.title("Motion Magnitude (Normal Heart)")
plt.axis('off')
plt.show()

plt.imshow(motion_magnitude2, cmap='hot')
plt.title("Motion Magnitude (Abnormal Heart)")
plt.axis('off')
plt.show()
'''
