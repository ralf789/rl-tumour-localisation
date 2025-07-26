import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import random
import numpy as np
import cv2
import nibabel as nib

import gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def collect_volume_pairs(root_dir="image_with_masks"):
    """
    Collect (t2_file, mask_file, subdir_name) from subfolders.
    mask_file can be None if no tumour mask is found.
    """
    pairs = []
    for subdir in os.listdir(root_dir):
        subpath = os.path.join(root_dir, subdir)
        if os.path.isdir(subpath):
            t2_file = None
            mask_file = None
            for f in os.listdir(subpath):
                if f in ["t2.nii.gz", "t2_0.nii.gz"]:
                    t2_file = os.path.join(subpath, f)
                elif "l_a1.nii.gz" in f:
                    mask_file = os.path.join(subpath, f)
            if t2_file is not None:
                pairs.append((t2_file, mask_file, subdir))
    return pairs

def preload_volumes(volume_pairs):
    """
    Preload all T2 and mask volumes into memory once.
    Returns a dict: loaded_data[subdir] = (t2_data, mask_data_or_None).
    """
    print(f"[DEBUG] preload_volumes: got {len(volume_pairs)} pairs to load.")
    loaded_data = {}
    for (t2_file, mask_file, subdir) in volume_pairs:
        print(f"[DEBUG]   Loading subdir='{subdir}' => T2='{t2_file}', mask='{mask_file}'")
        t2_data = nib.load(t2_file).get_fdata()
        if mask_file is not None:
            mask_data = nib.load(mask_file).get_fdata()
        else:
            mask_data = None
        loaded_data[subdir] = (t2_data, mask_data)
    print("[DEBUG] preload_volumes: done preloading.")
    return loaded_data
    
class TumourEnv(gym.Env):

    def __init__(self, volume_pairs, loaded_data, crop_ratio=1/5, max_steps=500):
        super(TumourEnv, self).__init__()
        self.volume_pairs = volume_pairs
        self.loaded_data = loaded_data
        self.crop_ratio = crop_ratio
        self.max_steps = max_steps

        # Continuous action space
        self.action_space = spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32)

        # Observations: single-channel 128x128 images, dtype=uint8
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 128, 128), dtype=np.uint8
        )

        self.img_3d = None
        self.mask_3d = None
        self.X = self.Y = self.Z = 0
        self.cx = self.cy = self.cz = 0
        self.x = self.y = self.z = 0
        self.current_step = 0

        self.all_tumour_voxels = None
        self.volume_name = None
        self.visited_states = set()

        self.skip_active = False
        self.skip_start = (0, 0, 0)
        self.skip_end   = (0, 0, 0)
        self.backtrack_success = False


    def reset(self):
        (t2_file, mask_file, subdir) = random.choice(self.volume_pairs)
        self.volume_name = subdir

        t2_data, mask_data = self.loaded_data[subdir]
        self.img_3d = t2_data
        self.mask_3d = mask_data

        self.X, self.Y, self.Z = self.img_3d.shape
        self.cx = int(self.X * self.crop_ratio)
        self.cy = int(self.Y * self.crop_ratio)
        self.cz = int(self.Z * self.crop_ratio)

         # Use NumPy for random indexing
        self.x = np.random.randint(0, self.X - self.cx)
        self.y = np.random.randint(0, self.Y - self.cy)
        self.z = np.random.randint(0, self.Z - self.cz)
        self.current_step = 0

        # Reset visited states
        self.visited_states.clear()
        self.visited_states.add((int(self.x), int(self.y), int(self.z)))

        return self._get_observation()



    def step(self, action):
        dx, dy, dz = action
    
        old_x, old_y, old_z = self.x, self.y, self.z
        proposed_x = old_x + dx
        proposed_y = old_y + dy
        proposed_z = old_z + dz
    
        # ========== SUB-STEP LOGIC ==========
        sub_steps = 6           
        crossing_penalty = 0.005  # make it smaller
        midstep_penalty = 0.0
        traversed_tumour = False
    
        if self.mask_3d is not None:
            ratio = np.linspace(1 / sub_steps, 1, sub_steps, dtype=np.float32)
            mid_x = old_x + ratio*(proposed_x - old_x)
            mid_y = old_y + ratio*(proposed_y - old_y)
            mid_z = old_z + ratio*(proposed_z - old_z)
            for i in range(sub_steps):
                bx1 = int(mid_x[i])
                bx2 = int(mid_x[i] + self.cx)
                by1 = int(mid_y[i])
                by2 = int(mid_y[i] + self.cy)
                bz1 = int(mid_z[i])
                bz2 = int(mid_z[i] + self.cz)

                # clamp bounding box indices
                bx1 = max(bx1, 0)
                bx2 = min(bx2, self.X)
                by1 = max(by1, 0)
                by2 = min(by2, self.Y)
                bz1 = max(bz1, 0)
                bz2 = min(bz2, self.Z)

                if (bx2 > bx1 and by2 > by1 and bz2 > bz1):
                    sub_mask = self.mask_3d[bx1:bx2, by1:by2, bz1:bz2]
                    if np.any(sub_mask > 0):
                        traversed_tumour = True
                        break
    
        if traversed_tumour:
            midstep_penalty -= crossing_penalty
            self.skip_active = True
            self.skip_start = (old_x, old_y, old_z)
            self.skip_end   = (proposed_x, proposed_y, proposed_z)
    
         # ========== clamp final coords with NumPy clip ==========
        new_x = np.clip(proposed_x, 0, self.X - self.cx)
        new_y = np.clip(proposed_y, 0, self.Y - self.cy)
        new_z = np.clip(proposed_z, 0, self.Z - self.cz)

        # Check if clamping actually occurred
        clamped = (
            (new_x != proposed_x) or
            (new_y != proposed_y) or
            (new_z != proposed_z)
        )
    
        # === optional: nudge inward if exactly on boundary
        eps_in = 2.0
        if new_x == (self.X - self.cx):
            new_x -= eps_in
        if new_y == (self.Y - self.cy):
            new_y -= eps_in
        if new_z == (self.Z - self.cz):
            new_z -= eps_in
    
        self.x, self.y, self.z = new_x, new_y, new_z
    
        # ========== base step + midstep penalty ==========
        base_penalty = -0.01*(1 + self.current_step/self.max_steps)
        reward = base_penalty + midstep_penalty
    
        # ========== clamp penalty (bigger) ==========
        if clamped:
            reward -= 0.05  # big so agent avoids corner
    
        # ========== no-op / small move penalty ==========
        action_mag = np.linalg.norm([dx, dy, dz])  # uses NumPy
        if action_mag < 5.0:
            reward -= 0.05
    
        # ========== big XY early moves ==========
        time_ratio = self.current_step / self.max_steps
        if time_ratio < 0.3:
            xy_mag = np.linalg.norm([dx, dy])
            if xy_mag > 20.0:
                reward += 1.0
    
        # ========== check tumour overlap at final coords ==========
        found_tumour = False
        tumour_bonus = 200.0 #crazy to 200
        if self.mask_3d is not None:
            bx1 = int(self.x)
            bx2 = int(self.x + self.cx)
            by1 = int(self.y)
            by2 = int(self.y + self.cy)
            bz1 = int(self.z)
            bz2 = int(self.z + self.cz)
            if (bx2>bx1 and by2>by1 and bz2>bz1):
                final_patch = self.mask_3d[bx1:bx2, by1:by2, bz1:bz2]
                if (final_patch>0).any():
                    reward += tumour_bonus
                    found_tumour = True
    
        # leftover if tumour
        if found_tumour:
            leftover_steps = self.max_steps - self.current_step
            reward += 0.01* leftover_steps #crazy to 0.01
    
        # ========== increment step, check done ==========
        self.current_step += 1
        done = (found_tumour or self.current_step>=self.max_steps)
    
        # repeated states penalty
        dis_x = round(self.x*2)/2.0
        dis_y = round(self.y*2)/2.0
        dis_z = round(self.z*2)/2.0
        cur_st = (dis_x, dis_y, dis_z)
        if (cur_st in self.visited_states) and (not found_tumour):
            reward -= 1.0  # bigger penalty for repeating
        else:
            self.visited_states.add(cur_st)
    
        # backtrack
        if self.skip_active and not found_tumour:
            sx, sy, sz = self.skip_start
            ex, ey, ez = self.skip_end
            nx, ny, nz = self.x, self.y, self.z
            seg_x = ex-sx
            seg_y = ey-sy
            seg_z = ez-sz
            new_vx = nx-sx
            new_vy = ny-sy
            new_vz = nz-sz
            seg_len = seg_x*seg_x + seg_y*seg_y + seg_z*seg_z
            dot_val = seg_x*new_vx + seg_y*new_vy + seg_z*new_vz
    
            if seg_len>1e-8 and (0<=dot_val<=seg_len):
                reward += 0.3
                self.backtrack_success = True
                self.skip_active = False
    
        if found_tumour and hasattr(self,'backtrack_success') and self.backtrack_success:
            reward += 10.0 #crazy to 10
            self.backtrack_success = False
    
        # finalize
        obs = self._get_observation()
        info = {}
        if done:
            info["last_coords"] = (self.x,self.y,self.z)
            info["volume_name"] = self.volume_name
    
        return obs, reward, done, info



    def _get_observation(self):
        patch_3d = self.img_3d[
            int(self.x): int(self.x+self.cx),
            int(self.y): int(self.y+self.cy),
            int(self.z): int(self.z+self.cz)
        ]
        if patch_3d.shape[2] < 1:
            return np.zeros((1,128,128), dtype=np.uint8)

        sums = [patch_3d[:,:,z].sum() for z in range(patch_3d.shape[2])]
        best_z = np.argmax(sums)
        slice_2d = patch_3d[:,:,best_z]

        mn, mx = slice_2d.min(), slice_2d.max()
        if (mx - mn) < 1e-8:
            slice_norm = np.zeros_like(slice_2d, dtype=np.float32)
        else:
            slice_norm = (slice_2d - mn)/(mx - mn)

        slice_128 = cv2.resize(slice_norm, (128,128), interpolation=cv2.INTER_LINEAR)
        slice_255 = (slice_128 * 255.0).astype(np.uint8)

        return slice_255[None, :, :]

 


def main():
    script_start = time.time()  # measure entire script

    
    all_pairs = collect_volume_pairs("image_with_masks")
    if not all_pairs:
        print("No volumes found in 'image_with_masks'. Exiting.")
        return

    masked_pairs = []
    for (t2_file, mask_file, subdir) in all_pairs:
        if mask_file is not None:
            masked_pairs.append((t2_file, mask_file, subdir))

    if not masked_pairs:
        print("No volumes with masks found. Exiting.")
        return

    print("Total masked volumes:", len(masked_pairs))

    loaded_data = preload_volumes(masked_pairs)
    
    random.shuffle(masked_pairs)
    train_count = max(1, int(0.5 * len(masked_pairs)))
    train_vols = masked_pairs[:train_count]
    test_vols  = masked_pairs[train_count:]

    print("Training set size:", len(train_vols))
    print("Testing  set size:", len(test_vols))

    print("\n=== TRAINING FOLDERS ===")
    train_subdirs = [subdir for (_, _, subdir) in train_vols]
    print(", ".join(train_subdirs))
    
    print("\n=== TESTING FOLDERS ===")
    test_subdirs = [subdir for (_, _, subdir) in test_vols]
    print(", ".join(test_subdirs))

    train_env = TumourEnv(train_vols, loaded_data, crop_ratio=1/5, max_steps=500)
    vec_train = DummyVecEnv([lambda: train_env])

    test_env = TumourEnv(test_vols, loaded_data, crop_ratio=1/5, max_steps=500)
    vec_test = DummyVecEnv([lambda: test_env])

    model = PPO("CnnPolicy", vec_train, verbose=1)
    total_timesteps_used = 30000000

    print("\n===== Training PPO Agent with advanced reward shaping =====")
    start_train = time.time()
    model.learn(total_timesteps_used)
    end_train = time.time()
    print("crazyTraining done.\n")
    training_duration = end_train - start_train
    print(f"Training took {training_duration:.2f} seconds.")

    model.save("cc")
    print("Model saved as 'cc.zip'")

    print("===== Testing the Trained Model =====")
    test_episodes = 300
    num_success = 0
    total_rewards_sum = 0.0
    steps_list = []

    start_test = time.time()
    for ep in range(test_episodes):
        obs = vec_test.reset()
        done = [False]
        total_reward_ep = 0.0
        steps = 0
        last_coords = (0,0,0)
        volume_name = None

        while not done[0]:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = vec_test.step(action)
            total_reward_ep += rewards[0]
            steps += 1

            if dones[0]:
                if "last_coords" in infos[0]:
                    last_coords = infos[0]["last_coords"]
                if "volume_name" in infos[0]:
                    volume_name = infos[0]["volume_name"]
            done[0] = dones[0]

        total_rewards_sum += total_reward_ep
        steps_list.append(steps)
        if total_reward_ep > 0:
            num_success += 1

        print(f"Episode {ep+1}: volume={volume_name}, reward={total_reward_ep:.3f}, "
              f"steps={steps}, last_coords=({last_coords[0]:.2f}, {last_coords[1]:.2f}, {last_coords[2]:.2f})")

    end_test = time.time()
    testing_duration = end_test - start_test
    print(f"Testing took {testing_duration:.2f} seconds.")

    mean_reward = total_rewards_sum / test_episodes
    success_rate = 100.0 * num_success / test_episodes
    mean_steps = sum(steps_list) / len(steps_list)
    print(f"Mean reward over {test_episodes} episodes: {mean_reward:.2f}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Mean steps taken (across {test_episodes} episodes): {mean_steps:.2f}")

    script_end = time.time()
    total_duration = script_end - script_start
    print(f"Entire script took {total_duration:.2f} seconds.")



if __name__ == "__main__":
    main()
