import os
import argparse
import numpy as np

def common_hand_proc_args():
    """Gather commonly used command line args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', '-v', action='store_true')
    parser.add_argument('--num_envs', '-B', type=int, default=1)
    parser.add_argument('--hand', type=str, default='inspire')
    parser.add_argument('--clip', type=str, default='box-20-100')
    # take in multiple hand names into a list of args
    parser.add_argument('--hands_list', type=str, nargs='+', default=['inspire'])
    parser.add_argument('--side', type=str, default='left')
    parser.add_argument('--asset_path', type=str, default='/home/mandi/chiral/assets')
    parser.add_argument('--vis_collision', '-vc', action='store_true')
    parser.add_argument('--record_video', '-rv', action='store_true')
    parser.add_argument('--raytrace', action='store_true') # if true, use raytracing renderer
    parser.add_argument('--overwrite', '-o', action='store_true')

    return parser

def parse_clip_string(clip):
    vals = clip.split('-')
    if len(vals) == 3:
        print("Default to using subject s01 and clip 01")
        vals += ['s01', 'u01']
    assert len(vals) == 5, "Clip should be in format: obj_name-start-end-subject-clip"
    obj_name = vals[0]
    start = int(vals[1])
    end = int(vals[2])
    subject = vals[3]
    use_clip = vals[4].replace('u', '') # just 01/02
    return obj_name, start, end, subject, use_clip



def generate_90degree_rotation_quaternions(max_combined_rotations=2):
    """
    Generate a complete list of quaternions that represent rotations around x, y, and z axes
    in 90-degree increments, with up to the specified number of combined rotations.

    Args:
        max_combined_rotations: Maximum number of sequential rotations to combine (default: 2)

    Returns:
        list: A list of dictionaries containing quaternion information
    """
    # Constants for quaternion calculations
    sin_45 = np.sin(np.pi/4)  # sin(π/4) = 1/√2 ≈ 0.7071
    cos_45 = np.cos(np.pi/4)  # cos(π/4) = 1/√2 ≈ 0.7071

    # Define basic quaternions for rotations around principal axes
    # Quaternion format: (w, x, y, z) where:
    # - w is the scalar part
    # - (x, y, z) is the vector part

    # Create a dictionary of base rotations
    base_rotations = {
        "I": (1.0, 0.0, 0.0, 0.0),           # Identity
        "X_90": (cos_45, sin_45, 0.0, 0.0),   # 90° around X
        "X_180": (0.0, 1.0, 0.0, 0.0),        # 180° around X
        "X_270": (cos_45, -sin_45, 0.0, 0.0), # 270° around X (or -90°)
        "Y_90": (cos_45, 0.0, sin_45, 0.0),   # 90° around Y
        "Y_180": (0.0, 0.0, 1.0, 0.0),        # 180° around Y
        "Y_270": (cos_45, 0.0, -sin_45, 0.0), # 270° around Y (or -90°)
        "Z_90": (cos_45, 0.0, 0.0, sin_45),   # 90° around Z
        "Z_180": (0.0, 0.0, 0.0, 1.0),        # 180° around Z
        "Z_270": (cos_45, 0.0, 0.0, -sin_45)  # 270° around Z (or -90°)
    }

    # Start with single rotations
    result = []
    for name, quat in base_rotations.items():
        result.append({
            "name": name,
            "quaternion": quat,
            "description": get_rotation_description(name)
        })

    # Only proceed with combined rotations if requested
    if max_combined_rotations >= 2:
        # Generate all combinations of two rotations
        for first_name, first_quat in base_rotations.items():
            for second_name, second_quat in base_rotations.items():
                if first_name == "I" or second_name == "I":
                    continue  # Skip combinations with identity

                combined_quat = quaternion_multiply(second_quat, first_quat)
                combined_name = f"{first_name}_{second_name}"

                # Check if this rotation is effectively the same as one we already have
                if not is_duplicate_rotation(combined_quat, [item["quaternion"] for item in result]):
                    result.append({
                        "name": combined_name,
                        "quaternion": combined_quat,
                        "description": f"{get_rotation_description(first_name)} followed by {get_rotation_description(second_name)}"
                    })

    return result

def get_rotation_description(rotation_name):
    """Convert rotation name to human-readable description"""
    if rotation_name == "I":
        return "No rotation (identity)"

    parts = rotation_name.split("_")
    axis = parts[0]
    degrees = parts[1]

    return f"{degrees}° around {axis}-axis"
