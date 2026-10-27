"""
Reference implementation:
https://github.com/google-research/robopianist/blob/d9cde23e46cb30ebb8eeebb375a9c52191238a30/robopianist/models/hands/shadow_hand.py#L41C1-L41C14


==== to print out all the joint names: ====
from lxml import etree
lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
tree = etree.parse("xxx.urdf", parser=lxml_parser)
print([j.attrib['name'] for j in tree.findall("joint") if j.attrib['type'] != 'fixed'])
"""
import os
import argparse
from lxml import etree
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

@dataclass
class Dof:
    """Forearm degree of freedom."""
    joint_type: str
    axis: Tuple[int, int, int]
    joint_range: Tuple[float, float] = (-2.0, 2.0)
    stiffness: int = 1
    reflect: bool = False
    effort: int = 100 # don't make it too small for heavier hands!!
    velocity: int = 5

_FOREARM_DOFS: Dict[str, Dof] = {
    "forearm_tx": Dof(
        joint_type="prismatic", axis=(1, 0, 0),
    ),
    "forearm_ty": Dof(
        joint_type="prismatic", axis=(0, 1, 0),
    ),
    "forearm_tz": Dof(
        joint_type="prismatic", axis=(0, 0, 1),
    ),
    "forearm_roll": Dof(
        joint_type="revolute", axis=(0, 0, 1), joint_range=(-6.2, 6.2),
    ),
    "forearm_pitch": Dof(
        joint_type="revolute", axis=(1, 0, 0), joint_range=(-6.2, 6.2),
    ),
    "forearm_yaw": Dof(
        joint_type="revolute", axis=(0, -1, 0), joint_range=(-6.2, 6.2),
    ),
}

def add_new_elem_for_dof(dof: Dof, root_elem, prev_joint_elem, new_link_name: str, base_link_name: str, left_hand=True):
    """ for each new 1-dof joint, need to:
    1. a new dummy link
    2. change the prev_joint_elem child to this dummy link
    3. add a new joint to connect this dummy link to the base_link.

    No need to add transmission/actuators since the USD converter ignores them.
    """
    new_link = etree.Element("link", name=new_link_name)
    # add mass 0.01 to this new_link! seems to stablize simulation
    mass = etree.Element("mass", value="0.01")
    new_link.append(mass)
    # also add inertial e.g.
    # <inertial>
    #   <origin xyz="0 0 0" rpy="0 0 0"/>
    #   <mass value="0.01"/>
    #   <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    # </inertial>
    inertial = etree.Element("inertial")
    origin = etree.Element("origin", xyz="0 0 0", rpy="0 0 0")
    mass = etree.Element("mass", value="0.01")
    inertia = etree.Element("inertia", ixx="0.01", ixy="0", ixz="0", iyy="0.01", iyz="0", izz="0.01")
    inertial.append(inertia)
    inertial.append(origin)
    inertial.append(mass)
    inertial.append(inertia)
    # new_link.append(inertial)
    root_elem.insert(1, new_link)

    # modify the <child> of prev_joint_elem to be new_link
    child_elem = prev_joint_elem.find("child")
    assert child_elem is not None, "No child element found."
    child_elem.attrib["link"] = new_link_name

    # Now add a new joint that connect this new link to the base_link, e.g.
    new_joint_name = f"{new_link_name}_joint"
    joint_type = dof.joint_type
    new_joint = etree.Element("joint", name=new_joint_name, type=joint_type)
    parent = etree.Element("parent", link=new_link_name)
    child = etree.Element("child", link=base_link_name)
    axis = dof.axis
    if left_hand and dof.reflect:
        axis = tuple(-x for x in axis)
    axis_str = " ".join(str(x) for x in axis)
    axis = etree.Element("axis", xyz=axis_str)
    lower, upper = dof.joint_range
    limit = etree.Element("limit", lower=str(lower), upper=str(upper), effort=str(dof.effort), velocity=str(dof.velocity))
    new_joint.append(parent)
    new_joint.append(child)
    new_joint.append(axis)
    new_joint.append(limit)
    root_elem.insert(1, new_joint)
    return new_joint_name


def add_forearm_dof(input_fname, output_fname, dof_choices: Sequence[str], base_link_name="base_link", left_hand=True, skip_side_prefix=False):
    """
    Add forearm degree of freedom to the URDF file.
    For every new joint, need to add a new dummy link to the URDF file and add new transmission and actuator tags.
    """
    side_prefix= "L_" if left_hand else "R_"
    if skip_side_prefix:
        side_prefix = ""
    lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    tree = etree.parse(input_fname, parser=lxml_parser)
    robot_elem = tree.getroot()
    assert robot_elem.tag == "robot", "The first element should be a robot element."

    # remove all transmission and actuator elements
    for elem in robot_elem.findall("transmission"):
        robot_elem.remove(elem)
    #  find the first joint name='fixed' and type='fixed':
    joint_elem = robot_elem.find("joint")
    assert joint_elem is not None, "No joint element found."
    print(joint_elem.attrib)
    assert joint_elem.attrib["type"] == "fixed", "The first joint should be a fixed joint."

    curr_base_name = base_link_name
    new_joint_names = []
    for dof_choice in dof_choices:
        dof = _FOREARM_DOFS[dof_choice]
        new_link_name = f"{side_prefix}{dof_choice}_link"
        new_joint_name = add_new_elem_for_dof(
            dof, robot_elem, joint_elem, new_link_name, curr_base_name, left_hand=left_hand
            )
        new_joint_names.append(new_joint_name)
        curr_base_name = new_link_name
    tree.write(output_fname, pretty_print=True)
    return new_joint_names

if __name__ == "__main__":
    # now parse commandline arguments
    parser = argparse.ArgumentParser(description="Add new degree of freedom to the URDF file.")
    parser.add_argument("input", type=str, help="Input URDF file name.")
    parser.add_argument("output", type=str, help="Output URDF file name.")
    parser.add_argument("--dof_choices", type=str, default=["all"], nargs="+", help="List of degree of freedom choices.")
    parser.add_argument("--base_link", type=str, default="base_link", help="Base link name.")
    parser.add_argument("--skip_side_prefix", action="store_true", help="Skip the side prefix in the new joint names.")
    parser.add_argument("--print_mimic_joints", "-pm", action="store_true")
    parser.add_argument("--print_links", "-pl", action="store_true")
    parser.add_argument("--print_joint_names", "-pj", action="store_true")

    args = parser.parse_args()
    input_fname = args.input
    input_path = os.path.dirname(input_fname)
    output_fname = os.path.join(input_path, args.output)
    # fname = "handright01091.urdf"
    # output_fname = "handright01091/urdf/handright01091_new.urdf"
    if args.dof_choices == ["all"]:
        dof_choices = ["forearm_tx", "forearm_ty", "forearm_tz", "forearm_roll", "forearm_pitch", "forearm_yaw"]
        # NOTE: reverse the order!
        dof_choices = dof_choices[::-1]
    else:
        dof_choices = args.dof_choices
    if args.print_joint_names:
        lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
        tree = etree.parse(input_fname, parser=lxml_parser)
        robot_elem = tree.getroot()
        assert robot_elem.tag == "robot", "The first element should be a robot element."
        joint_names = []
        for joint_elem in robot_elem.findall("joint"):
            if joint_elem.attrib["type"] != "fixed":
                joint_names.append(joint_elem.attrib["name"])
        print("Joint names:\n")
        print(' '.join([
                f"'{name}'," for name in joint_names
                ]
            ))
        exit()

    if args.print_mimic_joints:
            lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
            tree = etree.parse(input_fname, parser=lxml_parser)
            robot_elem = tree.getroot()
            assert robot_elem.tag == "robot", "The first element should be a robot element."
            act_joints = []
            mimic_joints = dict()
            for joint_elem in robot_elem.findall("joint"):
                jname = joint_elem.attrib["name"]
                mimic_elem = joint_elem.find("mimic")
                if mimic_elem is None and joint_elem.attrib.get("type", "fixed") != "fixed":
                    act_joints.append(jname)
                elif mimic_elem is not None:
                    parent_joint = mimic_elem.attrib["joint"]
                    mul = mimic_elem.attrib["multiplier"]
                    mimic_joints[jname] = (parent_joint, float(mul))

            # print
            print("Actuated joints:\n")
            print('\n'.join([
                f"'{name}'," for name in act_joints
                ]
            ))

            print("Mimic joints:\n")
            for k, (parent, mul) in mimic_joints.items():
                print(f"'{k}': ('{parent}', {mul}),")
            exit()

    if args.print_links:
        lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
        tree = etree.parse(input_fname, parser=lxml_parser)
        robot_elem = tree.getroot()
        assert robot_elem.tag == "robot", "The first element should be a robot element."
        links = []
        for link_elem in robot_elem.findall("link"):
            if link_elem.find("collision") is not None:
                links.append(link_elem.attrib["name"])
        print("Links with collision:\n")
        print('\n'.join([
                f"'{name}'," for name in links
                ]
            ))
        exit()
    # dof_choices = args.dof_choices
    base_link_name = args.base_link
    is_left = True
    if 'right' in input_fname:
        is_left = False
    print(f"Converting URDF, left hand: {is_left}.")
    new_joint_names = add_forearm_dof(
        input_fname, output_fname, dof_choices, base_link_name,
        left_hand=is_left, skip_side_prefix=args.skip_side_prefix
        )

    print(f"Successfully added {len(dof_choices)} new degree of freedom to the URDF file.\n Output file: {output_fname}\n New joint names: {new_joint_names}")
